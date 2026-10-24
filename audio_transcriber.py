"""
audio_transcriber.py - Real-Time Dual-Source Audio Capture & faster-whisper CPU (int8) Transcriber

Features:
- Captures system audio loopback (Interviewer: BlackHole / WASAPI / Monitor) and microphone (Candidate).
- Uses `faster-whisper` with `model_size_or_path="base.en"`, `compute_type="int8"`, `device="cpu"`.
- Enables Voice Activity Detection (VAD) both via lightweight RMS energy gating and Silero `vad_filter=True`
  inside `faster-whisper` for low-latency question boundary detection.
- Maintains a strict rolling ring buffer of the last 5 conversation turns (`collections.deque(maxlen=5)`).
- Performs explicit garbage collection (`gc.collect()`) after audio buffer transcription.
"""

from __future__ import annotations

import collections
import gc
import math
import queue
import re
import struct
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None

try:
    import sounddevice as sd  # type: ignore
except ImportError:
    sd = None

WhisperModel = None  # Lazily imported on demand to keep startup memory and dylib footprint minimal

SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_MODEL_NAME = "base.en"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_DEVICE = "cpu"
MAX_TRANSCRIPT_TURNS = 5

INTERROGATIVE_PREFIXES = (
    "what ",
    "how ",
    "why ",
    "when ",
    "where ",
    "who ",
    "which ",
    "can you",
    "could you",
    "would you",
    "tell me",
    "walk me through",
    "describe ",
    "explain ",
    "have you ever",
    "do you have",
    "what's ",
    "how's ",
    "in your experience",
    "give me an example",
)


def compute_rms_energy(samples: Any) -> float:
    """
    Compute Root-Mean-Square (RMS) energy of a 1D float32 audio frame.
    Works with both `numpy.ndarray` and standard Python sequences/bytes.
    """
    if samples is None:
        return 0.0
    if np is not None and isinstance(samples, np.ndarray):
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
    if isinstance(samples, (bytes, bytearray)):
        count = len(samples) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"<{count}h", samples[: count * 2])
        mean_sq = sum((s / 32768.0) ** 2 for s in shorts) / count
        return math.sqrt(mean_sq)
    seq = list(samples)
    if not seq:
        return 0.0
    return math.sqrt(sum(float(x) ** 2 for x in seq) / len(seq))


def is_interview_question(text: str) -> bool:
    """
    Determine whether a transcribed utterance is likely an interview question.
    Checks for `?` punctuation or common behavioral/technical interview prompts.
    """
    cleaned = text.strip()
    if not cleaned or len(cleaned.split()) < 3:
        return False
    if "?" in cleaned:
        return True
    lowered = re.sub(r"^\[[^\]]+\]\s*", "", cleaned).strip().lower()
    return any(lowered.startswith(prefix) for prefix in INTERROGATIVE_PREFIXES)


class AudioTranscriber:
    """
    Real-time dual-channel (System Loopback + Microphone) audio transcriber powered by
    `faster-whisper` (`base.en`, `compute_type="int8"`, `device="cpu"`) with VAD and
    a strict 5-turn rolling transcript ring buffer.
    """

    def __init__(
        self,
        model_name: str = WHISPER_MODEL_NAME,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        cpu_threads: int = 4,
        vad_rms_threshold: float = 0.012,
        silence_duration_sec: float = 0.85,
        max_phrase_sec: float = 12.0,
        mic_device: Optional[int | str] = None,
        loopback_device: Optional[int | str] = None,
        on_question_detected: Optional[Callable[[str, List[str]], None]] = None,
        on_transcript_update: Optional[Callable[[str, List[str]], None]] = None,
        lazy_load_model: bool = True,
    ) -> None:
        if compute_type != "int8":
            raise ValueError("compute_type must be 'int8' to comply with the 8 GB RAM budget.")
        if device != "cpu":
            raise ValueError("device must be 'cpu' to avoid GPU/unified-memory contention with Ollama.")

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.vad_rms_threshold = vad_rms_threshold
        self.silence_duration_sec = silence_duration_sec
        self.max_phrase_sec = max_phrase_sec

        self.mic_device = mic_device
        self.loopback_device = loopback_device if loopback_device is not None else self.find_loopback_device()

        self.on_question_detected = on_question_detected
        self.on_transcript_update = on_transcript_update

        # Strict rolling ring buffer of the last 5 conversation turns
        self.transcript_history: Deque[str] = collections.deque(maxlen=MAX_TRANSCRIPT_TURNS)
        self.last_question: str = ""

        self._whisper_model: Optional[Any] = None
        self._model_lock = threading.Lock()
        self._audio_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=64)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._streams: List[Any] = []

        if not lazy_load_model:
            self._ensure_model_loaded()

    @staticmethod
    def find_loopback_device() -> Optional[int]:
        """
        Auto-detect system audio loopback device (e.g., BlackHole 2ch on macOS,
        Stereo Mix / WASAPI Loopback on Windows, Monitor of Built-in Audio on Linux).
        """
        if sd is None:
            return None
        try:
            devices = sd.query_devices()
            loopback_keywords = ("blackhole", "loopback", "soundflower", "stereo mix", "monitor", "vb-audio")
            for idx, dev in enumerate(devices):
                name = str(dev.get("name", "")).lower()
                max_in = int(dev.get("max_input_channels", 0))
                if max_in > 0 and any(kw in name for kw in loopback_keywords):
                    return idx
        except Exception:
            pass
        return None

    def _ensure_model_loaded(self) -> Any:
        """Lazily initialize `faster-whisper` (`base.en`, `int8`, `cpu`)."""
        global WhisperModel
        with self._model_lock:
            if WhisperModel is None:
                try:
                    from faster_whisper import WhisperModel as _WhisperModel  # type: ignore

                    WhisperModel = _WhisperModel
                except ImportError:
                    return None
            if self._whisper_model is None and WhisperModel is not None:
                self._whisper_model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                )
            return self._whisper_model

    def record_turn(
        self,
        text: str,
        speaker: str = "Interviewer",
        trigger_callbacks: bool = True,
    ) -> Optional[str]:
        """
        Append a transcribed turn into the 5-turn rolling ring buffer.
        If the utterance is a question, updates `self.last_question` and fires `on_question_detected`.
        """
        cleaned = text.strip()
        if not cleaned:
            return None

        formatted_turn = f"[{speaker}] {cleaned}" if not cleaned.startswith("[") else cleaned
        self.transcript_history.append(formatted_turn)
        history_snapshot = list(self.transcript_history)

        if trigger_callbacks and self.on_transcript_update is not None:
            self.on_transcript_update(formatted_turn, history_snapshot)

        if is_interview_question(cleaned):
            self.last_question = cleaned
            if trigger_callbacks and self.on_question_detected is not None:
                self.on_question_detected(cleaned, history_snapshot)

        return formatted_turn

    def get_recent_turns(self) -> List[str]:
        """Return a list of up to the last 5 transcript turns from the ring buffer."""
        return list(self.transcript_history)

    def get_latest_question_or_turn(self) -> str:
        """
        Return the most recent detected question, or the latest turn in the ring buffer
        when the user manually triggers answer generation via hotkey.
        """
        if self.last_question:
            return self.last_question
        if self.transcript_history:
            latest = self.transcript_history[-1]
            return re.sub(r"^\[[^\]]+\]\s*", "", latest).strip()
        return ""

    def clear_history(self) -> None:
        """Clear the rolling 5-turn ring buffer and run garbage collection."""
        self.transcript_history.clear()
        self.last_question = ""
        gc.collect()

    def transcribe_audio_segment(
        self,
        audio_buffer: Any,
        speaker: str = "Interviewer",
    ) -> str:
        """
        Transcribe a 16kHz float32 audio segment using `faster-whisper` (`base.en`, `int8`, `cpu`)
        with Silero VAD filtering (`vad_filter=True`), then immediately free audio buffers.
        """
        try:
            # Lightweight RMS VAD gate: skip silent buffers before invoking Whisper
            rms = compute_rms_energy(audio_buffer)
            if rms < self.vad_rms_threshold:
                return ""

            model = self._ensure_model_loaded()
            if model is None:
                return ""

            segments, _info = model.transcribe(
                audio_buffer,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 400,
                    "speech_pad_ms": 200,
                },
            )
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
            if text:
                self.record_turn(text, speaker=speaker, trigger_callbacks=True)
            return text
        finally:
            del audio_buffer
            gc.collect()

    def _create_stream_callback(self, speaker_label: str) -> Callable[[Any, int, Any, Any], None]:
        """Create a non-blocking `sounddevice.InputStream` callback for a given audio source."""

        def _callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            if not self._running:
                return
            try:
                frame_copy = indata[:, 0].copy() if hasattr(indata, "copy") else bytes(indata)
                self._audio_queue.put_nowait((speaker_label, frame_copy))
            except queue.Full:
                # Drop oldest frame if queue is saturated to preserve low latency & bounded RAM
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait((speaker_label, frame_copy))
                except queue.Empty:
                    pass

        return _callback

    def _processing_loop(self) -> None:
        """
        Background worker that accumulates speech frames per source using RMS VAD,
        flushes phrases on pause (`silence_duration_sec`) or `max_phrase_sec`, and transcribes.
        """
        buffers: Dict[str, List[Any]] = {"Interviewer": [], "Candidate": []}
        speaking_state: Dict[str, bool] = {"Interviewer": False, "Candidate": False}
        silence_start: Dict[str, float] = {"Interviewer": 0.0, "Candidate": 0.0}
        phrase_start: Dict[str, float] = {"Interviewer": 0.0, "Candidate": 0.0}

        while self._running:
            try:
                speaker, frame = self._audio_queue.get(timeout=0.15)
            except queue.Empty:
                # Check if any active speech buffer has timed out into silence
                now = time.monotonic()
                for spk in ("Interviewer", "Candidate"):
                    if speaking_state[spk] and buffers[spk]:
                        if (now - silence_start[spk]) >= self.silence_duration_sec:
                            self._flush_speaker_buffer(spk, buffers[spk])
                            buffers[spk] = []
                            speaking_state[spk] = False
                continue

            now = time.monotonic()
            rms = compute_rms_energy(frame)

            if rms >= self.vad_rms_threshold:
                if not speaking_state[speaker]:
                    speaking_state[speaker] = True
                    phrase_start[speaker] = now
                silence_start[speaker] = now
                buffers[speaker].append(frame)

                if (now - phrase_start[speaker]) >= self.max_phrase_sec:
                    self._flush_speaker_buffer(speaker, buffers[speaker])
                    buffers[speaker] = []
                    speaking_state[speaker] = False
            elif speaking_state[speaker]:
                buffers[speaker].append(frame)
                if (now - silence_start[speaker]) >= self.silence_duration_sec:
                    self._flush_speaker_buffer(speaker, buffers[speaker])
                    buffers[speaker] = []
                    speaking_state[speaker] = False

    def _flush_speaker_buffer(self, speaker: str, frames: List[Any]) -> None:
        """Concatenate buffered frames for `speaker`, transcribe, and free memory."""
        if not frames or np is None:
            return
        try:
            combined = np.concatenate(frames, axis=0).astype(np.float32, copy=False)
            self.transcribe_audio_segment(combined, speaker=speaker)
        except Exception:
            pass
        finally:
            frames.clear()
            gc.collect()

    def start(self) -> bool:
        """
        Start background audio capture streams (System Loopback + Microphone) and worker thread.
        Returns True if live hardware capture started, or False if running in manual/text-only mode.
        """
        if self._running:
            return True

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._processing_loop,
            name="AudioTranscriberWorker",
            daemon=True,
        )
        self._worker_thread.start()

        if sd is None or np is None:
            return False

        started_any = False
        blocksize = int(SAMPLE_RATE * 0.1)  # 100ms low-latency frames

        # 1. Start Microphone Stream (Candidate)
        try:
            mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=blocksize,
                device=self.mic_device,
                channels=CHANNELS,
                dtype="float32",
                callback=self._create_stream_callback("Candidate"),
            )
            mic_stream.start()
            self._streams.append(mic_stream)
            started_any = True
        except Exception:
            pass

        # 2. Start System Audio Loopback Stream (Interviewer) if loopback device is available
        if self.loopback_device is not None:
            try:
                loop_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=blocksize,
                    device=self.loopback_device,
                    channels=CHANNELS,
                    dtype="float32",
                    callback=self._create_stream_callback("Interviewer"),
                )
                loop_stream.start()
                self._streams.append(loop_stream)
                started_any = True
            except Exception:
                pass

        return started_any

    def stop(self) -> None:
        """Stop all audio streams, join worker thread, and clean up buffers."""
        self._running = False
        for stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._streams.clear()

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        self._worker_thread = None

        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        gc.collect()
