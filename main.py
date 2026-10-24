"""
main.py - Main Entrypoint & Coordinator for Local Real-Time AI Interview & Meeting Copilot

Strict 8 GB RAM Architecture:
- Enforces `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_NUM_PARALLEL=1` on startup.
- Uses `faster-whisper` (`base.en`, `compute_type="int8"`, `device="cpu"`) + VAD for audio transcription.
- Indexes `resume.pdf` / `resume.txt` and `job_description.txt` locally using `rank_bm25` (< 5ms retrieval).
- Streams concise bullet-point answers and OCR coding solutions from `qwen2.5:3b` via Ollama
  with `stream=True`, `keep_alive=-1`, `num_ctx=2048`, `temperature=0.3`, `num_predict=350`.
- Keeps conversation history in a strict 5-turn rolling ring buffer and runs `gc.collect()` after OCR.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

try:
    import resource  # Unix/macOS only (not available on Windows)
except ImportError:
    resource = None  # type: ignore

# Enforce single-model concurrency before any Ollama interaction
os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
os.environ["OLLAMA_NUM_PARALLEL"] = "1"

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

from audio_transcriber import AudioTranscriber
from ocr_capture import OCRCapture
from ollama_client import DEFAULT_MODEL, OllamaClient, auto_select_best_coding_model
from overlay_ui import StealthOverlayUI
from rag_engine import RAGEngine

MAX_PYTHON_MEMORY_MB = 600.0


def get_process_memory_mb() -> float:
    """
    Return the current Python process Resident Set Size (RSS) in Megabytes (MB).
    Uses `psutil` if available (macOS, Windows, Linux), falling back to stdlib `resource.getrusage`.
    """
    if psutil is not None:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            # macOS reports ru_maxrss in bytes
            return float(usage) / (1024.0 * 1024.0)
        # Linux/BSD report ru_maxrss in kilobytes
        return float(usage) / 1024.0

    return 0.0


def read_clipboard_text() -> str:
    """
    Cross-platform clipboard text reader supporting macOS (`pbpaste`),
    Windows (`PyQt6` / PowerShell `Get-Clipboard` / `tkinter`), and Linux (`xclip` / `xsel`).
    """
    # 1. Try active PyQt6 QApplication clipboard first (works on macOS, Windows, Linux)
    try:
        from PyQt6.QtWidgets import QApplication  # type: ignore

        app = QApplication.instance()
        if app is not None:
            txt = app.clipboard().text()
            if txt and txt.strip():
                return txt.strip()
    except Exception:
        pass

    # 2. Platform-native CLI clipboard tools
    if sys.platform == "darwin":
        try:
            proc = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2.0, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass
    else:
        for cmd in (["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0, check=False)
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
            except Exception:
                continue

    # 3. Tkinter fallback
    try:
        import tkinter as tk

        r = tk.Tk()
        r.withdraw()
        txt = r.clipboard_get()
        r.destroy()
        if txt and str(txt).strip():
            return str(txt).strip()
    except Exception:
        pass

    return ""


class InterviewCopilotApp:
    """
    Coordinates AudioTranscriber, RAGEngine, OCRCapture, OllamaClient, and StealthOverlayUI
    under a strict 600 MB Python RSS ceiling (and 8 GB total system RAM budget).
    """

    def __init__(
        self,
        resume_path: Optional[str | Path] = None,
        jd_path: Optional[str | Path] = None,
        model: str = DEFAULT_MODEL,
        opacity: float = 0.86,
        ollama_client: Optional[OllamaClient] = None,
        rag_engine: Optional[RAGEngine] = None,
        transcriber: Optional[AudioTranscriber] = None,
        ocr_capture: Optional[OCRCapture] = None,
        auto_answer_questions: bool = True,
        enable_mic: bool = False,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self.auto_answer_questions = auto_answer_questions
        self.enable_mic = enable_mic
        self.ollama = ollama_client or OllamaClient(model=model, gemini_api_key=gemini_api_key)
        self.rag = rag_engine or RAGEngine()
        self.ocr = ocr_capture or OCRCapture()

        if resume_path or jd_path:
            self.rag.load_documents(
                resume_path=resume_path,
                job_description_path=jd_path,
            )

        self.transcriber = transcriber or AudioTranscriber(
            on_question_detected=self._on_question_detected,
            on_transcript_update=self._on_transcript_update,
            lazy_load_model=True,
        )

        self.overlay = StealthOverlayUI(
            on_trigger_answer=self.handle_hotkey_answer,
            on_screenshot_code=self.handle_hotkey_coding_ocr,
            on_clear_screen=self.handle_hotkey_clear,
            on_solve_clipboard=self.handle_hotkey_clipboard,
            on_answer_general=self.handle_hotkey_general_question,
            opacity=opacity,
        )

    def _on_transcript_update(self, latest_turn: str, _history: List[str]) -> None:
        self.overlay.set_transcript_preview(f"Heard: {latest_turn}")

    def _on_question_detected(self, question: str, history: List[str]) -> None:
        if not self.auto_answer_questions:
            return
        self.answer_question_to_overlay(question, transcript_history=history)

    def _capture_screen_text_cleanly(self) -> str:
        """
        Temporarily hide the stealth overlay (opacity 0.0), capture the screen via OCR,
        and immediately restore overlay visibility so it never screenshots itself.
        """
        self.overlay.set_capture_stealth(True)
        time.sleep(0.08)
        try:
            text = self.ocr.capture_and_extract_text()
        finally:
            self.overlay.set_capture_stealth(False)
            gc.collect()
        if text:
            one_line = " ".join(text.split())[:110]
            self.overlay.set_transcript_preview(f"Screen OCR: {one_line}...")
        return text

    def answer_question_stream(
        self,
        question: str,
        transcript_history: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Retrieve top-3 relevant resume chunks from local BM25 RAG and return
        the streaming response iterator from Ollama.
        """
        top_chunks = self.rag.retrieve(question, top_k=3)
        history = (
            transcript_history
            if transcript_history is not None
            else self.transcriber.get_recent_turns()
        )
        return self.ollama.stream_interview_answer(
            question=question,
            context_chunks=top_chunks,
            job_description=self.rag.job_description,
            transcript_history=history,
        )

    def answer_question_to_overlay(
        self,
        question: str,
        transcript_history: Optional[List[str]] = None,
    ) -> None:
        """Stream an interview answer directly to the stealth overlay UI."""
        if not question.strip():
            return
        header_q = " ".join(question.strip().split())[:90]
        stream = self.answer_question_stream(question, transcript_history=transcript_history)
        self.overlay.stream_to_overlay(f"Q: {header_q}", stream)

    def handle_hotkey_answer(self) -> None:
        """
        Handler for `Ctrl+Shift+A` ('Solve SQL'):
        In default screen-only mode, captures the SQL problem & table schema from the
        screen via OCR and streams an optimal SQL query (CTEs, window functions, edge cases).
        If `--enable-mic` is active and a spoken question was heard, answers via RAG.
        """
        if self.enable_mic:
            spoken_q = self.transcriber.get_latest_question_or_turn()
            if spoken_q:
                self.answer_question_to_overlay(spoken_q)
                return

        self.overlay.set_status("Capturing SQL problem & schema from screen via OCR...")
        screen_text = self._capture_screen_text_cleanly()
        problem_text, stream = self.ocr.capture_and_solve_coding_problem(
            self.ollama,
            fallback_problem_text=screen_text,
            mode="sql",
        )
        header = "Optimal SQL Solution (Screen OCR)" if problem_text else "OCR Notice"
        self.overlay.stream_to_overlay(header, stream)
        gc.collect()

    def handle_hotkey_coding_ocr(self) -> None:
        """
        Handler for `Ctrl+Shift+S` ('Solve DSA'):
        Hides overlay briefly, captures active DSA coding problem screenshot, extracts via OCR,
        runs explicit `gc.collect()`, and streams optimal code + pattern + complexity.
        """
        self.overlay.set_status("Capturing DSA coding problem from screen & running OCR...")
        screen_text = self._capture_screen_text_cleanly()
        problem_text, stream = self.ocr.capture_and_solve_coding_problem(
            self.ollama,
            fallback_problem_text=screen_text,
            mode="auto",
        )
        header = "Optimal DSA / Coding Solution (Screen OCR)" if problem_text else "OCR Notice"
        self.overlay.stream_to_overlay(header, stream)
        gc.collect()

    def handle_hotkey_clipboard(self) -> None:
        """
        Handler for `Ctrl+Shift+V` ('Solve Copied'):
        Solves the exact problem text copied to the clipboard (`Cmd+C` / `Ctrl+C`) with 100% text fidelity
        (zero OCR noise or missing symbols), falling back to screen OCR if clipboard is empty.
        """
        copied_text = read_clipboard_text()

        if not copied_text:
            self.handle_hotkey_coding_ocr()
            return

        one_line = " ".join(copied_text.split())[:110]
        self.overlay.set_transcript_preview(f"Copied Problem: {one_line}...")
        self.overlay.set_status("Solving copied problem from clipboard (100% exact text)...")
        problem_text, stream = self.ocr.capture_and_solve_coding_problem(
            self.ollama,
            fallback_problem_text=copied_text,
            mode="auto",
        )
        header = "Optimal Solution (Clipboard Text)" if problem_text else "Clipboard Notice"
        self.overlay.stream_to_overlay(header, stream)
        gc.collect()

    def handle_hotkey_general_question(self) -> None:
        """
        Handler for `Ctrl+Shift+G` ('Answer Any'):
        Captures the active screen via OCR and answers ANY question in front of the user
        (MCQ, conceptual/theory, system design, behavioral/interview, aptitude, or code/SQL).
        """
        self.overlay.set_status("Capturing question from screen via OCR...")
        screen_text = self._capture_screen_text_cleanly()
        cleaned_text = self.ocr.clean_coding_ocr_text(screen_text)
        if not cleaned_text:
            def _empty_stream() -> Iterator[str]:
                yield "No question text detected on screen. Please ensure the question is visible."
            self.overlay.stream_to_overlay("OCR Notice", _empty_stream())
            gc.collect()
            return

        top_chunks = self.rag.retrieve(cleaned_text, top_k=2) if self.rag.chunks else []
        stream = self.ollama.stream_general_question_answer(
            question_text=cleaned_text,
            context_chunks=top_chunks,
            job_description=self.rag.job_description,
        )
        self.overlay.stream_to_overlay("Answer (Screen Question)", stream)
        gc.collect()

    def handle_hotkey_clear(self) -> None:
        """Handler for `Ctrl+Shift+C`: clear transcript & overlay memory."""
        self.transcriber.clear_history()
        gc.collect()

    def start(self, headless: bool = False) -> None:
        """Start optional audio transcription and launch the stealth overlay UI."""
        if self.enable_mic:
            self.transcriber.start()
        try:
            if not headless:
                self.overlay.run()
        finally:
            if self.enable_mic:
                self.transcriber.stop()
            gc.collect()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local Real-Time AI Interview & Meeting Copilot (8 GB RAM Optimized)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="resume.txt",
        help="Path to resume.pdf or resume.txt (default: resume.txt)",
    )
    parser.add_argument(
        "--jd",
        type=str,
        default="job_description.txt",
        help="Path to job_description.txt (default: job_description.txt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        help="Model name (default: auto-selects gemini-3.8-flash if GEMINI_API_KEY is set, else qwen2.5-coder:3b)",
    )
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=None,
        help="Google AI Studio Gemini Free Tier API key (or set GEMINI_API_KEY in .env / environment)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.86,
        help="Stealth overlay window opacity between 0.25 and 1.0 (default: 0.86)",
    )
    parser.add_argument(
        "--enable-mic",
        action="store_true",
        help="Enable live microphone/audio transcription (disabled by default for screen-only OCR solving)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Initialize pipeline, verify memory footprint, and exit without launching GUI loop",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    resume_file = args.resume if os.path.exists(args.resume) else None
    if resume_file is None and os.path.exists("resume.pdf"):
        resume_file = "resume.pdf"

    jd_file = args.jd if os.path.exists(args.jd) else None
    resolved_model = (
        auto_select_best_coding_model(gemini_api_key=args.gemini_key)
        if args.model == "auto"
        else args.model
    )

    app = InterviewCopilotApp(
        resume_path=resume_file,
        jd_path=jd_file,
        model=resolved_model,
        opacity=args.opacity,
        enable_mic=args.enable_mic,
        gemini_api_key=args.gemini_key,
    )

    mem_mb = get_process_memory_mb()
    if args.headless:
        print(
            f"[Copilot Ready] Model={app.ollama.model} | "
            f"Indexed Chunks={len(app.rag.chunks)} | "
            f"Python RSS={mem_mb:.1f} MB (Limit: {MAX_PYTHON_MEMORY_MB:.0f} MB)"
        )
        return 0

    app.start(headless=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
