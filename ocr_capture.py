"""
ocr_capture.py - Zero-VLM Screen Capture & Lightweight OCR for Coding Problems (8 GB RAM Optimized)

Features:
- Captures the active screen / coding problem region via `mss`, `PIL.ImageGrab`, or OS-native
  `screencapture` (macOS) without loading any Vision-Language Model (VLM).
- Extracts coding problem text using `rapidocr-onnxruntime` (primary) or `pytesseract` /
  `tesseract` CLI binary (fallback).
- Immediately runs explicit memory cleanup (`del` raw pixel buffers + `gc.collect()`) after
  every OCR screenshot so large RGBA image arrays never accumulate in RAM.
- Streams an optimal coding solution + Time/Space complexity analysis from `OllamaClient`.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

try:
    import mss  # type: ignore
except ImportError:
    mss = None

try:
    from PIL import Image, ImageGrab  # type: ignore
except ImportError:
    Image = None
    ImageGrab = None

try:
    import pytesseract  # type: ignore
except ImportError:
    pytesseract = None

RapidOCR = None  # Lazily imported on demand to keep startup memory and dylib footprint minimal

if sys.platform == "win32":
    # Enable Per-Monitor DPI awareness on Windows 10/11 so `mss` and `PIL.ImageGrab` capture
    # the full native 1080p/1440p/4K screen without 125%/150% DPI scaling cropping.
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
    except Exception:
        pass


def _find_default_tesseract_cmd() -> Optional[str]:
    """Locate the `tesseract` binary across macOS, Windows, and Linux."""
    found = shutil.which("tesseract")
    if found:
        return found
    candidates = [
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


class OCRCapture:
    """
    Lightweight screen capture + OCR pipeline for LeetCode/HackerRank coding problems.
    Enforces strict post-capture garbage collection (`gc.collect()`) to stay well under 600 MB RSS.
    """

    def __init__(
        self,
        prefer_engine: str = "auto",
        release_engine_after_capture: bool = False,
        tesseract_cmd: Optional[str] = None,
    ) -> None:
        self.prefer_engine = prefer_engine.lower()
        self.release_engine_after_capture = release_engine_after_capture
        self.tesseract_cmd = tesseract_cmd or _find_default_tesseract_cmd()
        if pytesseract is not None and self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        self._rapid_ocr: Optional[Any] = None
        self.cleanup_count: int = 0

    def _get_rapid_ocr(self) -> Optional[Any]:
        """Lazily initialize `RapidOCR` (ONNX Runtime CPU) if available."""
        global RapidOCR
        if RapidOCR is None:
            try:
                from rapidocr_onnxruntime import RapidOCR as _RapidOCR  # type: ignore

                RapidOCR = _RapidOCR
            except ImportError:
                return None
        if self._rapid_ocr is None:
            try:
                self._rapid_ocr = RapidOCR(
                    use_cls=False,
                    det_limit_side_len=1920,
                    det_limit_type="max",
                )
            except TypeError:
                self._rapid_ocr = RapidOCR()
        return self._rapid_ocr

    def capture_screen_to_file(
        self,
        output_path: str | Path,
        region: Optional[Dict[str, int]] = None,
    ) -> Path:
        """
        Capture the active display (or bounding box `region={"top": y, "left": x, "width": w, "height": h}`)
        to a temporary PNG file and immediately free in-memory framebuffers.
        """
        target = Path(output_path)

        # 1. On macOS, prefer native `/usr/sbin/screencapture -x` first because it captures at
        # full 2x Retina resolution (2880x1800) with 0 MB Python RAM overhead, whereas `mss`
        # downsamples to 1x logical resolution (1440x900) on Retina displays.
        if sys.platform == "darwin":
            screencapture_bin = shutil.which("screencapture") or "/usr/sbin/screencapture"
            if os.path.exists(screencapture_bin):
                cmd = [screencapture_bin, "-x"]
                if region is not None:
                    rect = f"{region['left']},{region['top']},{region['width']},{region['height']}"
                    cmd.extend(["-R", rect])
                cmd.append(str(target))
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5.0,
                    )
                    if target.exists() and target.stat().st_size > 0:
                        gc.collect()
                        return target
                except Exception:
                    pass

        # 2. Fast cross-platform capture via `mss` (Windows, Linux, macOS fallback)
        if mss is not None:
            try:
                with mss.mss() as sct:
                    monitor = region if region is not None else sct.monitors[1]
                    sct_img = sct.grab(monitor)
                    mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(target))
                    del sct_img
                if target.exists() and target.stat().st_size > 0:
                    gc.collect()
                    return target
            except Exception:
                pass

        # 3. Pillow ImageGrab fallback (native GDI/Win32 capture on Windows)
        if ImageGrab is not None:
            bbox = None
            if region is not None:
                bbox = (
                    region["left"],
                    region["top"],
                    region["left"] + region["width"],
                    region["top"] + region["height"],
                )
            img = ImageGrab.grab(bbox=bbox)
            try:
                img.save(str(target), format="PNG")
            finally:
                img.close()
                del img
                gc.collect()
            return target

        raise RuntimeError(
            "No screen capture backend available. Install `mss` or `Pillow`."
        )

    @staticmethod
    def _extract_text_macos_vision(image_path: Path) -> str:
        """
        Use Apple's native Vision framework (`VNRecognizeTextRequest` Accurate level) on macOS.
        Runs on the Apple Neural Engine in ~300ms with 0 MB ONNX RAM overhead, preserves
        code/math symbols on Retina screenshots, and reconstructs multi-column layouts
        (e.g., problem statement on left + code editor on right) in natural reading order.
        """
        if sys.platform != "darwin":
            return ""
        try:
            import objc  # type: ignore
            from Foundation import NSURL  # type: ignore

            vision_ns: Dict[str, Any] = {}
            objc.loadBundle(
                "Vision",
                bundle_path="/System/Library/Frameworks/Vision.framework",
                module_globals=vision_ns,
            )
            VNImageRequestHandler = vision_ns.get("VNImageRequestHandler")
            VNRecognizeTextRequest = vision_ns.get("VNRecognizeTextRequest")
            if VNImageRequestHandler is None or VNRecognizeTextRequest is None:
                return ""

            is_fullscreen = False
            if Image is not None:
                try:
                    with Image.open(str(image_path)) as im:
                        w_px, h_px = im.size
                        is_fullscreen = w_px >= 1200 and h_px >= 750
                except Exception:
                    pass

            url = NSURL.fileURLWithPath_(str(image_path.resolve()))
            handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
            req = VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(0)  # 0 = VNRequestTextRecognitionLevelAccurate
            req.setUsesLanguageCorrection_(True)
            handler.performRequests_error_([req], None)

            observations = req.results() or []
            filtered = []
            for obs in observations:
                cands = obs.topCandidates_(1)
                if not cands:
                    continue
                text = str(cands[0].string()).strip()
                if not text:
                    continue
                bbox = obs.boundingBox()
                x = float(bbox.origin.x)
                y = float(bbox.origin.y)
                w = float(bbox.size.width)
                h = float(bbox.size.height)

                # Filter out macOS top menu bar (y > 0.962) and bottom Dock (y < 0.025) on full-screen captures
                if is_fullscreen and (y > 0.962 or y < 0.025):
                    continue
                # Skip narrow standalone editor gutter line numbers (1-3 digits)
                if w < 0.025 and text.isdigit() and len(text) <= 3:
                    continue
                filtered.append([x, y, w, h, text])

            if not filtered:
                return ""

            # Step 1: Sort top-to-bottom (descending y in Vision coords), then left-to-right,
            # and merge fragments on the same horizontal line that have a small gap (< 0.045)
            filtered.sort(key=lambda b: (-round(b[1] / 0.012), b[0]))
            merged = []
            for b in filtered:
                x, y, w, h, t = b
                placed = False
                for m in reversed(merged[-8:]):
                    mx, my, mw, mh, mt = m
                    if abs(y - my) <= max(h, mh) * 0.65 and -0.015 <= (x - (mx + mw)) < 0.045:
                        m[2] = max(mx + mw, x + w) - mx
                        m[3] = max(mh, h)
                        m[4] = f"{mt} {t}"
                        placed = True
                        break
                if not placed:
                    merged.append([x, y, w, h, t])

            # Step 2: Separate two-column split layouts (e.g., LeetCode problem on left < 0.46,
            # starter code editor on right >= 0.46) so columns never interleave line-by-line.
            left = [m for m in merged if m[0] < 0.46]
            right = [m for m in merged if m[0] >= 0.46]
            left.sort(key=lambda m: (-round(m[1] / 0.012), m[0]))
            right.sort(key=lambda m: (-round(m[1] / 0.012), m[0]))

            if left and right:
                return "\n".join([m[4] for m in left] + [m[4] for m in right]).strip()
            merged.sort(key=lambda m: (-round(m[1] / 0.012), m[0]))
            return "\n".join(m[4] for m in merged).strip()
        except Exception:
            return ""

    @staticmethod
    def _reconstruct_layout_from_rapidocr(result: Any, image_path: Path) -> str:
        """
        Reconstruct natural reading order and two-column split layouts (e.g., LeetCode/HackerRank
        problem statement on left + code editor on right) from `RapidOCR` bounding boxes on Windows/Linux.
        Also filters out top browser tab bars, bottom Windows taskbar, and editor gutter line numbers.
        """
        if not result:
            return ""
        w_px, h_px = 0, 0
        if Image is not None:
            try:
                with Image.open(str(image_path)) as im:
                    w_px, h_px = im.size
            except Exception:
                pass

        # Fallback: infer bounds from bounding boxes if Pillow couldn't open header
        if w_px <= 0 or h_px <= 0:
            max_x, max_y = 1.0, 1.0
            for item in result:
                if len(item) >= 2 and isinstance(item[0], (list, tuple)):
                    for pt in item[0]:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            max_x = max(max_x, float(pt[0]))
                            max_y = max(max_y, float(pt[1]))
            w_px, h_px = int(max_x), int(max_y)

        is_fullscreen = w_px >= 1200 and h_px >= 700
        filtered = []
        for item in result:
            if len(item) < 2 or not item[1]:
                continue
            text = str(item[1]).strip()
            if not text:
                continue
            box = item[0]
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                filtered.append([0.0, float(len(filtered)) * 0.02, 0.5, 0.015, text])
                continue
            xs = [float(pt[0]) for pt in box if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            ys = [float(pt[1]) for pt in box if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            if not xs or not ys or w_px <= 0 or h_px <= 0:
                continue
            x = min(xs) / float(w_px)
            y = min(ys) / float(h_px)  # Top-down normalized [0.0 = top, 1.0 = bottom]
            w = (max(xs) - min(xs)) / float(w_px)
            h = (max(ys) - min(ys)) / float(h_px)

            # Filter out top browser tab strip (y < 0.032) and bottom Windows Taskbar (y > 0.952)
            if is_fullscreen and (y < 0.032 or y > 0.952):
                continue
            # Skip narrow standalone editor gutter line numbers (1-3 digits)
            if w < 0.025 and text.isdigit() and len(text) <= 3:
                continue
            filtered.append([x, y, w, h, text])

        if not filtered:
            return ""

        # Step 1: Sort top-to-bottom (ascending y in RapidOCR coords), then left-to-right,
        # and merge horizontal fragments on the same line
        filtered.sort(key=lambda b: (round(b[1] / 0.012), b[0]))
        merged = []
        for b in filtered:
            x, y, w, h, t = b
            placed = False
            for m in reversed(merged[-8:]):
                mx, my, mw, mh, mt = m
                if abs(y - my) <= max(h, mh) * 0.65 and -0.015 <= (x - (mx + mw)) < 0.045:
                    m[2] = max(mx + mw, x + w) - mx
                    m[3] = max(mh, h)
                    m[4] = f"{mt} {t}"
                    placed = True
                    break
            if not placed:
                merged.append([x, y, w, h, t])

        # Step 2: Separate two-column split layouts (problem on left < 0.46, code editor on right >= 0.46)
        left = [m for m in merged if m[0] < 0.46]
        right = [m for m in merged if m[0] >= 0.46]
        left.sort(key=lambda m: (round(m[1] / 0.012), m[0]))
        right.sort(key=lambda m: (round(m[1] / 0.012), m[0]))

        if left and right:
            return "\n".join([m[4] for m in left] + [m[4] for m in right]).strip()
        merged.sort(key=lambda m: (round(m[1] / 0.012), m[0]))
        return "\n".join(m[4] for m in merged).strip()

    def extract_text_from_image(self, image_path: str | Path) -> str:
        """
        Run lightweight OCR (`Vision` on macOS, `rapidocr-onnxruntime` on Windows/Linux, or `tesseract`)
        on `image_path` and immediately garbage-collect intermediate buffers.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Screenshot file not found: {path}")

        extracted_text = ""
        try:
            # 0. On macOS, prefer native Apple Vision (`VNRecognizeTextRequest`) for fast,
            # column-aware Retina OCR with 0 MB ONNX RAM overhead.
            if sys.platform == "darwin" and self.prefer_engine in ("auto", "vision"):
                extracted_text = self._extract_text_macos_vision(path)
                if extracted_text:
                    return extracted_text

            # 1. Try rapidocr-onnxruntime if available and preferred (primary engine on Windows & Linux)
            if self.prefer_engine in ("auto", "rapidocr"):
                # Only use a subprocess on macOS when PyAV (`av`) is loaded in sys.modules
                # (avoids OpenMP dylib clash on macOS). On Windows, always run in-process (~0.35s).
                if sys.platform == "darwin" and "av" in sys.modules:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            (
                                "import json, sys; "
                                "from rapidocr_onnxruntime import RapidOCR; "
                                "ocr = RapidOCR(use_cls=False, det_limit_side_len=1920, det_limit_type='max'); "
                                "res, _ = ocr(sys.argv[1], use_cls=False); "
                                "lines = [str(x[1]).strip() for x in (res or []) if len(x) > 1 and x[1]]; "
                                "print(json.dumps('\\n'.join(lines).strip()))"
                            ),
                            str(path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=25.0,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        try:
                            extracted_text = json.loads(proc.stdout.strip().splitlines()[-1])
                        except Exception:
                            extracted_text = ""
                    del proc
                    if extracted_text:
                        return extracted_text

                engine = self._get_rapid_ocr()
                if engine is not None:
                    try:
                        result, _elapse = engine(str(path), use_cls=False)
                    except TypeError:
                        result, _elapse = engine(str(path))
                    if result:
                        extracted_text = self._reconstruct_layout_from_rapidocr(result, path)
                        if not extracted_text:
                            lines = [str(item[1]).strip() for item in result if len(item) > 1 and item[1]]
                            extracted_text = "\n".join(lines).strip()
                    del result
                    if self.release_engine_after_capture:
                        self._rapid_ocr = None
                    if extracted_text:
                        return extracted_text

            # 2. Try pytesseract Python wrapper if available
            if self.prefer_engine in ("auto", "tesseract", "pytesseract") and pytesseract is not None and Image is not None:
                with Image.open(str(path)) as pil_img:
                    extracted_text = pytesseract.image_to_string(pil_img).strip()
                if extracted_text:
                    return extracted_text

            # 3. Direct `tesseract` CLI binary invocation (zero Python memory overhead)
            if self.tesseract_cmd and os.path.exists(self.tesseract_cmd):
                proc = subprocess.run(
                    [self.tesseract_cmd, str(path), "stdout", "--oem", "1", "-l", "eng"],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if proc.returncode == 0 and proc.stdout:
                    extracted_text = proc.stdout.strip()
                del proc
                return extracted_text

            return extracted_text
        finally:
            self.cleanup_memory()

    @staticmethod
    def clean_coding_ocr_text(raw_text: str) -> str:
        """
        Strip common browser/IDE UI chrome noise and overlay self-text from OCR output
        while preserving DSA problem statements, constraints, starter signatures, MCQs,
        and SQL table schemas.
        """
        if not raw_text:
            return ""
        noise_exact = {
            "problems", "editorial", "solutions", "submissions", "description",
            "testcase", "test result", "run", "submit", "auto", "saved",
            "accepted", "companies", "hint 1", "hint 2", "hint 3", "topics",
            "copilot stealth overlay", "answer any", "answer q", "solve dsa",
            "solve sql", "solve copied", "copy code", "clear", "quit",
            "type here to search", "activate windows", "go to settings to activate windows.",
            "utf-8", "crlf", "lf",
        }
        noise_prefixes = (
            "https://",
            "http://",
            "leetcode.com",
            "hackerrank.com",
            "ready | ",
            "streaming from ",
            "solving ",
            "capturing ",
            "click 'answer any'",
            "dsa & sql screen mode",
            "screen ocr:",
            "optimal dsa / coding solution",
            "optimal sql solution",
            "answer (screen question)",
            "solve dsa (ctrl+shift",
            "solve sql (ctrl+shift",
            "chrome file edit view",
            "apple file edit view",
        )
        cleaned_lines = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            low = stripped.lower()
            if low in noise_exact or low.startswith(noise_prefixes):
                continue
            # Strip standalone code editor line numbers (e.g. "1", "2", "14")
            if stripped.isdigit() and len(stripped) <= 3:
                continue
            cleaned_lines.append(stripped)
        return "\n".join(cleaned_lines).strip()

    def capture_and_extract_text(
        self,
        region: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Capture a screenshot of the active screen/region, extract text via OCR,
        delete the temporary screenshot file, and force Python garbage collection.
        """
        tmp_fd, tmp_path_str = tempfile.mkstemp(prefix="copilot_ocr_", suffix=".png")
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        try:
            self.capture_screen_to_file(tmp_path, region=region)
            text = self.extract_text_from_image(tmp_path)
            return self.clean_coding_ocr_text(text)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            self.cleanup_memory()

    def capture_and_solve_coding_problem(
        self,
        ollama_client: Any,
        region: Optional[Dict[str, int]] = None,
        language: str = "Python",
        fallback_problem_text: Optional[str] = None,
        mode: str = "auto",
    ) -> Tuple[str, Iterator[str]]:
        """
        End-to-end handler for the `Ctrl+Shift+S` / `Ctrl+Shift+A` hotkeys:
        1. Captures the coding problem screenshot (or uses `fallback_problem_text` if provided in tests).
        2. Extracts problem text via OCR and cleans up image buffers via `gc.collect()`.
        3. Returns `(extracted_problem_text, solution_token_stream)` from `ollama_client`.
        """
        if fallback_problem_text is not None:
            problem_text = self.clean_coding_ocr_text(fallback_problem_text)
            self.cleanup_memory()
        else:
            problem_text = self.capture_and_extract_text(region=region)

        if not problem_text:
            def _empty_stream() -> Iterator[str]:
                yield "No coding problem text detected on screen. Please ensure the problem statement is visible."
            return "", _empty_stream()

        try:
            stream = ollama_client.stream_coding_solution(problem_text, language=language, mode=mode)
        except TypeError:
            stream = ollama_client.stream_coding_solution(problem_text, language=language)
        return problem_text, stream

    def cleanup_memory(self) -> None:
        """Explicitly run garbage collection after OCR operations to keep RAM < 600 MB."""
        if self.release_engine_after_capture:
            self._rapid_ocr = None
        gc.collect()
        self.cleanup_count += 1
