"""
overlay_ui.py - Always-On-Top, Semi-Transparent, Non-Focus-Stealing Stealth Overlay UI

Features:
- Dual backend support: `PyQt6` (with `WindowDoesNotAcceptFocus` + `WA_ShowWithoutActivating`)
  and `tkinter` (with `-topmost`, `-alpha`, and `overrideredirect` non-activating flags).
- Global hotkeys via `pynput.keyboard.GlobalHotKeys` (works without stealing focus from Zoom/Meet/Browser):
  - `Ctrl+Shift+A` : Trigger RAG-augmented interview answer from Ollama
  - `Ctrl+Shift+S` : Capture coding problem screenshot -> OCR -> Stream optimal solution & complexity
  - `Ctrl+Shift+C` : Clear overlay screen & run `gc.collect()`
  - `Ctrl+Shift+Q` : Quit copilot cleanly
- Thread-safe token queue (`queue.Queue`) for smooth, sub-second streaming display updates.
"""

from __future__ import annotations

import contextlib
import gc
import queue
import re
import signal
import subprocess
import sys
import threading
from typing import Callable, Dict, Iterator, List, Optional

try:
    from pynput import keyboard as pynput_keyboard  # type: ignore

    if sys.platform == "darwin":
        # CRITICAL macOS 15/26 fix: HIToolbox's TSMGetInputSourceProperty asserts that it is
        # called on the main dispatch queue whenever NSApplication (PyQt6) is active.
        # Pre-cache the keyboard layout context on the main thread so pynput's background
        # Listener thread never calls TSMGetInputSourceProperty off the main thread.
        import pynput._util.darwin as _darwin_util  # type: ignore
        import pynput.keyboard._darwin as _darwin_kb  # type: ignore

        with _darwin_util.keycode_context() as _MAIN_THREAD_KEYCODE_CTX:
            pass

        @contextlib.contextmanager
        def _main_thread_cached_keycode_context():
            yield _MAIN_THREAD_KEYCODE_CTX

        _darwin_util.keycode_context = _main_thread_cached_keycode_context
        _darwin_kb.keycode_context = _main_thread_cached_keycode_context
except ImportError:
    pynput_keyboard = None

try:
    from PyQt6.QtCore import Qt, QTimer  # type: ignore
    from PyQt6.QtGui import QColor, QFont, QPalette  # type: ignore
    from PyQt6.QtWidgets import (  # type: ignore
        QApplication,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

try:
    import tkinter as tk
    from tkinter import scrolledtext

    HAS_TKINTER = True
except ImportError:
    tk = None  # type: ignore
    scrolledtext = None  # type: ignore
    HAS_TKINTER = False


HOTKEY_ANSWER = "<ctrl>+<shift>+a"
HOTKEY_SCREENSHOT = "<ctrl>+<shift>+s"
HOTKEY_CLIPBOARD = "<ctrl>+<shift>+v"
HOTKEY_GENERAL_QUESTION = "<ctrl>+<shift>+g"
HOTKEY_CLEAR = "<ctrl>+<shift>+c"
HOTKEY_QUIT = "<ctrl>+<shift>+q"


def extract_latest_code_block(full_text: str) -> str:
    """Extract the first fenced ```...``` code or SQL block from a markdown response."""
    match = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", full_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return full_text.strip()


def _get_default_mono_font() -> str:
    """Return the native crisp monospace font family for the current OS."""
    if sys.platform == "win32":
        return "Consolas"
    if sys.platform == "darwin":
        return "Menlo"
    return "DejaVu Sans Mono"


def _apply_windows_stealth_flags(hwnd: int) -> None:
    """
    Apply Windows 10/11 native stealth flags via Win32 user32 APIs:
    1. `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE=0x00000011)`:
       Excludes the window from Google Meet, Zoom, Microsoft Teams, Discord, OBS,
       and Windows Snipping Tool / screen recordings (falls back to `WDA_MONITOR=0x00000001`).
    2. `WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE`:
       Hides the overlay from the Windows Taskbar and Alt+Tab switcher and prevents focus stealing.
    """
    if sys.platform != "win32" or not hwnd:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        WDA_MONITOR = 0x00000001
        WDA_EXCLUDEFROMCAPTURE = 0x00000011

        # Hide window from screen capture / screen sharing
        ok = user32.SetWindowDisplayAffinity(ctypes.wintypes.HWND(hwnd), WDA_EXCLUDEFROMCAPTURE)
        if not ok:
            user32.SetWindowDisplayAffinity(ctypes.wintypes.HWND(hwnd), WDA_MONITOR)

        # Hide from Windows Taskbar & Alt+Tab and prevent focus activation
        GWL_EXSTYLE = -20
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000
        get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        ex_style = get_long(hwnd, GWL_EXSTYLE)
        set_long(hwnd, GWL_EXSTYLE, ex_style | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
    except Exception:
        pass


class GlobalHotkeyManager:
    """
    System-wide global hotkey listener using `pynput` so hotkeys (`Ctrl+Shift+S`,
    `Ctrl+Shift+A`, `Ctrl+Shift+V`, `Ctrl+Shift+C`) trigger actions without ever focusing the overlay window.
    """

    def __init__(
        self,
        on_trigger_answer: Callable[[], None],
        on_screenshot_code: Callable[[], None],
        on_clear_screen: Callable[[], None],
        on_solve_clipboard: Optional[Callable[[], None]] = None,
        on_answer_general: Optional[Callable[[], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
    ) -> None:
        self.on_trigger_answer = on_trigger_answer
        self.on_screenshot_code = on_screenshot_code
        self.on_clear_screen = on_clear_screen
        self.on_solve_clipboard = on_solve_clipboard
        self.on_answer_general = on_answer_general
        self.on_quit = on_quit
        self._listener: Optional[object] = None

    def get_hotkey_map(self) -> Dict[str, Callable[[], None]]:
        mapping: Dict[str, Callable[[], None]] = {
            HOTKEY_ANSWER: self.on_trigger_answer,
            HOTKEY_SCREENSHOT: self.on_screenshot_code,
            HOTKEY_CLEAR: self.on_clear_screen,
        }
        if self.on_solve_clipboard is not None:
            mapping[HOTKEY_CLIPBOARD] = self.on_solve_clipboard
        if self.on_answer_general is not None:
            mapping[HOTKEY_GENERAL_QUESTION] = self.on_answer_general
        if self.on_quit is not None:
            mapping[HOTKEY_QUIT] = self.on_quit
        return mapping

    def start(self) -> bool:
        if pynput_keyboard is None:
            return False
        try:
            self._listener = pynput_keyboard.GlobalHotKeys(self.get_hotkey_map())
            self._listener.daemon = True  # type: ignore[attr-defined]
            self._listener.start()  # type: ignore[attr-defined]
            return True
        except Exception:
            self._listener = None
            return False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._listener = None


class StealthOverlayUI:
    """
    Lightweight, semi-transparent, always-on-top desktop overlay that streams tokens
    in real time without stealing window focus from the active interview or IDE window.
    """

    def __init__(
        self,
        on_trigger_answer: Optional[Callable[[], None]] = None,
        on_screenshot_code: Optional[Callable[[], None]] = None,
        on_clear_screen: Optional[Callable[[], None]] = None,
        on_solve_clipboard: Optional[Callable[[], None]] = None,
        on_answer_general: Optional[Callable[[], None]] = None,
        opacity: float = 0.86,
        width: int = 630,
        height: int = 410,
        x: int = 40,
        y: int = 40,
        prefer_backend: str = "auto",
    ) -> None:
        self.on_trigger_answer = on_trigger_answer or (lambda: None)
        self.on_screenshot_code = on_screenshot_code or (lambda: None)
        self.on_solve_clipboard = on_solve_clipboard or (lambda: None)
        self.on_answer_general = on_answer_general or (lambda: None)
        self._external_clear_callback = on_clear_screen
        self.opacity = max(0.25, min(1.0, opacity))
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.prefer_backend = prefer_backend.lower()

        # Thread-safe UI command queue: ("append" | "clear" | "status" | "transcript" | "stealth", payload)
        self._ui_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._current_text_chunks: List[str] = []
        self._status_text: str = "Ready | Answer Any: Ctrl+Shift+G | DSA: Ctrl+Shift+S | SQL: Ctrl+Shift+A"
        self._transcript_preview: str = "Click 'Answer Any' for any question on screen, or 'Solve DSA' / 'Solve SQL'."
        self._generation_lock = threading.Lock()

        self.hotkey_manager = GlobalHotkeyManager(
            on_trigger_answer=self.trigger_answer_hotkey,
            on_screenshot_code=self.trigger_screenshot_hotkey,
            on_clear_screen=self.clear_screen,
            on_solve_clipboard=self.trigger_clipboard_hotkey,
            on_answer_general=self.trigger_general_question_hotkey,
        )

    # ------------------------------------------------------------------
    # Thread-Safe Public API (callable from audio/OCR/LLM worker threads)
    # ------------------------------------------------------------------

    def set_status(self, status: str) -> None:
        self._status_text = status
        self._ui_queue.put(("status", status))

    def set_transcript_preview(self, transcript_line: str) -> None:
        self._transcript_preview = transcript_line
        self._ui_queue.put(("transcript", transcript_line))

    def set_capture_stealth(self, hidden: bool) -> None:
        """Temporarily make overlay transparent (0.0) during screen capture so it doesn't OCR itself."""
        self._ui_queue.put(("stealth", "1" if hidden else "0"))

    def append_token(self, token: str) -> None:
        self._current_text_chunks.append(token)
        self._ui_queue.put(("append", token))

    def clear_screen(self) -> None:
        self._current_text_chunks.clear()
        self._ui_queue.put(("clear", ""))
        if self._external_clear_callback is not None:
            self._external_clear_callback()
        gc.collect()

    def get_displayed_text(self) -> str:
        return "".join(self._current_text_chunks)

    def extract_latest_code_block(self) -> str:
        """Extract the first fenced ```...``` code or SQL block from the displayed response."""
        return extract_latest_code_block(self.get_displayed_text())

    def copy_code_to_clipboard(self) -> None:
        """Copy the extracted DSA code or SQL query directly to the clipboard without stealing focus."""
        code = self.extract_latest_code_block()
        if not code:
            self.set_status("Nothing to copy yet | Capture a DSA or SQL problem first")
            return
        try:
            if HAS_PYQT6 and QApplication.instance() is not None:
                QApplication.clipboard().setText(code)
                self.set_status("Copied code/SQL block to clipboard!")
                return
        except Exception:
            pass
        try:
            if sys.platform == "darwin":
                subprocess.run(["pbcopy"], input=code, text=True, timeout=2.0, check=False)
            elif sys.platform == "win32":
                subprocess.run(
                    ["clip"],
                    input=code,
                    text=True,
                    timeout=2.0,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except Exception:
            pass
        self.set_status("Copied solution to clipboard!")

    def stream_to_overlay(self, header: str, token_iterator: Iterator[str]) -> None:
        """
        Consume a streaming token iterator in a background thread and push tokens
        to the overlay UI in real time.
        """
        def _worker() -> None:
            with self._generation_lock:
                self._current_text_chunks.clear()
                self._ui_queue.put(("clear", ""))
                if header:
                    self.append_token(f"{header}\n{'─' * 48}\n")
                self.set_status("Streaming from Ollama (qwen2.5:3b | ctx=2048)...")
                try:
                    for token in token_iterator:
                        self.append_token(token)
                    self.set_status("Ready | Answer Any: Ctrl+Shift+G | DSA: Ctrl+Shift+S | SQL: Ctrl+Shift+A")
                except Exception as exc:
                    self.append_token(f"\n[Error]: {exc}")
                    self.set_status("Error during generation | Check Ollama daemon")
                finally:
                    gc.collect()

        threading.Thread(target=_worker, name="OverlayStreamWorker", daemon=True).start()

    def trigger_answer_hotkey(self) -> None:
        threading.Thread(target=self.on_trigger_answer, name="HotkeyAnswerTrigger", daemon=True).start()

    def trigger_screenshot_hotkey(self) -> None:
        threading.Thread(target=self.on_screenshot_code, name="HotkeyOCRTrigger", daemon=True).start()

    def trigger_clipboard_hotkey(self) -> None:
        threading.Thread(target=self.on_solve_clipboard, name="HotkeyClipboardTrigger", daemon=True).start()

    def trigger_general_question_hotkey(self) -> None:
        threading.Thread(target=self.on_answer_general, name="HotkeyGeneralQATrigger", daemon=True).start()

    # ------------------------------------------------------------------
    # GUI Backends (PyQt6 primary, tkinter fallback)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Launch the stealth overlay UI event loop and global hotkey listener."""
        self.hotkey_manager.start()
        try:
            if self.prefer_backend in ("auto", "pyqt6", "qt") and HAS_PYQT6:
                self._run_pyqt6()
            elif HAS_TKINTER:
                self._run_tkinter()
            else:
                raise RuntimeError("Neither PyQt6 nor tkinter is available in this Python environment.")
        finally:
            self.hotkey_manager.stop()

    def _run_pyqt6(self) -> None:
        app = QApplication.instance() or QApplication([])

        class _DraggableStealthWindow(QWidget):
            def __init__(self) -> None:
                super().__init__()
                self._drag_offset = None

            def mousePressEvent(self, event) -> None:  # type: ignore[override]
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    event.accept()

            def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
                if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
                    self.move(event.globalPosition().toPoint() - self._drag_offset)
                    event.accept()

            def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
                self._drag_offset = None

        window = _DraggableStealthWindow()
        window.setWindowTitle("Copilot Stealth Overlay")
        window.setGeometry(self.x, self.y, self.width, self.height)

        # Critical stealth flags: Always-On-Top, Frameless, Does Not Accept Focus.
        # Note: Do NOT set Qt.WindowType.Tool on macOS, as Tool windows set hidesOnDeactivate=YES
        # and disappear when switching apps or browser tabs.
        window.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        window.setWindowOpacity(self.opacity)
        window.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        palette = window.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(18, 20, 26))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(230, 236, 242))
        window.setPalette(palette)

        layout = QVBoxLayout(window)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        status_label = QLabel(self._status_text)
        status_label.setStyleSheet("color: #7ee787; font-size: 11px; font-weight: bold;")
        layout.addWidget(status_label)

        transcript_label = QLabel(self._transcript_preview)
        transcript_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        transcript_label.setWordWrap(True)
        layout.addWidget(transcript_label)

        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        text_area.setFont(QFont(_get_default_mono_font(), 11))
        text_area.setStyleSheet(
            "background-color: #161b22; color: #e6edf3; border: 1px solid #30363d; padding: 6px;"
        )
        layout.addWidget(text_area)

        btn_row = QHBoxLayout()
        for label, callback in (
            ("Answer Any", self.trigger_general_question_hotkey),
            ("Solve DSA", self.trigger_screenshot_hotkey),
            ("Solve SQL", self.trigger_answer_hotkey),
            ("Solve Copied", self.trigger_clipboard_hotkey),
            ("Copy Code", self.copy_code_to_clipboard),
            ("Clear", self.clear_screen),
            ("Quit", app.quit),
        ):
            btn = QPushButton(label)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(
                "background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; "
                "padding: 4px 7px; font-size: 11px;"
            )
            btn.clicked.connect(callback)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        def _poll_queue() -> None:
            try:
                while not self._ui_queue.empty():
                    try:
                        cmd, payload = self._ui_queue.get_nowait()
                    except queue.Empty:
                        break
                    if cmd == "append":
                        cursor = text_area.textCursor()
                        cursor.movePosition(cursor.MoveOperation.End)
                        cursor.insertText(str(payload))
                        text_area.setTextCursor(cursor)
                        text_area.ensureCursorVisible()
                    elif cmd == "clear":
                        text_area.clear()
                    elif cmd == "status":
                        status_label.setText(str(payload))
                    elif cmd == "transcript":
                        transcript_label.setText(str(payload))
                    elif cmd == "stealth":
                        window.setWindowOpacity(0.0 if payload == "1" else self.opacity)
            except KeyboardInterrupt:
                app.quit()
            except BaseException:
                pass

        try:
            signal.signal(signal.SIGINT, lambda *_: app.quit())
        except Exception:
            pass

        timer = QTimer(window)
        timer.timeout.connect(_poll_queue)
        timer.start(30)  # 30ms UI refresh for sub-second streaming smoothness

        window.show()

        if sys.platform == "darwin":
            try:
                import objc  # type: ignore
                from AppKit import (  # type: ignore
                    NSApp,
                    NSApplicationActivationPolicyAccessory,
                    NSStatusWindowLevel,
                    NSWindowCollectionBehaviorCanJoinAllSpaces,
                    NSWindowCollectionBehaviorFullScreenAuxiliary,
                    NSWindowCollectionBehaviorStationary,
                    NSWindowSharingNone,
                )

                # Hide Python icon from macOS Dock and Cmd+Tab switcher during screen share
                NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

                ns_view = objc.objc_object(c_void_p=int(window.winId()))
                ns_window = ns_view.window()
                if ns_window is not None:
                    # Hide window completely from Google Meet / Zoom / Teams screen sharing & recording
                    ns_window.setSharingType_(NSWindowSharingNone)
                    # Stick to screen when switching apps/tabs and across all full-screen Spaces
                    ns_window.setHidesOnDeactivate_(False)
                    ns_window.setLevel_(NSStatusWindowLevel)
                    ns_window.setCollectionBehavior_(
                        NSWindowCollectionBehaviorCanJoinAllSpaces
                        | NSWindowCollectionBehaviorFullScreenAuxiliary
                        | NSWindowCollectionBehaviorStationary
                    )
            except Exception:
                pass
        elif sys.platform == "win32":
            _apply_windows_stealth_flags(int(window.winId()))

        app.exec()

    def _run_tkinter(self) -> None:
        mono_font = _get_default_mono_font()
        root = tk.Tk()
        root.title("Copilot Stealth Overlay")
        root.geometry(f"{self.width}x{self.height}+{self.x}+{self.y}")
        root.configure(bg="#12141a")

        # Always on top, semi-transparent, frameless overlay without stealing focus
        root.wm_attributes("-topmost", True)
        try:
            root.wm_attributes("-alpha", self.opacity)
        except tk.TclError:
            pass
        root.overrideredirect(True)

        if sys.platform == "win32":
            root.update_idletasks()
            _apply_windows_stealth_flags(int(root.winfo_id()))

        status_var = tk.StringVar(value=self._status_text)
        status_lbl = tk.Label(
            root,
            textvariable=status_var,
            bg="#12141a",
            fg="#7ee787",
            font=(mono_font, 10, "bold"),
            anchor="w",
        )
        status_lbl.pack(fill=tk.X, padx=8, pady=(6, 2))

        transcript_var = tk.StringVar(value=self._transcript_preview)
        transcript_lbl = tk.Label(
            root,
            textvariable=transcript_var,
            bg="#12141a",
            fg="#8b949e",
            font=(mono_font, 10),
            anchor="w",
            wraplength=self.width - 20,
            justify=tk.LEFT,
        )
        transcript_lbl.pack(fill=tk.X, padx=8, pady=(0, 4))

        text_widget = scrolledtext.ScrolledText(
            root,
            bg="#161b22",
            fg="#e6edf3",
            insertbackground="#e6edf3",
            font=(mono_font, 11),
            wrap=tk.WORD,
            state=tk.DISABLED,
            takefocus=0,
            relief=tk.FLAT,
            padx=6,
            pady=6,
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        btn_frame = tk.Frame(root, bg="#12141a")
        btn_frame.pack(fill=tk.X, padx=8, pady=(2, 6))

        for btn_text, cmd_fn in (
            ("Answer Any [Ctrl+Shift+G]", self.trigger_general_question_hotkey),
            ("Solve DSA [Ctrl+Shift+S]", self.trigger_screenshot_hotkey),
            ("Solve SQL [Ctrl+Shift+A]", self.trigger_answer_hotkey),
            ("Clear [Ctrl+Shift+C]", self.clear_screen),
            ("Quit", root.destroy),
        ):
            btn = tk.Button(
                btn_frame,
                text=btn_text,
                command=cmd_fn,
                bg="#21262d",
                fg="#c9d1d9",
                activebackground="#30363d",
                relief=tk.FLAT,
                font=(mono_font, 9),
                takefocus=0,
            )
            btn.pack(side=tk.LEFT, padx=3)

        def _poll_tk_queue() -> None:
            while not self._ui_queue.empty():
                try:
                    cmd, payload = self._ui_queue.get_nowait()
                except queue.Empty:
                    break
                if cmd == "append":
                    text_widget.configure(state=tk.NORMAL)
                    text_widget.insert(tk.END, payload)
                    text_widget.see(tk.END)
                    text_widget.configure(state=tk.DISABLED)
                elif cmd == "clear":
                    text_widget.configure(state=tk.NORMAL)
                    text_widget.delete("1.0", tk.END)
                    text_widget.configure(state=tk.DISABLED)
                elif cmd == "status":
                    status_var.set(payload)
                elif cmd == "transcript":
                    transcript_var.set(payload)
                elif cmd == "stealth":
                    try:
                        root.wm_attributes("-alpha", 0.0 if payload == "1" else self.opacity)
                    except tk.TclError:
                        pass
            root.after(30, _poll_tk_queue)

        root.after(30, _poll_tk_queue)
        root.mainloop()
