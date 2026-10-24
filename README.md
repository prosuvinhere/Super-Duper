# Super-Duper — Stealth AI Coding & Interview Copilot (macOS & Windows)

A lightweight, always-on-top, screen-share-invisible AI Interview & Coding Copilot built for **8 GB RAM laptops** on **macOS** and **Windows 10/11**.

## Key Features
- **Screen-Share Invisible (Stealth Overlay)**:
  - **macOS**: Uses `NSWindowSharingNone` + `NSApplicationActivationPolicyAccessory` (hidden from Google Meet, Zoom, Teams screen sharing, Dock, and `Cmd+Tab`).
  - **Windows 10/11**: Uses Win32 `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` + `WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE` (hidden from Google Meet, Zoom, Teams, OBS, Snipping Tool, Taskbar, and `Alt+Tab`).
- **Answers Anything on Screen**:
  - **Answer Any (`Ctrl+Shift+G`)**: Solves MCQs, conceptual/theory questions, system design, behavioral questions, math/aptitude, or code on screen.
  - **Solve DSA (`Ctrl+Shift+S`)**: Captures LeetCode/HackerRank DSA problems via OCR and returns optimal runnable code + complexity.
  - **Solve SQL (`Ctrl+Shift+A`)**: Captures SQL schema & problem tables via OCR and writes runnable PostgreSQL/MySQL queries.
  - **Solve Copied (`Ctrl+Shift+V`)**: Solves text copied to your clipboard (`Cmd+C` / `Ctrl+C`) with 100% text fidelity.
- **Dual AI Engine**:
  - **Google Gemini Free Tier API** (`gemini-3.8-flash` -> `gemini-3.6-flash` -> `gemini-3.5-flash` -> `gemini-3.5-flash-lite` automatic rollover, 0 MB local LLM RAM).
  - **Offline Local Ollama Fallback** (`qwen2.5-coder:3b` / `qwen2.5:3b`, strictly `<= 3B` models for 8 GB RAM safety).
- **Local Reference Banks**:
  - Automatically indexes [`dsa_reference.txt`](./dsa_reference.txt) and [`sql_reference.txt`](./sql_reference.txt) alongside [`resume.txt`](./resume.txt) and [`job_description.txt`](./job_description.txt) using sub-5ms BM25 retrieval.

---

## Quick Start — Windows 10 / 11

1. **Clone the repository**:
   ```cmd
   git clone https://github.com/prosuvinhere/Super-Duper.git
   cd Super-Duper
   ```
2. **Run the Windows setup script**:
   ```cmd
   setup.bat
   ```
3. **Add your Gemini API Key**:
   Open `.env` in Notepad and paste your key:
   ```env
   GEMINI_API_KEY="your_actual_gemini_api_key"
   ```
4. **Launch the Copilot**:
   ```cmd
   run.bat
   ```
   *(Or run `.venv\Scripts\python.exe main.py`)*

---

## Quick Start — macOS

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prosuvinhere/Super-Duper.git
   cd Super-Duper
   ```
2. **Run the macOS setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. **Add your Gemini API Key**:
   Create or edit `.env`:
   ```bash
   cp -n .env.example .env
   ```
   Put your key inside `.env`:
   ```env
   GEMINI_API_KEY="your_actual_gemini_api_key"
   ```
4. **Launch the Copilot**:
   ```bash
   .venv/bin/python main.py
   ```

---

## Controls & Global Hotkeys

| Button | Global Hotkey | Description |
| :--- | :--- | :--- |
| **`Answer Any`** | `Ctrl + Shift + G` | Captures screen via OCR and answers **any** question (MCQ, theory, system design, behavioral, code) |
| **`Solve DSA`** | `Ctrl + Shift + S` | Captures DSA coding problem from screen and generates runnable code + Big-O complexity |
| **`Solve SQL`** | `Ctrl + Shift + A` | Captures SQL schema/problem from screen and generates an optimal SQL query |
| **`Solve Copied`** | `Ctrl + Shift + V` | Solves problem text copied to clipboard (`Cmd+C` / `Ctrl+C`) with zero OCR noise |
| **`Copy Code`** | *(Click button)* | Extracts only the fenced code/SQL block and copies it to clipboard |
| **`Clear`** | `Ctrl + Shift + C` | Clears the overlay text and runs garbage collection |
| **`Quit`** | `Ctrl + Shift + Q` | Cleanly exits the copilot |

---

## Running Tests

```bash
# macOS / Linux
.venv/bin/python -m pytest -v test_copilot.py

# Windows
.venv\Scripts\python.exe -m pytest -v test_copilot.py
```
