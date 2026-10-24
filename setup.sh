#!/usr/bin/env bash
# setup.sh - Setup & Startup Script for 8 GB RAM AI Interview Copilot (macOS / Linux)
set -euo pipefail

# 1. Enforce strict Single-Model & Single-Parallelism settings for 8 GB RAM
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1

MODEL_NAME="${1:-qwen2.5:3b}"
OLLAMA_HOST_URL="http://localhost:11434"

echo "============================================================"
echo " AI Interview & Meeting Copilot Setup (8 GB RAM Optimized)"
echo " OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}"
echo " OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
echo " Target Model: ${MODEL_NAME}"
echo "============================================================"

# 2. Check if Ollama CLI is installed
if ! command -v ollama >/dev/null 2>&1; then
    echo "[ERROR] 'ollama' CLI not found. Install Ollama from https://ollama.com first."
    exit 1
fi

# 3. Check if Ollama server is running; start it with memory flags if not
if ! curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
    echo "[INFO] Ollama server is not running. Starting 'ollama serve' with 8 GB RAM flags..."
    nohup ollama serve >/tmp/ollama_copilot.log 2>&1 &
    for i in {1..15}; do
        if curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
            echo "[OK] Ollama server is up and running."
            break
        fi
        sleep 1
    done
else
    echo "[OK] Ollama server is already running at ${OLLAMA_HOST_URL}."
fi

# 4. Pull the lightweight 3B model (qwen2.5:3b by default)
echo "[INFO] Ensuring model '${MODEL_NAME}' is pulled..."
ollama pull "${MODEL_NAME}"

# 5. Pre-warm and pin model in RAM with keep_alive: -1 and num_ctx: 2048
echo "[INFO] Pinning '${MODEL_NAME}' in RAM (keep_alive: -1, num_ctx: 2048)..."
curl -fsS "${OLLAMA_HOST_URL}/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"ready\"}],
        \"stream\": false,
        \"keep_alive\": -1,
        \"options\": {\"num_ctx\": 2048, \"temperature\": 0.3, \"num_predict\": 1}
    }" >/dev/null 2>&1 || true

# 6. Optional Python virtual environment setup
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing Python dependencies from requirements.txt..."
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

echo "[SUCCESS] Setup complete! Run: .venv/bin/python main.py --resume resume.txt --jd job_description.txt"
