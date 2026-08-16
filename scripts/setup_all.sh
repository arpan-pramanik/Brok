#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== bragi setup ==="

# check gpu
bash scripts/check_vram.sh
echo ""

# create models dir
mkdir -p models

# download qwen2.5-3b gguf if not present
MODEL_PATH="models/qwen2.5-3b-instruct-q4_k_m.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "downloading Qwen2.5-3B-Instruct Q4_K_M..."
    pip install -q huggingface-hub
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir models/
    echo "model downloaded"
else
    echo "model already exists at $MODEL_PATH"
fi

echo ""

# install python deps for local dev
echo "installing python dependencies..."
pip install -q \
    fastapi uvicorn httpx websockets pydantic \
    sentence-transformers qdrant-client \
    faster-whisper torch torchaudio \
    llama-cpp-python tenacity boto3 \
    scikit-learn numpy ruff pytest

echo ""

# install frontend deps
echo "installing frontend dependencies..."
cd frontend
npm ci
cd ..

echo ""

# start qdrant via docker
echo "starting qdrant..."
docker compose up -d qdrant redis

echo "waiting for qdrant to be ready..."
until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do
    sleep 1
done
echo "qdrant is ready"

echo ""

# build index
echo "building search index..."
cd services/indexing-service/src
python build_index.py
cd "$PROJECT_DIR"

echo ""
echo "=== setup complete ==="
echo "run services with: docker compose up"
echo "or run locally:"
echo "  cd services/orchestrator/src && uvicorn main:app --port 8000"
echo "  cd frontend && npm run dev"
