#!/bin/bash
source .venv/bin/activate

export ABSTAIN_THRESHOLD=0.3
export RETRIEVAL_URL=http://localhost:8002
export GUARDRAIL_URL=http://localhost:8003
export GENERATION_URL=http://localhost:8004
export ASR_URL=http://localhost:8001
export TTS_URL=http://localhost:8005

echo "Starting Retrieval Service (8002)"
cd services/retrieval-service/src
PYTHONPATH=../../../ uvicorn server:app --port 8002 > /tmp/retrieval.log 2>&1 &
RETRIEVAL_PID=$!
cd ../../..

echo "Starting Guardrail Service (8003)"
cd services/guardrail-service/src
PYTHONPATH=../../../ uvicorn server:app --port 8003 > /tmp/guardrail.log 2>&1 &
GUARDRAIL_PID=$!
cd ../../..

echo "Starting Generation Service (8004)"
cd services/generation-service/src
PYTHONPATH=../../../ uvicorn server:app --port 8004 > /tmp/generation.log 2>&1 &
GENERATION_PID=$!
cd ../../..

echo "Starting TTS Service (8005)"
cd services/tts-service/src
PYTHONPATH=../../../ uvicorn server:app --port 8005 > /tmp/tts.log 2>&1 &
TTS_PID=$!
cd ../../..

echo "Starting ASR Service (8001)"
cd services/asr-service/src
PYTHONPATH=../../../ uvicorn stream_server:app --port 8001 > /tmp/asr.log 2>&1 &
ASR_PID=$!
cd ../../..

echo "Starting Orchestrator (8000)"
cd services/orchestrator/src
PYTHONPATH=../../../ uvicorn main:app --port 8000 > /tmp/orch.log 2>&1 &
ORCH_PID=$!
cd ../../..

echo "All services started! Press Ctrl+C to stop."
trap "kill $RETRIEVAL_PID $GUARDRAIL_PID $GENERATION_PID $ASR_PID $ORCH_PID $TTS_PID; exit" SIGINT SIGTERM
wait
