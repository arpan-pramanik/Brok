# Brok

voice-enabled RAG system with local-first inference, hybrid retrieval, calibrated guardrails, and circuit-breaker cloud fallback.

built for a hackathon, runs entirely on a single GPU machine. cloud (AWS) used surgically for fallback and benchmarking only.

## architecture

```
mic/text -> ASR (faster-whisper) -> Retrieval (hybrid dense+BM25, RRF, cross-encoder rerank)
         -> Guardrail (reranker-score gate) -> Generation (Qwen2.5-3B local, Bedrock fallback)
         -> Frontend (React, latency waterfall, confidence gauge)
```

### what each service does

| service | port | job |
|---------|------|-----|
| asr | 8001 | voice activity detection + streaming transcription |
| retrieval | 8002 | hybrid search (dense + sparse), RRF fusion, cross-encoder reranking |
| guardrail | 8003 | reranker-score-based abstention gate |
| generation | 8004 | local LLM (Qwen2.5-3B Q4_K_M) with circuit breaker to Bedrock |
| orchestrator | 8000 | async pipeline glue, REST + WebSocket API |
| frontend | 3000 | voice input, streaming transcript, latency waterfall, benchmark runner |

### retrieval pipeline

1. query hits dense (MiniLM) + sparse (BM25) search on Qdrant
2. results from multiple chunking strategies (fixed overlap, semantic, structural) fused via RRF
3. top candidates reranked by cross-encoder
4. guardrail checks top reranker score against calibrated threshold
5. if confident -> generate answer; if not -> abstain

### circuit breaker

local LLM (2 retries, exponential backoff) -> Bedrock fallback -> cached "unavailable" response. kill the GPU process mid-demo and watch it survive.

## setup

```bash
bash scripts/setup_all.sh
```

this will: check your GPU, download the model (~2GB), install deps, start Qdrant, and build the search index.

## run

### docker (recommended)
```bash
docker compose up
```

### local dev
```bash
# terminal 1: services
cd services/retrieval-service/src && uvicorn server:app --port 8002 &
cd services/guardrail-service/src && uvicorn server:app --port 8003 &
cd services/generation-service/src && uvicorn server:app --port 8004 &
cd services/orchestrator/src && uvicorn main:app --port 8000

# terminal 2: frontend
cd frontend && npm run dev
```

## benchmark

```bash
python benchmark/replay.py | python benchmark/report.py
```

generates P50/P70/P100 latency tables per stage.

## hardware requirements

- GPU: 8GB+ VRAM (tested on RTX 5070)
- RAM: 16GB+
- storage: ~5GB for models + deps

## corpus

10 documents about Goa covering geography, beaches, history, cuisine, architecture, wildlife, festivals, economy, transport, and practical travel info. swap with your own docs in `services/indexing-service/data/corpus/` and rebuild the index.

## AWS (optional, surgical)

- S3: corpus + artifact backup
- Bedrock: circuit-breaker fallback (Claude 3 Haiku)
- EC2: benchmark replay at higher concurrency

configure via `~/.aws/credentials` or env vars.
