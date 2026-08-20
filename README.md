# Brok

Ultra-fast voice-enabled Pokédex RAG system delivering end-to-end question answering in under 120 milliseconds.

## Why I Built This

Most voice AI pipelines feel clunky and sluggish because they chain together high-latency cloud APIs, taking 2 to 4 seconds just to hear back. I wanted to see how fast an end-to-end voice RAG system could actually get if every millisecond was treated as a hard budget — combining local in-memory embeddings, sub-2ms vector retrieval in Rust, Groq LPU hardware inference, Sarvam voice-to-text, and multi-tier guardrails.

---

## Architecture

```
User Voice / Text Query
          │
          ├─── [Sarvam AI ASR (saarika:v2.5) / Web Speech API] ───► Clean Transcript
          │
          ▼
Orchestration Harness (Rust / Axum on Port 8000)
          │
          ├───► 1. Safety Guardrail (<0.2ms Pattern & Policy Scan)
          │
          ├───► 2. Retrieval Service (Rust / FastEmbed ONNX on Port 8002)
          │          │
          │          ├── Multi-Strategy Dynamic Chunking & Dual-Query Normalization
          │          ├── Batch Dense Embedding (bge-small-en-v1.5 in ~1.8ms)
          │          └── Qdrant Vector Search (msmarco_xi Collection in ~1.0ms)
          │
          ├───► 3. Context Relevance Gate (Vector Score Calibrated Threshold)
          │
          ├───► 4. Speculative Model Harness Race (tokio::select!)
          │          ├── Groq Primary LPU (allam-2-7b / gpt-oss-20b)
          │          ├── Groq Secondary LPU (Failover)
          │          └── OpenRouter Fallback
          │
          └───► 5. Hallucination Guardrail (<0.2ms Grounding Token Verification)
          │
          ▼
Streamed CRT Output + Audio TTS to Pokédex Client (<120ms Total)
```

---

## 1. Speech-to-Text (Sarvam AI)

Voice transcription is powered by **Sarvam AI (`saarika:v2.5`)** with low-latency client-side streaming:
- **Audio Capture**: Browser records 16kHz mono Float32 PCM audio directly from the microphone.
- **In-Memory WAV Encoder**: On button release, client packs the audio into an in-memory 16kHz mono WAV Blob in under 1ms.
- **Direct HTTPS API**: Audio is sent to the backend `/api/transcribe` endpoint, routing directly to Sarvam STT.
- **Instant Visual Feedback**: Browser-native Web Speech API streams real-time partial words on the CRT display while holding the button, eliminating perceived speech latency.

---

## 2. Dynamic Multi-Strategy Chunking

Rather than arbitrary fixed-size character slicing that splits words and breaks semantic continuity, Brok uses a multi-strategy dynamic chunking pipeline tailored to the `AI4Bharat/MSMARCO-XI` dataset:

- **Semantic Boundary Passage Chunking**: Splits text along natural sentence, clause, and grammatical discourse boundaries, ensuring each chunk contains a complete, self-contained factual proposition.
- **Structural Markdown Hierarchy Splitting**: For long-form documents, text is segmented on header boundaries (`#`, `##`, `###`) and logical paragraph breaks (`\n\n`), preventing cross-topic context pollution.
- **Metadata-Aware Payload Enrichment**: Every chunk is indexed with rich structural metadata:
  - `query_id`: Parent cluster identifier linking passages to their source queries.
  - `language`: Source language code (`en` or native Indic language code).
  - `chunk_index`: Position index within the parent document.
  - `is_selected`: Supervision ground-truth relevance flag.
  - `source_doc`: Source document lineage.
  - `length`: Character and token density metrics.
- **Dense Token Optimization**: Chunks are sized to fit the exact 384-token receptive field of `bge-small-en-v1.5`, maximizing vector embedding fidelity without wasting sparse vector capacity.

---

## 3. Latency Analytics & Performance Metrics

Latency numbers measured across **225 real test queries** running against the live AWS EC2 backend with 1,997 MSMARCO dataset vectors in Qdrant (recorded in `extreme_benchmark_output.json`):

| Pipeline Stage | P50 (Median) | P70 | P95 | P100 (Max) | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Safety Guardrail** | < 0.1 ms | < 0.1 ms | 0.2 ms | 0.4 ms | 0.1 ms |
| **Embedding (FastEmbed ONNX)** | 1.8 ms | 2.1 ms | 2.6 ms | 3.8 ms | 1.9 ms |
| **Vector Retrieval (Qdrant)** | 0.9 ms | 1.2 ms | 2.1 ms | 3.5 ms | 1.1 ms |
| **LLM Generation (Groq LPU)** | 102.0 ms | 104.0 ms | 119.1 ms | 192.0 ms | 87.1 ms |
| **Hallucination Verification** | 0.1 ms | 0.2 ms | 0.3 ms | 0.5 ms | 0.2 ms |
| **Total End-to-End Latency** | **102.4 ms** | **106.0 ms** | **117.5 ms** | **192.7 ms** | **87.8 ms** |

*All 225 benchmarked cloud runs completed well within the 200ms target budget.*

### Cloud vs. Local GPU (NVIDIA RTX 5070) Benchmark

When running the pipeline completely on-device using a dedicated local GPU (**NVIDIA GeForce RTX 5070 8GB**) with quantized local models (`Qwen2.5-3B-Instruct Q4_K_M` via `llama.cpp` CUDA backend), network RTT is completely eliminated, yielding even lower processing latency:

| Deployment Mode | Retrieval | TTFT | Generation (Avg) | Total E2E Latency (P50) | P95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cloud Deployment (AWS EC2 + Groq LPU)** | 2.7 ms | 68.0 ms | 34.0 ms | **102.4 ms** | **117.5 ms** |
| **Local GPU (NVIDIA RTX 5070, Qwen2.5-3B Q4)** | 1.4 ms | 28.5 ms | 24.2 ms | **54.1 ms** | **68.3 ms** |

---

## 4. Orchestration Harness

Brok does not use a single raw prompt-in, text-out call. The entire pipeline executes inside a structured **`OrchestrationHarness`** in Rust (`services/orchestrator/src/harness.rs`):

- **Structured Tool Execution**: Discrete tool wrappers (`vector_search_tool`, `tts_synthesis_tool`, `audit_tool`) with standardized input/output contracts and execution telemetry.
- **Automatic Retries with Backoff**: Tool calls automatically retry transient network failures up to 2 times before failing gracefully.
- **Speculative Multi-Provider Racing**: Non-blocking `tokio::select!` races primary Groq LPU, secondary Groq LPU, and OpenRouter streams in parallel. The fastest valid stream wins, canceling remaining requests instantly.
- **Circuit-Breaker Error Recovery**: If an engine returns an error or rate limit, the harness falls back to local candidate extraction and cleanly terminates streams without hanging client UI.

---

## 5. Multi-Stage Guardrail Engine

The system knows when **not** to answer through three real-time guardrail gates (`services/orchestrator/src/guardrails.rs`):

1. **Input Safety & Policy Guardrail (<0.1ms)**: Scans incoming queries for dangerous intent, prompt injection, and prohibited patterns with medical/technical exemption allowlists. Unsafe inputs are refused before vector search runs.
2. **Context Relevance Gate (Sub-1ms)**: Evaluates top vector similarity scores against a calibrated threshold (`0.30`). If the question is off-topic or ungrounded in the dataset, the system cleanly abstains (*"couldnt locate in the dataset."*) without burning LLM tokens.
3. **Hallucination Verification (<0.2ms)**: Analyzes generated output tokens against retrieved context passages. If the ungrounded content token ratio exceeds 80%, the answer is intercepted and replaced with an abstention notice.

---

## Repository Structure

```
.
├── frontend/                     # React + Vite Pokédex retro interface
│   ├── src/                      # App components, audio waveform visualizer, sound effects
│   └── vercel.json               # Edge proxy configuration for live deployment
├── services/
│   ├── orchestrator/             # Rust async pipeline coordinator, guardrails, and harness
│   ├── retrieval-service/        # Rust FastEmbed ONNX dense vector search engine
│   ├── asr-service/              # Rust voice audio transcription & Sarvam bridge
│   └── indexing-service/         # Dataset ingestion scripts for MSMARCO / custom corpus
├── scripts/                      # AWS EC2 deployment and provisioning scripts
├── docker-compose.yml            # Multi-service container definitions
├── ARCHITECTURE.md               # Deep dive on technical choices and trade-offs
└── CONTRIBUTING.md               # Guidelines for local setup and pull requests
```

---

## Setup in Under 5 Minutes

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for frontend)
- Rust toolchain (optional, for local non-containerized dev)

### 1. Clone & Configure Environment
```bash
git clone https://github.com/arpan-pramanik/Brok.git
cd Brok
cp .env.example .env
```
Add your API keys to `.env`:
```ini
GROQ_API_KEY=your_groq_api_key
GROQ_API_KEY_SECONDARY=your_backup_groq_key
OPENROUTER_API_KEY=your_openrouter_key
SARVAM_API_KEY=your_sarvam_api_key
```

### 2. Run with Docker Compose
```bash
docker compose up -d --build
```

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` to interact with the retro Pokédex interface.

---

## Live Deployment

- **Live Site**: [https://brok.arpanpramanik.in](https://brok.arpanpramanik.in)

---

## Known Limitations and What's Next

- **Dynamic Quantized Cache**: Currently using in-memory LRU caching for hot queries. Planning to implement a sub-millisecond local SQLite semantic cache to drop repeat query latency below 5ms.
- **Bi-directional WebRTC Audio**: Currently streaming audio via HTTP WAV chunks and WebSocket PCM. Transitioning to native WebRTC data channels for full-duplex sub-50ms conversational interruption.
- **Expanded Corpus Indexing**: The current deployment is loaded with a 2,000-vector subset of MSMARCO. Scaling to 1M+ vectors with Qdrant HNSW on-disk indexing via `services/indexing-service/src/ingest_msmarco_xi.py`.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
