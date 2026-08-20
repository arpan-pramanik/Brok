# Brok

Ultra-fast voice-enabled Pokédex RAG system delivering end-to-end question answering in under 120 milliseconds.

## Why I Built This

Most voice AI pipelines feel clunky and sluggish because they chain together high-latency cloud APIs, taking 2 to 4 seconds just to hear back. I wanted to see how fast an end-to-end voice RAG system could actually get if every millisecond was treated as a hard budget — combining local in-memory embeddings, sub-2ms vector retrieval in Rust, Groq LPU hardware inference, and instant voice-to-text feedback.

---

## Architecture

```
User Voice / Text Query
          │
          ├─── [Web Speech API / Groq Whisper Turbo (~80ms)] ───► Clean Transcript
          │
          ▼
Orchestrator (Rust / Axum on Port 8000)
          │
          ├───► 1. Safety Guardrail (<0.2ms Pattern & Policy Scan)
          │
          ├───► 2. Retrieval Service (Rust / FastEmbed ONNX on Port 8002)
          │          │
          │          ├── Dual Query Normalization (Raw + Keyword Filter)
          │          ├── Batch Dense Embedding (bge-small-en-v1.5 in ~1.8ms)
          │          └── Qdrant Vector Search (msmarco_xi Collection in ~1.0ms)
          │
          ├───► 3. Context Relevance Gate (Vector Score Calibrated Threshold)
          │
          ├───► 4. Speculative LLM Race (tokio::select!)
          │          ├── Groq Primary LPU (allam-2-7b / gpt-oss-20b)
          │          ├── Groq Secondary LPU (Failover)
          │          └── OpenRouter Llama 3.1 8B (Fallback)
          │
          └───► 5. Hallucination Guardrail (<0.2ms Grounding Token Verification)
          │
          ▼
Streamed CRT Output + Audio TTS to Pokédex Client (<120ms Total)
```

---

## Measured Performance Metrics

Latency numbers measured across **225 real test queries** running against the live AWS EC2 backend with 1,997 MSMARCO dataset vectors in Qdrant (recorded in `extreme_benchmark_output.json`):

| Pipeline Stage | P50 (Median) | P70 | P95 | P100 (Max) | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Safety Guardrail** | < 0.1 ms | < 0.1 ms | 0.2 ms | 0.4 ms | 0.1 ms |
| **Embedding (FastEmbed ONNX)** | 1.8 ms | 2.1 ms | 2.6 ms | 3.8 ms | 1.9 ms |
| **Vector Retrieval (Qdrant)** | 0.9 ms | 1.2 ms | 2.1 ms | 3.5 ms | 1.1 ms |
| **LLM Generation (Groq LPU)** | 102.0 ms | 104.0 ms | 119.1 ms | 192.0 ms | 87.1 ms |
| **Hallucination Verification** | 0.1 ms | 0.2 ms | 0.3 ms | 0.5 ms | 0.2 ms |
| **Total End-to-End Latency** | **102.4 ms** | **106.0 ms** | **117.5 ms** | **192.7 ms** | **87.8 ms** |

---

## Key Technical Decisions

- **Rust Microservices over Python**: Rewrote the retrieval and orchestration pipelines in Rust (Tokio + Axum). This eliminated Python GIL locking and garbage collection latency, dropping steady-state retrieval processing to sub-3ms.
- **In-Memory FastEmbed ONNX**: Embedded `bge-small-en-v1.5` directly inside the retrieval service binary. Instead of paying 100ms+ network RTT to external embedding APIs, local vector embedding runs in ~1.8ms on CPU.
- **Dual-Query Keyword Normalization**: Strips conversational prefixes (*"what is"*, *"can you tell me"*, *"when was"*) and embeds both raw and normalized queries in a single parallel ONNX batch, improving dense retrieval hit rate for conversational voice queries.
- **Speculative Multi-Provider LLM Racing**: Orchestrator races requests across primary and secondary Groq LPU endpoints and OpenRouter via `tokio::select!`. The fastest stream wins and is forwarded directly to the client, buffering against provider jitter.
- **Multi-Tier Speech-to-Text**: Real-time browser Web Speech API provides instant visual feedback while holding the voice button, while the recorded 16kHz mono WAV buffer is transcribed via Groq Whisper Large V3 Turbo with Sarvam AI fallback.

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

- **Site** : https://brok.arpanpramanik.in

---

## Known Limitations and What's Next

- **Dynamic Quantized Cache**: Currently using in-memory LRU caching for hot queries. Planning to implement a sub-millisecond local SQLite semantic cache to drop repeat query latency below 5ms.
- **Bi-directional WebRTC Audio**: Currently streaming audio via HTTP WAV chunks and WebSocket PCM. Transitioning to native WebRTC data channels for full-duplex sub-50ms conversational interruption.
- **Expanded Corpus Indexing**: The current deployment is loaded with a 2,000-vector subset of MSMARCO. Scaling to 1M+ vectors with Qdrant HNSW on-disk indexing.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
