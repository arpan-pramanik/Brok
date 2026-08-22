# Architecture & Technical Decisions

This document breaks down the core technical decisions behind Brok's low-latency voice RAG pipeline, what alternatives we evaluated, and why we chose this architecture.

---

## 1. Rust Microservices vs Monolithic Python Backend

### Problem
Python-based async frameworks (FastAPI/Uvicorn) suffered from GIL contention, unpredictable garbage collection pauses, and high baseline memory footprints when simultaneously handling audio PCM streaming, vector search routing, and speculative LLM race conditions under high concurrency.

### Decision
We wrote the core latency-critical microservices (ASR proxy, Vector Retrieval, Orchestrator) in **Rust (Axum + Tokio + Reqwest)**.

### Rejected Alternative
- **Python FastAPI + Asyncio**: P95 retrieval latency hovered around 35–60ms due to Python JSON serialization overhead and async task scheduling latency.
- **Why Rust won**: Zero-cost abstractions, deterministic sub-1ms task switching in Tokio, native memory safety, and steady-state retrieval processing times under **2.5ms**.

---

## 2. In-Memory ONNX FastEmbed vs Heavy External Embedding APIs

### Problem
Calling cloud-hosted embedding APIs (e.g. OpenAI `text-embedding-3-small` or Cohere Embed) introduces 80–180ms network RTT and unpredictable rate limits before search even begins.

### Decision
We embedded an in-memory **FastEmbed ONNX runtime (`BAAI/bge-small-en-v1.5`, 384-dim)** directly inside the Rust retrieval service with dual-query normalization (raw query + keyword normalized query in a single parallel batch).

### Rejected Alternative
- **Cloud Embedding APIs**: Rejected due to network round-trip overhead making sub-100ms end-to-end RAG impossible.
- **Python Sentence-Transformers via PyTorch**: Rejected due to high RAM consumption (1.2GB) and slow cold-start initialization times.
- **Why ONNX FastEmbed won**: Runs in **~1.8ms** per query batch using CPU AVX2 instructions with only ~120MB memory footprint.

---

## 3. Speculative Multi-Provider LLM Racing with Early Stream Selection

### Problem
Individual LLM providers occasionally experience network jitter, cold starts, or transient queuing spikes, degrading P95/P100 latency from 90ms up to 800ms.

### Decision
The orchestrator implements a non-blocking **speculative race** (`tokio::select!`) across multiple endpoints (Groq Primary LPU, Groq Secondary LPU, OpenRouter failover). The first provider that emits valid HTTP 200 chunk headers wins the stream; remaining tasks are cancelled instantly without blocking client output.

### Rejected Alternative
- **Sequential Fallback**: Try Provider A, wait 2s for timeout, then failover to Provider B. Rejected because user experience suffers on any transient provider hiccup.
- **Why Speculative Racing won**: Keeps P50 at **102.4ms** and caps P95 at **117.5ms** under live conditions.

---

## 4. Multi-Tier Speech-to-Text Pipeline (Browser Native + Sarvam)

### Problem
Browser WebSocket audio streaming on mobile devices often drops or gets blocked by strict HTTPS mixed-content policies, while pure client-side STT can struggle with Indian accents or noisy environments.

### Decision
1. **Client-Side Tier**: Instant visual feedback via Web Speech API (`SpeechRecognition`) for 0ms partial transcript rendering.
2. **Audio Buffer Tier**: On button release, client encodes recorded 16kHz mono PCM Float32 audio into a WAV Blob and sends it via HTTPS `/api/transcribe`.
3. **Engine Tier**: Transcribed directly via Sarvam AI (`saarika:v2.5`) for highly accurate processing without multi-hop routing delays.

---

## 5. Dynamic Multi-Strategy Chunking vs Fixed-Size Arbitrary Slicing

### Problem
Traditional fixed-size character chunking splits text arbitrarily, often breaking semantic continuity mid-sentence or mid-thought. This causes dense vector embeddings to lose context and degrades recall significantly on complex datasets like MSMARCO-XI.

### Decision
Brok implements a **Dynamic Multi-Strategy Chunking pipeline** that segments text based on semantic boundaries (sentences, clauses), structural Markdown hierarchy (headers, paragraphs), and injects rich metadata (document titles, source info) into each chunk to preserve global context. 

### Rejected Alternative
- **Fixed-Size Chunking (e.g., 512 characters with overlap)**: Rejected because it dilutes dense vectors and fragments the semantic meaning of passages, resulting in measurably lower MRR and Recall in our ablations.
- **Why Dynamic Chunking won**: Our structural + semantic chunking method guarantees each indexed vector encapsulates a complete factual proposition, maximizing the 384-token capacity of our FastEmbed `bge-small-en-v1.5` model.
