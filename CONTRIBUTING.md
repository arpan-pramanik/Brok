# Contributing to Brok

Thanks for checking out Brok. We welcome contributions to make the voice-enabled RAG pipeline even faster, more robust, and easier to deploy.

## Local Development Workflow

1. Fork and clone the repo:
   ```bash
   git clone https://github.com/arpan-pramanik/Brok.git
   cd Brok
   ```

2. Copy environment template:
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY, OPENROUTER_API_KEY, SARVAM_API_KEY
   ```

3. Run with Docker Compose:
   ```bash
   docker compose up -d --build
   ```

4. Run Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Pull Request Guidelines

- **Zero Secret Commits**: Double-check that no `.env`, API keys, or private credentials are included in your diff.
- **Latency Focus**: When touching the retrieval or orchestrator pipeline, run latency tests and verify that TTFT / end-to-end latency stays within the sub-150ms budget.
- **Keep it simple**: Prefer simple, native stdlib implementations over heavy dependencies.
