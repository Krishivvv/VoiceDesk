# Deployment Guide — VoiceDesk

This document covers **where** VoiceDesk is hosted and **why**, the exact
deployment plan, and how it fits a portfolio.

---

## 1. Hosting decision: Hugging Face Spaces (Docker SDK)

**VoiceDesk is deployed to Hugging Face Spaces using the Docker SDK.**

### Why Hugging Face Spaces

| Requirement | HF Spaces (free CPU) |
|---|---|
| Local Whisper (`torch` + `openai-whisper`, ~145 MB model) | ✅ 16 GB RAM / 2 vCPU free tier handles CPU inference |
| ChromaDB + `sentence-transformers` (~90 MB model) | ✅ Embeddings run on CPU; index baked into the image |
| `ffmpeg` system dependency for audio decoding | ✅ Installable via a Dockerfile (`apt-get install ffmpeg`) |
| Groq LLM (LLaMA 3.3 70B) | ✅ Free API key, called over HTTPS — no local GPU needed |
| Edge TTS | ✅ Free, no key, outbound HTTPS |
| Secrets management | ✅ `GROQ_API_KEY` stored as an encrypted Space secret |
| Persistent build artifacts | ✅ RAG index embedded at image build → no per-boot ingestion |
| Cost | ✅ $0 on the free CPU tier |

The whole stack — Whisper, ChromaDB, FastAPI, Streamlit — is heavy but **CPU-only
and free**, which is exactly the niche HF Spaces fills. The Docker SDK lets us
install `ffmpeg` (a hard requirement for non-WAV audio) and control the runtime
precisely, which the lighter SDKs do not allow.

### Why **not** Streamlit Community Cloud

Streamlit Community Cloud is the obvious alternative but is **too resource-limited
for this workload**:

- **~1 GB RAM ceiling.** Loading Whisper (`torch`) *and* a sentence-transformer
  *and* ChromaDB routinely exceeds this, causing OOM restarts.
- **No system-package control.** You cannot reliably `apt-get install ffmpeg`,
  which Whisper needs for MP3/OGG/FLAC decoding.
- **Single-process Streamlit only.** VoiceDesk also runs a FastAPI service; the
  Docker SDK on Spaces lets us run both in one container, Community Cloud cannot.
- **CPU/time constraints.** Cold Whisper inference can trip the platform's
  resource limits.

> **Mitigation already in place:** if the free CPU is slow, set the Space secret
> `WHISPER_MODEL=tiny` (the code reads it) for ~2–3× faster transcription at a
> small accuracy cost. The README documents this fallback.

---

## 2. Deployment plan (Docker SDK Space)

### 2.1 Artifacts in this repo

| File | Purpose |
|---|---|
| `Dockerfile` | Docker-SDK Space image: ffmpeg + CPU torch + deps; **bakes the ChromaDB index at build** |
| `start.sh` | Launches FastAPI (`:8000`) and Streamlit (`:8501`) in one container |
| `.dockerignore` | Keeps `.venv`, tests, and any local `chroma_db` out of the image |
| `.github/workflows/hf-sync.yml` | Mirrors `main` → the HF Space on every push |
| `README.md` front matter | HF Space config (`sdk: docker`, `app_port: 8501`) |

### 2.2 Why one container (not two services)

The Docker SDK exposes a single port. `start.sh` runs the FastAPI pipeline on
internal `:8000` and the Streamlit UI on public `:8501`; the UI talks to the API
over `localhost`. This keeps everything in the free single-container Space while
preserving the API/UI split used in local development.

### 2.3 Idempotent RAG ingestion + persistence

- **Build-time ingestion.** The Dockerfile runs `_setup_knowledge_base()` once
  during `docker build`, embedding the 16 docs into `data/chroma_db/` and caching
  the embedding model **inside the image**. Containers therefore **never ingest
  on boot** — first request is fast.
- **Runtime idempotency.** `_setup_knowledge_base()` checks
  `collection.count() > 0` and skips re-ingestion, so even if the index is rebuilt
  it happens at most once.
- **True persistence (optional upgrade).** Free Spaces have an ephemeral
  filesystem (reset on rebuild). Because the index is baked into the image this is
  a non-issue. If you later attach **HF persistent storage** (a paid add-on),
  point `chromadb.PersistentClient(path=...)` at the mounted volume (e.g.
  `/data/chroma_db`) and ingestion will persist across rebuilds.

### 2.4 Secret handling

`GROQ_API_KEY` is **never** committed. It is stored as a **Space secret**
(Settings → Variables and secrets → New secret) and read at runtime via
`os.getenv`. The GitHub Action needs an `HF_TOKEN` secret and `HF_USERNAME` /
`HF_SPACE` variables to push — also never in code.

### 2.5 Step-by-step (run at CHECKPOINT B)

1. Create the Space: `huggingface.co/new-space` → SDK **Docker** → name `voicedesk`.
2. In the Space, add secret **`GROQ_API_KEY`** = your Groq key.
   - (Optional) add `WHISPER_MODEL=tiny` if CPU transcription is too slow.
3. In the **GitHub repo** settings, add:
   - Secret `HF_TOKEN` (a Hugging Face *write* token).
   - Variables `HF_USERNAME` and `HF_SPACE` (`voicedesk`).
4. Push to `main` → the `hf-sync` Action force-pushes the repo to the Space,
   which builds the Docker image.
5. Wait for the Space build to go green, then open the public URL.
6. Put that URL at the top of `README.md` (replacing the placeholder).

> **First build is slow** (installs torch, downloads models, bakes the index).
> Subsequent boots are fast because everything is in the image.

---

## 3. Portfolio positioning — make this a pinned flagship

**Pin this repository as a flagship on your GitHub/portfolio.** It is a strong
signal for AI/ML hiring because it combines, end to end, the exact capabilities
teams are hiring for right now:

- **Agentic AI** — a real LangChain **ReAct** agent that reasons and calls tools,
  not a single prompt.
- **RAG** — retrieval over a vector store (ChromaDB + sentence-transformers) with
  a measurable eval (hit-rate + groundedness), plus a **prompt-injection guard**
  on retrieved content — a production concern most demos ignore.
- **Voice I/O** — local Whisper STT and Edge TTS wired into a fully async pipeline.
- **Engineering maturity** — typed, logged, validated FastAPI service; a real
  pytest suite; CI (ruff + pytest); pinned deps + Dependabot; containerised and
  deployed with a live demo.

Agentic + RAG + voice + shipped-and-tested is an uncommon combination in a single
portfolio project, which is precisely why it deserves the flagship slot.

### Suggested repository metadata

- **Description:** *Voice-to-voice AI customer-support agent — Whisper STT,
  LangChain ReAct + ChromaDB RAG, Groq LLaMA 3.3 70B, Edge TTS. FastAPI +
  Streamlit, containerised on Hugging Face Spaces.*
- **Topics:** `agentic-ai`, `rag`, `langchain`, `whisper`, `chromadb`, `groq`,
  `fastapi`, `streamlit`, `text-to-speech`, `speech-to-text`
