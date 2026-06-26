---
title: VoiceDesk Audio Support Agent
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: true
license: mit
---

# 🎧 VoiceDesk — AI Audio Customer Support Agent

> **A fully voice-driven customer support pipeline powered by OpenAI Whisper, LangChain ReAct + RAG, and Microsoft Edge TTS — from spoken question to spoken answer in one shot.**

### ▶️ Live demo: **https://huggingface.co/spaces/&lt;your-hf-username&gt;/voicedesk**

*(Hugging Face Space — Docker SDK. Provisioning steps in [DEPLOYMENT.md](DEPLOYMENT.md); the URL above is finalised once the Space is created.)*

<!-- Demo GIF placeholder — replace docs/demo.gif with a screen recording of a full voice round-trip (speak → transcript → spoken answer). -->
![VoiceDesk voice round-trip demo](docs/demo.gif)

[![CI](https://github.com/Krishivvv/AI-Audio-Support-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Krishivvv/AI-Audio-Support-Agent/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

---

## What It Does

You speak. VoiceDesk transcribes your voice with Whisper, routes the query through a LangChain ReAct agent that searches a local ChromaDB knowledge base (RAG), generates a grounded response with an LLM (Groq / LLaMA or OpenAI), and converts the reply back to audio using Edge TTS — all in a single async pipeline.

```
🎤 Audio Input
     │
     ▼
┌──────────────┐
│  STT Layer   │  OpenAI Whisper (local, base model)
│  base_stt.py │  soundfile decode → 16 kHz mono → Whisper inference
└──────┬───────┘
       │  transcribed text
       ▼
┌──────────────────────────┐
│  LLM Agent + RAG Layer   │  LangChain ReAct Agent
│  agent.py                │  ┌─ ChromaDB knowledge_search tool
│                          │  │  16 docs · all-MiniLM-L6-v2 embeddings
│                          │  └─ Top-3 semantic matches injected as context
│                          │  Groq (LLaMA-3.3-70B) or OpenAI GPT-3.5
└──────┬───────────────────┘
       │  agent response text
       ▼
┌──────────────┐
│  TTS Layer   │  Microsoft Edge TTS (free, en-US-AriaNeural)
│  base_tts.py │  Streams MP3 audio bytes
└──────┬───────┘
       │
       ▼
🔊 Audio Output  (+  📝 Transcript  +  ⏱ Processing time)
```

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| **STT** | OpenAI Whisper (`base`) | Local inference — no API key needed |
| **LLM** | Groq API — LLaMA 3.3 70B Versatile | Falls back to OpenAI GPT-3.5-Turbo |
| **RAG** | ChromaDB + `sentence-transformers` | Persistent vector store at `data/chroma_db/` |
| **TTS** | Microsoft Edge TTS | Free, no API key, streamed MP3 |
| **API** | FastAPI + Uvicorn | REST endpoints with CORS |
| **UI** | Streamlit | Dark-theme, tabs: text chat / audio chat / health / docs |
| **Agent** | LangChain ReAct (`langchain-classic`) | ConversationBufferMemory, max 5 iterations |

---

## Project Structure

```
audio_support_agent/
├── src/
│   ├── pipeline.py              # AudioSupportPipeline — STT → LLM → TTS orchestrator
│   ├── stt/
│   │   └── base_stt.py          # BaseSTT + STTService (Whisper, soundfile/ffmpeg fallback)
│   ├── llm/
│   │   └── agent.py             # BaseAgent + CustomerSupportAgent (LangChain ReAct + ChromaDB RAG)
│   ├── tts/
│   │   └── base_tts.py          # BaseTTS + TTSService (Edge TTS, streaming synthesis)
│   ├── api/
│   │   └── server.py            # FastAPI server — /health /chat/text /chat/audio /debug/stt
│   └── utils/
│       └── kb_test.py           # CLI debug tool for inspecting ChromaDB & testing RAG queries
├── data/
│   └── chroma_db/               # Persistent vector store (auto-created on first run)
├── docs/
│   └── RAG_IMPLEMENTATION_GUIDE.md
├── tests/
│   ├── __init__.py
│   └── test_stt.py              # Unit + integration tests for STT layer
├── streamlit_app.py             # Streamlit UI
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Prerequisites

- Python **3.10+**
- [ffmpeg](https://ffmpeg.org/download.html) on your `PATH` (for non-WAV audio formats)
- A **Groq API key** (free at [console.groq.com](https://console.groq.com)) **or** an OpenAI API key

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/your-username/voicedesk.git
cd voicedesk/audio_support_agent

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run downloads the Whisper `base` model (~145 MB) and the `all-MiniLM-L6-v2` sentence-transformer (~90 MB) automatically.

### 3. Configure environment

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` and set at minimum your LLM key:

```env
# Option A — Groq (recommended, free tier)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile

# Option B — OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

STT (Whisper) and TTS (Edge TTS) are fully local/free — no additional keys needed.

### 4. Run

Open **two terminals** from inside `audio_support_agent/`:

**Terminal 1 — API server:**
```bash
python -m src.api.server
```

**Terminal 2 — Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs (Swagger) | http://localhost:8000/docs |
| FastAPI docs (ReDoc) | http://localhost:8000/redoc |

---

## API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API metadata (name, version, doc links) |
| `GET` | `/health` | Component health: pipeline / STT / LLM / TTS |
| `POST` | `/chat/text` | Text → LLM → text + processing time |
| `POST` | `/chat/audio` | Audio → STT → LLM → TTS → base64 MP3 + transcript + timing |
| `GET` | `/chat/audio/{text}` | TTS-only: returns raw MP3 bytes for a given text |
| `POST` | `/debug/stt` | STT-only: upload audio, get back raw transcription |

### Example requests

```bash
# Health check
curl http://localhost:8000/health

# Text chat
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is your return policy?"}'

# Full audio pipeline
curl -X POST http://localhost:8000/chat/audio \
  -F "audio=@my_question.wav" \
  | jq '{transcript, processing_time_ms}'

# Debug STT only
curl -X POST http://localhost:8000/debug/stt \
  -F "audio=@my_question.wav"
```

### Audio response payload

```json
{
  "success": true,
  "audio_response": "<base64-encoded MP3>",
  "transcript": {
    "user_input": "What is the return policy?",
    "agent_response": "We offer a 30-day return policy for all products..."
  },
  "processing_time_ms": 3142
}
```

---

## Knowledge Base (RAG)

The agent answers from **16 pre-loaded customer support documents** across 10 categories, stored as vector embeddings in ChromaDB. No external knowledge base setup is needed — documents are ingested automatically on the first run.

| Category | Documents |
|---|---|
| `returns` | Return policy overview, return process steps, non-returnable items |
| `shipping` | Shipping methods & times, international shipping, order tracking |
| `support` | Contact information, response times |
| `warranty` | Product warranty coverage |
| `technical` | Technical support hours & channels |
| `account` | Account management |
| `orders` | Order modifications & cancellations |
| `payment` | Accepted payment methods |
| `billing` | Billing and invoices |
| `products` | Product availability, size & fit guide |

### How RAG works

1. **Ingest** — On first startup, each document is embedded using `all-MiniLM-L6-v2` and written to ChromaDB at `data/chroma_db/`. Subsequent restarts skip re-ingestion.
2. **Query** — The LangChain `knowledge_search` tool is called by the ReAct agent. ChromaDB performs cosine-similarity search and returns the top-3 most relevant documents.
3. **Generate** — Matched documents (with title, category, relevance %) are injected into the agent's context, and the LLM generates a grounded response.

```
Question: What is the return policy?
Thought: I need to look up return policy information.
Action: knowledge_search
Action Input: return policy
Observation: **Return Policy Overview** (Category: returns, Relevance: 94.2%)
             We offer a 30-day return policy...
Thought: I now know the final answer.
Final Answer: Our return policy allows returns within 30 days...
```

**Test the knowledge base directly:**

```bash
python src/utils/kb_test.py
```

---

## Streamlit UI Tabs

| Tab | Description |
|---|---|
| 💬 **Text Chat** | Type a question → get a text response with processing time; stores up to 10 messages in history |
| 🎙️ **Enhanced Audio Chat** | Record via microphone or upload WAV/MP3/OGG/FLAC → full pipeline → playable MP3 response + transcript card |
| 📊 **Health Monitor** | Live status of all three pipeline components with per-component ready indicators |
| 📖 **Docs** | Quick-start server commands and endpoint reference |

---

## Pipeline Architecture

### `AudioSupportPipeline` (`src/pipeline.py`)

| Method | Description |
|---|---|
| `initialize()` | Async init of STT → LLM → TTS in sequence; raises on any component failure |
| `process_audio(audio_bytes)` | Full pipeline; returns `bytes` (response audio) |
| `process_audio_with_transcript(audio_bytes)` | Full pipeline; returns `(audio_bytes, TranscriptData, processing_time_ms)` |
| `process_text(text)` | LLM + TTS only (skip STT); returns `(response_text, audio_bytes)` |
| `process_text_with_timing(text)` | LLM only with timing; returns `(response_text, processing_time_ms)` |
| `health_check()` | Dict of `bool` per component |
| `cleanup()` | Gracefully shuts down all components |

### `STTService` (`src/stt/base_stt.py`)

- Loads Whisper `base` model locally (configurable via `stt_config["model"]`)
- Decodes audio via `soundfile` in-memory; falls back to temp-file + ffmpeg for other formats
- Resamples to 16 kHz mono before Whisper inference
- Runs in a thread pool via `asyncio.to_thread` to avoid blocking the event loop

### `CustomerSupportAgent` (`src/llm/agent.py`)

- `langchain-classic` ReAct agent with `ConversationBufferMemory`
- Single tool: `knowledge_search` → `_rag_search()` → ChromaDB cosine query
- Supports Groq (via OpenAI-compatible base URL) and native OpenAI APIs
- Max 5 agent iterations; graceful fallback to direct RAG result on executor errors

### `TTSService` (`src/tts/base_tts.py`)

- Microsoft Edge TTS via `edge-tts` (free, requires internet)
- Default voice: `en-US-AriaNeural` (configurable via `tts_config["voice"]`)
- Streams MP3 chunks; returns complete `bytes` object
- `synthesize_stream()` wraps bytes in `io.BytesIO` for streaming-compatible consumers

---

## Running Tests

```bash
# From audio_support_agent/  (install dev tooling first)
pip install -r requirements-dev.txt

# Full suite (unit + RAG integration). Integration tests build the local
# ChromaDB index with sentence-transformers — no API key required.
pytest

# Unit tests only (skip the RAG/model-download integration tests)
pytest -m "not integration"

# RAG evaluation report (retrieval hit-rate + answer groundedness)
python -m tests.test_rag_eval
```

Lint with `ruff check .`. CI runs ruff + pytest on every push (see
[.github/workflows/ci.yml](.github/workflows/ci.yml)).

---

## Deployment

VoiceDesk deploys to **Hugging Face Spaces (Docker SDK)** — free CPU is enough
for Whisper + ChromaDB, and Groq + Edge TTS are free. A `Dockerfile` installs
ffmpeg and bakes the RAG index at build time, and a GitHub Action
([.github/workflows/hf-sync.yml](.github/workflows/hf-sync.yml)) mirrors `main`
to the Space on every push. `GROQ_API_KEY` is stored as a Space secret, never in
code. Full rationale and step-by-step instructions are in
[DEPLOYMENT.md](DEPLOYMENT.md).

```bash
# Build and run the whole stack locally with Docker:
docker build -t voicedesk .
docker run -p 8501:8501 -e GROQ_API_KEY=gsk_xxx voicedesk
# UI on http://localhost:8501
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError` on startup | Run `pip install -r requirements.txt` inside the venv |
| Pipeline stays `unhealthy` | Check `.env` has a valid `GROQ_API_KEY` or `OPENAI_API_KEY` and restart the server |
| Port 8000 already in use | `SERVER_PORT=8001` in `.env`, then `python -m src.api.server` |
| Empty transcription | Speak clearly; use 16 kHz WAV mono; check mic levels |
| Whisper very slow | Change `"model": "tiny"` in `stt_config` inside `server.py` startup |
| Edge TTS returns no audio | Verify internet connection — Edge TTS calls Microsoft servers |
| `ffmpeg not found` on non-WAV audio | Install [ffmpeg](https://ffmpeg.org/download.html) and add it to your system PATH |
| ChromaDB re-ingesting every run | Delete `data/chroma_db/` and let it rebuild cleanly |

---

## Configuration Reference

All configuration lives in `.env` (copy from `.env.example`):

```env
# LLM — pick one
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1

OPENAI_API_KEY=sk-...

# Optional STT/TTS service keys (not needed for default Whisper + Edge TTS setup)
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=true
LOG_LEVEL=INFO
```

---

## Extending the Agent

### Swap the STT engine

Implement `BaseSTT` in `src/stt/base_stt.py` — the abstract interface requires `initialize()`, `transcribe(audio_bytes)`, and `cleanup()`. The pipeline calls them automatically.

### Swap the TTS engine

Implement `BaseTTS` in `src/tts/base_tts.py` — requires `initialize()`, `synthesize(text)`, `synthesize_stream(text)`, and `cleanup()`.

### Add agent tools

Add new `Tool` objects inside `CustomerSupportAgent._create_tools()` in `src/llm/agent.py`. The ReAct agent will automatically choose the appropriate tool based on the query.

### Add knowledge base documents

Extend `_get_customer_support_documents()` in `agent.py` with new `{title, category, content}` dicts. Delete `data/chroma_db/` and restart to re-ingest.

---

## License

MIT — see `LICENSE` for details.
