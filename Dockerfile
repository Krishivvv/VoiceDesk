# Hugging Face Space — Docker SDK.
# Single container running the FastAPI pipeline (internal :8000) and the
# Streamlit UI (public :8501). The ChromaDB knowledge base is embedded ONCE
# at build time so no ingestion happens on boot.

FROM python:3.11-slim

# ffmpeg: Whisper audio decoding. libsndfile1: soundfile. git: optional VCS deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Run as the non-root user Hugging Face Spaces expects (UID 1000).
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install CPU-only PyTorch first (smaller, no CUDA), then the rest.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY . .

# Build the ChromaDB knowledge base and pre-cache the embedding model into the
# image. Idempotent: this runs once at build, never on container start.
RUN python -c "import asyncio; from src.llm.agent import CustomerSupportAgent; \
asyncio.run(CustomerSupportAgent({})._setup_knowledge_base())"

# Make the app (incl. data/chroma_db and model cache) owned by the runtime user.
RUN chown -R user:user /app /home/user
USER user

EXPOSE 8501

CMD ["bash", "start.sh"]
