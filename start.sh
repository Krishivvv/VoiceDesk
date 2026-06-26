#!/usr/bin/env bash
# Container entrypoint: run the FastAPI pipeline server (internal :8000) and
# the Streamlit UI (public :8501) in one container. The UI talks to the API
# over localhost. GROQ_API_KEY is read from the environment (a Space secret).
set -euo pipefail

# Start the API in the background.
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Ensure the API is torn down if Streamlit exits.
trap 'kill $API_PID 2>/dev/null || true' EXIT

# Point the UI at the local API and launch it in the foreground (port 8501).
export DEFAULT_SERVER_URL="http://localhost:8000"
exec streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
