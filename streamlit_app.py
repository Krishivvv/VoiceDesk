"""
Streamlit UI for Audio Customer Support Agent Testing

Enhanced with premium dark theme and transcript display.
Run with: streamlit run streamlit_app.py
"""

import base64
import binascii
import io
import os
import wave
from datetime import datetime
from typing import Any

import numpy as np
import requests
import streamlit as st

# Optional mic recording. sounddevice loads the native PortAudio library at
# import time, which raises OSError on headless hosts (e.g. Hugging Face Spaces)
# where no audio device/library exists. Treat that as "recording unavailable"
# and fall back to file upload — never crash the UI over it.
try:
    import sounddevice as sd
    AUDIO_RECORDING_AVAILABLE = True
except (ImportError, OSError):
    sd = None
    AUDIO_RECORDING_AVAILABLE = False

# Configuration
DEFAULT_SERVER_URL = os.getenv("DEFAULT_SERVER_URL", "http://localhost:8000")
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1


def inject_custom_css():
    """Inject premium dark theme CSS."""
    st.markdown("""
    <style>
        /* ---- Global Theme ---- */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
            color: #e0e0e0;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%) !important;
            border-right: 1px solid rgba(0,212,255,0.15);
        }

        /* ---- Cards / Containers ---- */
        div[data-testid="stExpander"], .stAlert, div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(0,212,255,0.12) !important;
            border-radius: 12px !important;
        }

        /* ---- Tabs ---- */
        button[data-baseweb="tab"] {
            color: #8892b0 !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00d4ff !important;
            border-bottom-color: #00d4ff !important;
        }

        /* ---- Primary buttons ---- */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0,212,255,0.35) !important;
        }

        /* ---- Secondary buttons ---- */
        .stButton > button {
            border: 1px solid rgba(0,212,255,0.3) !important;
            border-radius: 10px !important;
            color: #00d4ff !important;
            background: transparent !important;
        }

        /* ---- Inputs ---- */
        input, textarea, .stTextInput > div > div {
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(0,212,255,0.2) !important;
            border-radius: 8px !important;
            color: #e0e0e0 !important;
        }

        /* ---- Metrics ---- */
        div[data-testid="stMetricValue"] {
            color: #00d4ff !important;
            font-weight: 700 !important;
        }

        /* ---- Transcript card ---- */
        .transcript-card {
            background: rgba(0,212,255,0.06);
            border: 1px solid rgba(0,212,255,0.18);
            border-radius: 14px;
            padding: 1.2rem;
            margin-top: 0.5rem;
        }
        .transcript-card .user-msg {
            color: #00d4ff;
            font-weight: 600;
        }
        .transcript-card .agent-msg {
            color: #a78bfa;
            font-weight: 600;
        }
        .processing-badge {
            display: inline-block;
            background: linear-gradient(135deg, #00d4ff33, #7b2ff733);
            border: 1px solid rgba(0,212,255,0.25);
            border-radius: 20px;
            padding: 4px 16px;
            font-size: 0.85rem;
            color: #00d4ff;
            margin-top: 0.5rem;
        }

        /* ---- Headings ---- */
        h1, h2, h3 { color: #e0e0e0 !important; }

        /* ---- Divider ---- */
        hr { border-color: rgba(0,212,255,0.12) !important; }

        /* ---- Success / Error ---- */
        .stSuccess { background: rgba(16,185,129,0.1) !important; border-left: 3px solid #10b981 !important; }
        .stError   { background: rgba(239,68,68,0.1) !important;  border-left: 3px solid #ef4444 !important; }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "server_url": DEFAULT_SERVER_URL,
        "chat_history": [],
        "server_status": "Unknown",
        "recording": False,
        "audio_data": None,
        "audio_response_data": None,
        "audio_transcript": None,
        "audio_processing_time_ms": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def check_server_status(server_url: str) -> dict[str, Any]:
    try:
        root = requests.get(f"{server_url}/", timeout=5).json()
        health = requests.get(f"{server_url}/health", timeout=5).json()
        return {"server_running": True, "root_info": root, "health_info": health}
    except Exception as e:
        return {"server_running": False, "error": str(e)}


def send_text_message(server_url: str, text: str, parameters=None):
    try:
        resp = requests.post(
            f"{server_url}/chat/text",
            json={"text": text, "parameters": parameters or {}},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code == 200:
            return {"success": True, "data": resp.json()}
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_audio_message(server_url: str, audio_data: bytes):
    try:
        resp = requests.post(
            f"{server_url}/chat/audio",
            files={"audio": ("audio.wav", audio_data, "audio/wav")},
            timeout=60,
        )
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}

        try:
            payload = resp.json()
        except ValueError:
            return {"success": False, "error": "Invalid JSON response from server"}

        if not payload.get("success", True):
            return {"success": False, "error": payload.get("error", "Processing failed")}

        audio_b64 = payload.get("audio_response", "")
        if not audio_b64:
            return {"success": False, "error": "Missing audio_response in server reply"}

        try:
            decoded = base64.b64decode(audio_b64)
        except (binascii.Error, ValueError) as e:
            return {"success": False, "error": f"Invalid audio payload: {e}"}

        return {
            "success": True,
            "audio_data": decoded,
            "transcript": payload.get("transcript", {}),
            "processing_time_ms": payload.get("processing_time_ms"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def record_audio(sample_rate=AUDIO_SAMPLE_RATE):
    if not AUDIO_RECORDING_AVAILABLE:
        st.error("Install sounddevice for recording.")
        return None
    try:
        duration = 10
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=AUDIO_CHANNELS, dtype=np.float32)
        sd.wait()
        audio_int16 = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()
    except Exception as e:
        st.error(f"Recording error: {e}")
        return None


def create_audio_player(audio_data: bytes, label="Audio Response"):
    if audio_data:
        st.audio(audio_data, format="audio/mpeg")
        st.download_button(
            label=f"⬇ Download {label}",
            data=audio_data,
            file_name=f"{label.lower().replace(' ','_')}_{datetime.now():%Y%m%d_%H%M%S}.mp3",
            mime="audio/mpeg",
        )


# ──────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Audio Support Agent", page_icon="🎧", layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    inject_custom_css()

    st.title("🎧 Audio Customer Support Agent")
    st.caption("Enhanced with transcript generation & processing metrics")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")
        server_url = st.text_input("API Server URL", value=st.session_state.server_url)
        st.session_state.server_url = server_url

        if st.button("🔄 Check Server Status"):
            with st.spinner("Checking..."):
                st.session_state.server_status = check_server_status(server_url)

        if st.session_state.server_status != "Unknown":
            s = st.session_state.server_status
            if isinstance(s, dict) and s.get("server_running"):
                st.success("✅ Server online")
                h = s.get("health_info", {})
                if h.get("status") == "healthy":
                    st.success("✅ All components ready")
                elif h.get("status") in ("unhealthy", "degraded"):
                    st.warning(f"⚠️ {h.get('message','Components not ready')}")
            else:
                st.error("❌ Server offline")
                if s.get("error"):
                    st.caption(s["error"])

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Text Chat", "🎙️ Enhanced Audio Chat", "📊 Health Monitor", "📖 Docs"])

    # ────── TAB 1 : Text Chat ──────
    with tab1:
        st.header("💬 Text Chat")
        user_msg = st.text_input("Your message:", placeholder="Ask about returns, shipping…", key="text_input")
        if st.button("Send Message", type="primary", key="send_text_btn") and user_msg:
            with st.spinner("Sending…"):
                result = send_text_message(server_url, user_msg)
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append({"timestamp": ts, "user": user_msg, "result": result})

        if st.session_state.chat_history:
            st.subheader("Conversation History")
            for chat in reversed(st.session_state.chat_history[-10:]):
                with st.container():
                    st.markdown(f"**[{chat['timestamp']}] You:** {chat['user']}")
                    if chat["result"]["success"]:
                        d = chat["result"]["data"]
                        st.markdown(f"**🤖 Agent:** {d.get('response_text','')}")
                        if "processing_time_ms" in d:
                            st.caption(f"⏱ {d['processing_time_ms']}ms")
                    else:
                        st.error(chat["result"]["error"])
                    st.divider()

        if st.button("🗑 Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    # ────── TAB 2 : Enhanced Audio Chat ──────
    with tab2:
        st.header("🎙️ Enhanced Audio Chat")
        st.markdown("Record or upload audio → get **audio response + transcript**")

        col_audio, col_transcript = st.columns([1, 1])

        with col_audio:
            st.subheader("🔊 Audio Controls")

            if AUDIO_RECORDING_AVAILABLE:
                if st.button("🎤 Record Audio", key="rec", type="primary"):
                    with st.spinner("Recording 10s — speak now!"):
                        st.session_state.audio_data = record_audio()
                    if st.session_state.audio_data:
                        st.success("✅ Recorded!")
                        st.session_state.audio_response_data = None
                        st.session_state.audio_transcript = None
                        st.session_state.audio_processing_time_ms = None
            else:
                st.warning("Install sounddevice for mic recording.")

            if st.session_state.audio_data:
                st.audio(st.session_state.audio_data, format="audio/wav")

            st.markdown("**Or upload a file:**")
            uploaded = st.file_uploader("Choose audio", type=["wav", "mp3", "ogg", "flac"])
            if uploaded:
                st.session_state.audio_data = uploaded.read()
                st.audio(st.session_state.audio_data)
                st.session_state.audio_response_data = None
                st.session_state.audio_transcript = None
                st.session_state.audio_processing_time_ms = None

            if st.session_state.audio_data:
                if st.button("🚀 Send Audio to Agent", type="primary"):
                    with st.spinner("Processing audio pipeline…"):
                        result = send_audio_message(server_url, st.session_state.audio_data)
                    if result["success"]:
                        st.success("✅ Processed!")
                        st.session_state.audio_response_data = result.get("audio_data")
                        st.session_state.audio_transcript = result.get("transcript")
                        st.session_state.audio_processing_time_ms = result.get("processing_time_ms")
                    else:
                        st.error(f"❌ {result['error']}")

            if st.session_state.audio_response_data:
                st.subheader("🔈 Agent Response")
                create_audio_player(st.session_state.audio_response_data, "Agent Response")

        with col_transcript:
            st.subheader("📝 Transcript")
            transcript = st.session_state.audio_transcript or {}
            if transcript:
                st.markdown(f"""
                <div class="transcript-card">
                    <p class="user-msg">🗣️ User:</p>
                    <p>{transcript.get('user_input','')}</p>
                    <hr style="border-color:rgba(0,212,255,0.12)">
                    <p class="agent-msg">🤖 Agent:</p>
                    <p>{transcript.get('agent_response','')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Transcript will appear here after processing.")

        if st.session_state.audio_processing_time_ms is not None:
            ms = st.session_state.audio_processing_time_ms
            st.markdown(f'<div class="processing-badge">⏱ Processing: {ms/1000:.2f}s ({ms}ms)</div>', unsafe_allow_html=True)

    # ────── TAB 3 : Health ──────
    with tab3:
        st.header("📊 Health Monitor")
        if st.button("🔄 Refresh"):
            with st.spinner("Checking…"):
                st.session_state.server_status = check_server_status(server_url)

        if st.session_state.server_status != "Unknown":
            s = st.session_state.server_status
            if isinstance(s, dict) and s.get("server_running"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Server", "✅ Running")
                c2.metric("Version", s.get("root_info", {}).get("version", "?"))
                c3.metric("Checked", datetime.now().strftime("%H:%M:%S"))

                st.subheader("Components")
                health = s.get("health_info", {})
                for comp, ok in health.get("components", {}).items():
                    icon = "✅" if ok else "❌"
                    st.markdown(f"- **{comp}**: {icon} {'Ready' if ok else 'Not Ready'}")
            else:
                st.error("❌ Server not accessible")

    # ────── TAB 4 : Docs ──────
    with tab4:
        st.header("📖 Documentation")
        st.markdown("""
        **Start the server:**
        ```bash
        cd audio_support_agent
        python -m src.api.server
        ```

        **Endpoints:**
        | Endpoint | Description |
        |---|---|
        | `GET /` | API info |
        | `GET /health` | Health check |
        | `POST /chat/text` | Text chat |
        | `POST /chat/audio` | Audio pipeline (returns JSON with transcript) |

        **Audio tips:** Use WAV 16kHz mono for best results. Install `ffmpeg` for format conversion.
        """)


if __name__ == "__main__":
    main()
