"""Streamlit wrapper that embeds the production frontend and backend."""
from __future__ import annotations

import asyncio
import socket
import threading
import time
from os import environ, getcwd, path
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

try:
    import uvicorn
except ImportError as exc:  # pragma: no cover - defensive guard
    raise RuntimeError("uvicorn must be installed to run the integrated frontend") from exc

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------
load_dotenv(path.join(getcwd(), ".env"))
if not environ.get("GOOGLE_API_KEY") and environ.get("GEMINI_API_KEY"):
    environ["GOOGLE_API_KEY"] = environ["GEMINI_API_KEY"]

ROOT_DIR = Path(__file__).resolve().parent
DIST_DIR = ROOT_DIR / "frontend" / "dist"
BACKEND_HOST = environ.get("CHATBOT_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(environ.get("CHATBOT_BACKEND_PORT", "8000"))
BACKEND_BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
FRONTEND_EMBED_URL = environ.get("CHATBOT_FRONTEND_URL", BACKEND_BASE_URL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _wait_for_url(url: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1):  # noqa: S310 - internal address
                return True
        except URLError:
            time.sleep(0.2)
    return False


def _start_backend_server() -> None:
    """Run the FastAPI backend (serves API + static SPA)."""
    config = uvicorn.Config(
        "backend.server:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        log_level=environ.get("CHATBOT_UVICORN_LOG_LEVEL", "info"),
        reload=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[assignment]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())


def _ensure_backend_running() -> Optional[str]:
    if _port_is_open(BACKEND_HOST, BACKEND_PORT):
        return None

    if not DIST_DIR.exists():
        return "Frontend build not found. Run `npm install` and `npm run build` inside the `frontend/` folder first."

    backend_thread = threading.Thread(target=_start_backend_server, name="backend-server", daemon=True)
    backend_thread.start()

    index_url = f"{BACKEND_BASE_URL}/"
    if _wait_for_url(index_url):
        return None

    return f"Unable to start the backend service on port {BACKEND_PORT}. Check logs for details."


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------
if not DIST_DIR.exists():
    st.error('Frontend build not found. Run `npm install` and `npm run build` in the `frontend/` folder, then restart the app.')
    st.stop()

st.set_page_config(page_title="ChaCha Chatbot", page_icon="💬", layout="wide")
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "_backend_ready" not in st.session_state:
    with st.spinner("Booting backend service..."):
        error_message = _ensure_backend_running()
    if error_message:
        st.session_state["_backend_ready"] = False
        st.error(error_message)
        st.stop()
    st.session_state["_backend_ready"] = True

st.toast("Frontend served via embedded React app", icon="✅")


if "127.0.0.1" in FRONTEND_EMBED_URL or "localhost" in FRONTEND_EMBED_URL:
    st.info("The embedded frontend expects to run on the same machine as this Streamlit app. Remote hosting is not supported.")
components.html(
    f"""
    <iframe src="{FRONTEND_EMBED_URL}" style="border:none;width:100%;height:100vh;" allow="clipboard-write"></iframe>
    """,
    height=900,
)
