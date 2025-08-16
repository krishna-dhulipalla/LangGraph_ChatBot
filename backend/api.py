# backend/api.py
from __future__ import annotations

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, PlainTextResponse
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from typing import Optional
from uuid import uuid4
from pathlib import Path
import json, secrets, urllib.parse, os
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# --- import your compiled LangGraph app ---
# agent.py exports: app = graph.compile(...)
# ensure backend/__init__.py exists and you run uvicorn from repo root.
from .agent import app as lg_app

api = FastAPI(title="LangGraph Chat API")

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
REDIRECT_URI = f"{BASE_URL}/oauth/google/callback"
TOKEN_FILE = Path("/data/google_token.json")

# CORS (handy during dev; tighten in prod)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
REDIRECT_URI = f"{BASE_URL}/oauth/google/callback"
TOKEN_FILE = Path("/data/google_token.json")  # persistent on HF Spaces

def _client_config():
    return {
        "web": {
            "client_id": CLIENT_ID,
            "project_id": "chatk",  # optional
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_secret": CLIENT_SECRET,
            "redirect_uris": [REDIRECT_URI],
        }
    }

@api.get("/oauth/google/start")
def oauth_start():
    # optional CSRF protection
    state = secrets.token_urlsafe(16)
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES, redirect_uri=REDIRECT_URI)
    auth_url, _ = flow.authorization_url(
        access_type="offline",        # get refresh token
        include_granted_scopes="true",
        prompt="consent"              # ensures refresh token on repeated login
    )
    # You can store `state` server-side if you validate it later
    return RedirectResponse(url=auth_url)

@api.get("/oauth/google/callback")
def oauth_callback(request: Request):
    # Exchange code for tokens
    full_url = str(request.url)  # includes ?code=...
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES, redirect_uri=REDIRECT_URI)
    flow.fetch_token(authorization_response=full_url)
    creds = flow.credentials
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(creds.to_json())
    return PlainTextResponse("Google Calendar connected. You can close this tab.")

def get_gcal_service():
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    else:
        # Not authorized yet
        raise RuntimeError(
            f"Google not connected. Visit {BASE_URL}/oauth/google/start to connect."
        )
    return build("calendar", "v3", credentials=creds)

@api.get("/debug/routes")
def list_routes():
    return {"routes": sorted([getattr(r, "path", str(r)) for r in api.router.routes])}

@api.get("/api/debug/oauth")
def debug_oauth():
    return {
        "base_url_env": BASE_URL_RAW,
        "base_url_effective": BASE_URL,
        "redirect_uri_built": REDIRECT_URI,
    }

async def _event_stream(thread_id: str, message: str, request: Request):
    """
    Common generator for SSE. Emits:
      - thread: thread_id to persist in the client
      - token : streamed model text tokens
      - done  : end-of-stream sentinel
    """
    config = {"configurable": {"thread_id": thread_id}}

    # send thread id early so the client can store it immediately
    yield {"event": "thread", "data": thread_id}

    try:
        # stream events directly from LangGraph
        async for ev in lg_app.astream_events(
            {"messages": [("user", message)]},
            config=config,
            version="v2",
        ):
            # model token stream
            if ev["event"] == "on_chat_model_stream":
                chunk = ev["data"]["chunk"].content
                # chunk can be a string, None, or (rarely) list of content parts
                if isinstance(chunk, list):
                    text = "".join(getattr(p, "text", "") or str(p) for p in chunk)
                else:
                    text = chunk or ""
                if text:
                    yield {"event": "token", "data": text}

            # (optional) forward tool results:
            # if ev["event"] == "on_tool_end":
            #     tool_name = ev["name"]
            #     tool_out  = ev["data"].get("output")
            #     yield {"event": "tool", "data": json.dumps({"name": tool_name, "output": tool_out})}

            # stop if client disconnects
            if await request.is_disconnected():
                break

    finally:
        # explicit completion so the client can stop spinners immediately
        yield {"event": "done", "data": "1"}

# --- GET route for EventSource (matches the React UI I gave you) ---
# GET
@api.get("/chat")
async def chat_get(
    request: Request,
    message: str = Query(...),
    thread_id: Optional[str] = Query(None),
    is_final: Optional[bool] = Query(False),
):
    tid = thread_id or str(uuid4())
    async def stream():
        # pass both thread_id and is_final to LangGraph
        config = {"configurable": {"thread_id": tid, "is_final": bool(is_final)}}
        yield {"event": "thread", "data": tid}
        try:
            async for ev in lg_app.astream_events({"messages": [("user", message)]}, config=config, version="v2"):
                if ev["event"] == "on_chat_model_stream":
                    chunk = ev["data"]["chunk"].content
                    if isinstance(chunk, list):
                        text = "".join(getattr(p, "text", "") or str(p) for p in chunk)
                    else:
                        text = chunk or ""
                    if text:
                        yield {"event": "token", "data": text}
                if await request.is_disconnected():
                    break
        finally:
            yield {"event": "done", "data": "1"}
    return EventSourceResponse(stream())

# POST
@api.post("/chat")
async def chat_post(request: Request):
    body = await request.json()
    message = body.get("message", "")
    tid = body.get("thread_id") or str(uuid4())
    is_final = bool(body.get("is_final", False))
    config = {"configurable": {"thread_id": tid, "is_final": is_final}}
    return EventSourceResponse(_event_stream_with_config(tid, message, request, config))

# helper if you prefer to keep a single generator
async def _event_stream_with_config(thread_id: str, message: str, request: Request, config: dict):
    yield {"event": "thread", "data": thread_id}
    try:
        async for ev in lg_app.astream_events({"messages": [("user", message)]}, config=config, version="v2"):
            ...
    finally:
        yield {"event": "done", "data": "1"}

# --- Serve built React UI (ui/dist) under the same origin ---
# repo_root = <project>/  ; this file is <project>/backend/api.py
REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIST = REPO_ROOT / "ui" / "dist"

RESUME_PATH = REPO_ROOT / "backend" / "assets" / "KrishnaVamsiDhulipalla.pdf"

@api.get("/resume/download")
def resume_download():
    if not RESUME_PATH.is_file():
        return PlainTextResponse("Resume not found", status_code=404)
    # Same-origin download; content-disposition prompts save/open dialog
    return FileResponse(
        path=str(RESUME_PATH),
        media_type="application/pdf",
        filename="Krishna_Vamsi_Dhulipalla_Resume.pdf",
    )

if UI_DIST.is_dir():
    api.mount("/", StaticFiles(directory=str(UI_DIST), html=True), name="ui")
else:
    @api.get("/")
    def no_ui():
        return PlainTextResponse(
            f"ui/dist not found at: {UI_DIST}\n"
            "Run your React build (e.g., `npm run build`) or check the path.",
            status_code=404,
        )