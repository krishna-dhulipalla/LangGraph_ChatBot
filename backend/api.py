# backend/api.py
from __future__ import annotations

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse
from typing import Optional
from uuid import uuid4
from pathlib import Path
import json

# --- import your compiled LangGraph app ---
# agent.py exports: app = graph.compile(...)
# ensure backend/__init__.py exists and you run uvicorn from repo root.
from .agent import app as lg_app

api = FastAPI(title="LangGraph Chat API")

# CORS (handy during dev; tighten in prod)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@api.get("/health")
def health():
    return {"ok": True}

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