"""
main.py

FastAPI wrapper around the LangGraph agent, so the chat frontend has
something to talk to over HTTP.

Session memory design:
    The assignment asks for "a session, not isolated one-offs" but
    "memory resets between sessions; we're not asking for persistence
    across restarts or across users." A plain in-process dict keyed by
    session_id satisfies exactly that: it lives only as long as this
    server process runs, and is never written to disk. Restarting the
    server (or a fresh session_id) starts a fresh conversation, which
    matches the spec.

Run: uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph.build_graph import build_graph

app = FastAPI(title="Weather-Advisory Support Bot")

# Frontend is a plain static HTML/JS file opened directly (file://) or served
# from any localhost port -- keep CORS wide open for local dev/review use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph_app = build_graph()

# session_id -> BotState-shaped dict. In-memory only, by design (see above).
_sessions: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    matched_sop_id: Optional[str] = None
    resolved_location: Optional[str] = None


def _new_session() -> dict:
    return {"messages": [], "last_location": None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session_id = req.session_id or str(uuid.uuid4())
    session_state = _sessions.get(session_id, _new_session())

    session_state["user_message"] = req.message

    result = _graph_app.invoke(session_state)

    answer = result.get("final_answer", "(no answer produced)")

    # Persist session memory forward for the next turn -- same pattern as chat_cli.py.
    _sessions[session_id] = {
        "messages": result.get("messages", [])
        + [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": answer},
        ],
        "last_location": result.get("last_location"),
    }

    return ChatResponse(
        session_id=session_id,
        reply=answer,
        matched_sop_id=result.get("matched_sop_id"),
        resolved_location=result.get("resolved_location_name"),
    )


@app.post("/reset")
def reset(session_id: str) -> dict:
    """Explicitly drop a session's memory (useful for the frontend's 'New chat' button)."""
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
