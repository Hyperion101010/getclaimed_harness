from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from apps.insurance_claims.agent import ClaimsAgent
from apps.insurance_claims.config import PACKAGE_DIR, PROJECT_ROOT
from apps.insurance_claims.state import SessionState
from apps.insurance_claims.store import InMemorySessionStore

STATIC_DIR = PACKAGE_DIR / "static"

_agent: ClaimsAgent | None = None


def get_agent() -> ClaimsAgent:
    global _agent
    if _agent is None:
        _agent = ClaimsAgent(store=InMemorySessionStore(), use_llm_reply=True)
    return _agent


def reset_agent(agent: ClaimsAgent | None = None) -> None:
    global _agent
    _agent = agent


def _debug_payload(state: SessionState) -> dict[str, Any]:
    return {
        "phase": state.phase.value,
        "verified": state.verified,
        "verified_party_id": state.verified_party_id,
        "collected_pii": state.collected_pii.model_dump(),
        "match_counts": state.match_counts.model_dump(),
        "matched_count": state.match_counts.matched_count,
        "intent": state.intent.value if state.intent else None,
        "case_hint": state.case_hint.model_dump() if state.case_hint else None,
        "selected_case_id": state.selected_case_id,
        "consent_choice": state.consent_choice.value,
        "email_status": state.email_status,
        "escalated": state.escalated,
        "done": state.done,
        "sentiment": state.sentiment.value,
    }


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    phase: str
    verified: bool
    debug: dict[str, Any]


app = FastAPI(title="Insurance Claims SOP Agent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/logo.png")
def logo() -> FileResponse:
    """Serve the supplied brand asset without duplicating it in the package."""
    return FileResponse(PROJECT_ROOT / "logo.png", media_type="image/png")


@app.post("/session")
def create_session() -> dict[str, Any]:
    state = get_agent().start_session()
    return {"session_id": state.session_id, "debug": _debug_payload(state)}


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    state = get_agent().store.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"session_id": state.session_id, "debug": _debug_payload(state)}


@app.post("/chat")
def chat(body: ChatRequest) -> ChatResponse:
    agent = get_agent()
    session_id = body.session_id
    if not session_id:
        session_id = agent.start_session().session_id
    try:
        state, reply = agent.run_turn(session_id, body.message)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session") from None
    return ChatResponse(
        session_id=state.session_id,
        reply=reply,
        phase=state.phase.value,
        verified=state.verified,
        debug=_debug_payload(state),
    )
