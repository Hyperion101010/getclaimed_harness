from __future__ import annotations

from fastapi.testclient import TestClient

from apps.insurance_claims.agent import ClaimsAgent
from apps.insurance_claims.api import app, reset_agent
from apps.insurance_claims.state import SessionState
from apps.insurance_claims.store import InMemorySessionStore


def _margaret_scripted(_session: SessionState, _message: str):
    return [
        (
            "remember",
            {"case_type": "healthcare", "status": "denied", "time_hint": "january"},
        ),
        (
            "verify_identity",
            {
                "name": "Margaret Chen",
                "dob": "1985-03-15",
                "id_last4": "4472",
                "policy_number": "POL-9921",
            },
        ),
        ("list_claims", {}),
    ]


def setup_function() -> None:
    reset_agent(
        ClaimsAgent(
            store=InMemorySessionStore(),
            use_llm_reply=False,
            scripted_calls=_margaret_scripted,
        )
    )


def teardown_function() -> None:
    reset_agent(None)


def test_healthz_and_index():
    client = TestClient(app)
    assert client.get("/healthz").json()["status"] == "ok"
    home = client.get("/")
    assert home.status_code == 200
    assert "GetClaim" in home.text


def test_session_create_and_fetch():
    client = TestClient(app)
    created = client.post("/session").json()
    session_id = created["session_id"]
    assert created["debug"]["phase"] == "VERIFY_ID"
    assert created["debug"]["verified"] is False
    fetched = client.get(f"/session/{session_id}").json()
    assert fetched["session_id"] == session_id


def test_chat_unknown_session_404():
    client = TestClient(app)
    res = client.post("/chat", json={"session_id": "missing", "message": "hello"})
    assert res.status_code == 404


def test_chat_margaret_verifies_and_resolves():
    client = TestClient(app)

    first = client.post("/chat", json={"message": "I'm Margaret Chen about my denied claim"}).json()
    assert first["verified"] is True
    assert first["phase"] == "PROCESS_CASE"
    assert first["debug"]["case_hint"]["case_type"] == "healthcare"
    assert first["debug"]["selected_case_id"] == "CL-2048"
    assert "CL-2048" in first["reply"]
    assert "CL-2102" not in first["reply"]
    assert "CL-1899" not in first["reply"]
    assert "CL-2011" not in first["reply"]
