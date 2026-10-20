from __future__ import annotations

from apps.insurance_claims.agent import ClaimsAgent, reply_from_tool_results
from apps.insurance_claims.state import Phase, SessionState
from apps.insurance_claims.store import InMemorySessionStore
from apps.insurance_claims.tools import execute_tool


def _margaret_on_denied_claim(**kwargs) -> SessionState:
    defaults = {
        "verified": True,
        "verified_party_id": "P9",
        "phase": Phase.PROCESS_CASE,
        "selected_case_id": "CL-2048",
    }
    defaults.update(kwargs)
    return SessionState(**defaults)


def test_answers_from_grounded_claim_and_documents():
    state = _margaret_on_denied_claim()
    claim = execute_tool("get_claim", {"case_id": "CL-2048"}, state)
    docs = execute_tool("get_document_guidance", {"case_id": "CL-2048"}, state)
    reply = reply_from_tool_results(
        state,
        [claim, docs],
        "Why was my claim denied and what do I need to send?",
    )
    assert state.phase == Phase.PROCESS_CASE
    assert "pathology" in reply.casefold()
    assert "2026-03-18" in reply
    assert "CL-2048" in reply
    assert "- pathology report" in reply
    assert "- office note" in reply


def test_finish_case_advances_to_post_process():
    state = _margaret_on_denied_claim()
    result = execute_tool("finish_case", {}, state)
    assert result.ok is True
    assert state.phase == Phase.POST_PROCESS


def test_can_switch_to_owned_claim():
    state = _margaret_on_denied_claim()
    result = execute_tool("get_claim", {"case_id": "CL-2102"}, state)
    assert result.ok is True
    assert state.selected_case_id == "CL-2102"
    reply = reply_from_tool_results(state, [result], "Actually look at CL-2102 instead")
    assert "CL-2102" in reply


def test_cannot_switch_to_other_party_claim():
    state = _margaret_on_denied_claim()
    execute_tool("get_claim", {"case_id": "CL-3001"}, state)
    assert state.selected_case_id == "CL-2048"


def test_escalate_from_process_case():
    store = InMemorySessionStore()
    session = store.create()
    session.verified = True
    session.verified_party_id = "P9"
    session.phase = Phase.PROCESS_CASE
    session.selected_case_id = "CL-2048"
    store.save(session)

    def calls(_state: SessionState, _message: str):
        return [("escalate_to_human", {"reason": "repeated off topic"})]

    agent = ClaimsAgent(store=store, use_llm_reply=False, scripted_calls=calls)
    state, reply = agent.run_turn(session.session_id, "what is RL?")
    assert state.escalated is True
    assert state.phase == Phase.ESCALATED
    assert "human" in reply.casefold() or "representative" in reply.casefold()
