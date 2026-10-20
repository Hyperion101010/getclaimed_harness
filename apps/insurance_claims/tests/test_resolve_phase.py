from __future__ import annotations

import pytest
from pydantic import ValidationError

from apps.insurance_claims.agent import ClaimsAgent, reply_from_tool_results
from apps.insurance_claims.state import (
    CaseHint,
    CaseType,
    ClaimStatus,
    Phase,
    SessionState,
    TimeHint,
)
from apps.insurance_claims.store import InMemorySessionStore
from apps.insurance_claims.tools import execute_tool


def _verified_margaret(**kwargs) -> SessionState:
    defaults = {
        "verified": True,
        "verified_party_id": "P9",
        "phase": Phase.RESOLVE_INTENT,
    }
    defaults.update(kwargs)
    return SessionState(**defaults)


def test_case_hint_rejects_non_fixture_case_type():
    with pytest.raises(ValidationError):
        CaseHint(case_type="denied_healthcare_claim")


def test_remembered_hint_resolves_to_january_denied_healthcare():
    state = _verified_margaret(
        case_hint=CaseHint(
            case_type=CaseType.HEALTHCARE,
            status=ClaimStatus.DENIED,
            time_hint=TimeHint.JANUARY,
        )
    )
    listed = execute_tool("list_claims", {}, state)
    assert listed.ok is True
    assert listed.data["count"] == 1
    assert listed.data["claims"][0]["case_id"] == "CL-2048"
    assert state.selected_case_id == "CL-2048"
    assert state.phase == Phase.PROCESS_CASE
    reply = reply_from_tool_results(state, [listed], "yes, that's the one")
    assert "CL-2048" in reply


def test_no_hint_does_not_dump_claims_in_fallback():
    state = _verified_margaret()
    reply = reply_from_tool_results(state, [], "I have a question about a claim")
    assert state.selected_case_id is None
    assert state.phase == Phase.RESOLVE_INTENT
    assert "CL-2048" not in reply
    assert "CL-2102" not in reply
    assert "CL-1899" not in reply
    assert "CL-2011" not in reply


def test_partial_hint_lists_only_matching_claims():
    state = _verified_margaret(case_hint=CaseHint(case_type=CaseType.HEALTHCARE))
    listed = execute_tool("list_claims", {}, state)
    assert listed.ok is True
    ids = {claim["case_id"] for claim in listed.data["claims"]}
    assert ids == {"CL-2048", "CL-2011"}
    reply = reply_from_tool_results(state, [listed], "it's a healthcare claim")
    assert "CL-2048" in reply
    assert "CL-2011" in reply
    assert "CL-2102" not in reply
    assert "CL-1899" not in reply
    assert "\n1. " in reply
    assert "\n2. " in reply


def test_explicit_all_flag_shows_owned_claims():
    state = _verified_margaret()
    listed = execute_tool("list_claims", {"all": True}, state)
    assert listed.ok is True
    ids = {claim["case_id"] for claim in listed.data["claims"]}
    assert {"CL-2048", "CL-2102"} <= ids
    reply = reply_from_tool_results(state, [listed], "can you list all my claims?")
    assert "CL-2048" in reply
    assert "CL-2102" in reply
    assert "\n1. " in reply


def test_get_claim_selects_named_case():
    state = _verified_margaret()
    result = execute_tool("get_claim", {"case_id": "CL-2048"}, state)
    assert result.ok is True
    assert state.selected_case_id == "CL-2048"
    assert state.phase == Phase.PROCESS_CASE


def test_cannot_select_another_partys_claim():
    state = _verified_margaret()
    result = execute_tool("get_claim", {"case_id": "CL-3001"}, state)
    assert result.ok is False
    assert state.selected_case_id is None
    assert state.phase == Phase.RESOLVE_INTENT


def test_ava_empty_list_does_not_invent_menu():
    def calls(_session: SessionState, _message: str):
        return [("list_claims", {"all": True})]

    store = InMemorySessionStore()
    session = store.create()
    session.verified = True
    session.verified_party_id = "P7"
    session.phase = Phase.RESOLVE_INTENT
    store.save(session)
    agent = ClaimsAgent(store=store, use_llm_reply=False, scripted_calls=calls)
    state, reply = agent.run_turn(session.session_id, "What all claims I have")
    assert state.selected_case_id is None
    assert "CL-" not in reply
    assert "list_items" not in reply
