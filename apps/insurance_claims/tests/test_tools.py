from __future__ import annotations

from apps.insurance_claims.state import CaseHint, CaseType, ClaimStatus, Phase, SessionState, TimeHint
from apps.insurance_claims.tools import execute_tool


def _verified_margaret(**kwargs) -> SessionState:
    defaults = {
        "verified": True,
        "verified_party_id": "P9",
        "phase": Phase.PROCESS_CASE,
    }
    defaults.update(kwargs)
    return SessionState(**defaults)


def test_claim_tools_reject_when_unverified():
    state = SessionState(phase=Phase.PROCESS_CASE)

    for tool_name, args in [
        ("list_claims", {}),
        ("get_claim", {"case_id": "CL-2048"}),
        ("get_document_guidance", {"document_name": "pathology report"}),
        ("send_email_summary", {"summary_text": "hi", "confirmed": True}),
    ]:
        result = execute_tool(tool_name, args, state)
        assert result.ok is False
        assert result.error == "identity_not_verified"
        assert "pathology" not in str(result.data).casefold()


def test_claim_tools_blocked_in_verify_id_phase_even_if_verified_flag_set():
    state = SessionState(
        phase=Phase.VERIFY_ID,
        verified=True,
        verified_party_id="P9",
    )
    result = execute_tool("get_claim", {"case_id": "CL-2048"}, state)
    assert result.ok is False
    assert result.error == "tool_not_allowed_in_phase:VERIFY_ID"


def test_verify_identity_allowed_before_verification():
    state = SessionState(phase=Phase.VERIFY_ID)
    result = execute_tool(
        "verify_identity",
        {
            "name": "Margaret Chen",
            "dob": "1985-03-15",
            "id_last4": "4472",
            "policy_number": "POL-9921",
        },
        state,
    )
    assert result.ok is True
    assert result.data["verified"] is True
    assert state.verified is True
    assert state.verified_party_id == "P9"
    assert state.phase == Phase.RESOLVE_INTENT


def test_verify_identity_ignores_verified_flag_in_args():
    state = SessionState(phase=Phase.VERIFY_ID)
    result = execute_tool(
        "verify_identity",
        {"verified": True, "name": "Margaret Chen"},
        state,
    )
    assert result.ok is True
    assert result.data["verified"] is False
    assert state.verified is False
    assert state.phase == Phase.VERIFY_ID


def test_remember_during_verify_id_then_unique_claim():
    state = SessionState(phase=Phase.VERIFY_ID)
    remembered = execute_tool(
        "remember",
        {"case_type": "healthcare", "status": "denied", "time_hint": "january"},
        state,
    )
    assert remembered.ok is True
    assert state.phase == Phase.VERIFY_ID
    assert state.case_hint is not None
    assert state.case_hint.case_type == CaseType.HEALTHCARE

    execute_tool(
        "verify_identity",
        {
            "name": "Margaret Chen",
            "dob": "1985-03-15",
            "id_last4": "4472",
            "policy_number": "POL-9921",
        },
        state,
    )
    listed = execute_tool("list_claims", {}, state)
    assert listed.ok is True
    assert listed.data["count"] == 1
    assert listed.data["claims"][0]["case_id"] == "CL-2048"
    assert state.selected_case_id == "CL-2048"
    assert state.phase == Phase.PROCESS_CASE


def test_list_claims_filters_denied_healthcare_january():
    state = _verified_margaret(phase=Phase.RESOLVE_INTENT)
    result = execute_tool(
        "list_claims",
        {"case_type": "healthcare", "status": "denied", "time_hint": "january"},
        state,
    )
    assert result.ok is True
    assert result.data["count"] == 1
    assert result.data["claims"][0]["case_id"] == "CL-2048"


def test_list_claims_all_true_skips_stored_hint():
    state = _verified_margaret(
        phase=Phase.RESOLVE_INTENT,
        case_hint=CaseHint(
            case_type=CaseType.HEALTHCARE,
            status=ClaimStatus.DENIED,
            time_hint=TimeHint.JANUARY,
        ),
    )
    result = execute_tool("list_claims", {"all": True}, state)
    assert result.ok is True
    assert result.data["count"] == 4
    ids = {claim["case_id"] for claim in result.data["claims"]}
    assert ids == {"CL-2048", "CL-2102", "CL-1899", "CL-2011"}
    assert state.selected_case_id is None


def test_ava_list_claims_is_empty():
    state = SessionState(
        verified=True,
        verified_party_id="P7",
        phase=Phase.RESOLVE_INTENT,
    )
    result = execute_tool("list_claims", {"all": True}, state)
    assert result.ok is True
    assert result.data["count"] == 0
    assert result.data["claims"] == []
    assert state.selected_case_id is None
    assert state.phase == Phase.RESOLVE_INTENT


def test_get_claim_returns_grounded_denial_fields():
    state = _verified_margaret()
    result = execute_tool("get_claim", {"case_id": "CL-2048"}, state)
    assert result.ok is True
    claim = result.data["claim"]
    assert claim["status"] == "denied"
    assert "pathology report" in claim["denial_reason"]
    assert "pathology report" in claim["documents_needed"]
    assert claim["appeal_deadline"] == "2026-03-18"
    assert state.selected_case_id == "CL-2048"


def test_get_claim_rejects_other_party_claim():
    state = _verified_margaret()
    result = execute_tool("get_claim", {"case_id": "CL-3001"}, state)
    assert result.ok is False
    assert result.error == "claim_not_owned_by_verified_party"


def test_get_document_guidance_matches_pathology_report():
    state = _verified_margaret()
    result = execute_tool(
        "get_document_guidance",
        {"case_type": "healthcare", "document_name": "pathology report"},
        state,
    )
    assert result.ok is True
    assert "document_guidance" in result.data
    assert "pathology" in result.data["document_guidance"]["document"].casefold()


def test_get_document_guidance_by_case_id():
    state = _verified_margaret()
    result = execute_tool("get_document_guidance", {"case_id": "CL-2048"}, state)
    assert result.ok is True
    documents = result.data["documents"]
    names = [item["document"] for item in documents]
    assert "pathology report" in names
    assert "office note" in names


def test_send_email_requires_explicit_confirmation():
    state = _verified_margaret(phase=Phase.POST_PROCESS)
    result = execute_tool(
        "send_email_summary",
        {"summary_text": "summary"},
        state,
    )
    assert result.ok is False
    assert result.error == "consent_required"
    assert state.email_status is None


def test_send_email_default_scenario_pending_then_approved():
    state = _verified_margaret(
        phase=Phase.POST_PROCESS,
        email_scenario="default",
    )
    first = execute_tool(
        "send_email_summary",
        {"summary_text": "summary", "confirmed": True},
        state,
    )
    assert first.ok is True
    assert first.data["status"] == "pending"
    assert state.email_status == "pending"

    second = execute_tool(
        "send_email_summary",
        {"summary_text": "summary", "confirmed": True},
        state,
    )
    assert second.ok is True
    assert second.data["status"] == "approved"
    assert state.email_status == "approved"
    assert state.phase == Phase.DONE


def test_send_email_timeout_scenario_stays_pending():
    state = _verified_margaret(
        phase=Phase.POST_PROCESS,
        email_scenario="timeout",
    )
    for _ in range(3):
        result = execute_tool(
            "send_email_summary",
            {"summary_text": "summary", "confirmed": True},
            state,
        )
        assert result.ok is True
        assert result.data["status"] == "pending"


def test_escalate_to_human_sets_phase():
    state = SessionState(phase=Phase.VERIFY_ID)
    result = execute_tool("escalate_to_human", {"reason": "caller asked"}, state)
    assert result.ok is True
    assert state.phase == Phase.ESCALATED
    assert state.escalated is True
    assert state.done is True
