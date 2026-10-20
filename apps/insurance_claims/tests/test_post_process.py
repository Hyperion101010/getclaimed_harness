from __future__ import annotations

from apps.insurance_claims.agent import reply_from_tool_results
from apps.insurance_claims.state import Phase, SessionState
from apps.insurance_claims.tools import execute_tool


def _margaret_post(**kwargs) -> SessionState:
    defaults = {
        "verified": True,
        "verified_party_id": "P9",
        "phase": Phase.POST_PROCESS,
        "selected_case_id": "CL-2048",
        "email_scenario": "default",
    }
    defaults.update(kwargs)
    return SessionState(**defaults)


def test_offer_does_not_send_without_consent():
    state = _margaret_post()
    result = execute_tool("send_email_summary", {"summary_text": "summary"}, state)
    assert result.ok is False
    assert result.error == "consent_required"
    assert state.email_status is None
    assert state.phase == Phase.POST_PROCESS
    reply = reply_from_tool_results(state, [result], "ok I'm done")
    assert "send" in reply.casefold() or "skip" in reply.casefold() or "email" in reply.casefold()


def test_skip_closes_without_email():
    state = _margaret_post()
    result = execute_tool("skip_email_summary", {}, state)
    assert result.ok is True
    assert state.phase == Phase.DONE
    assert state.done is True
    assert state.email_status is None
    reply = reply_from_tool_results(state, [result], "skip the email")
    assert "skip" in reply.casefold() or "complete" in reply.casefold()


def test_send_default_pending_then_approved():
    state = _margaret_post()
    first = execute_tool(
        "send_email_summary",
        {"summary_text": "summary", "confirmed": True},
        state,
    )
    assert first.ok is True
    assert first.data["status"] == "pending"
    assert state.email_status == "pending"
    assert state.phase == Phase.POST_PROCESS
    first_reply = reply_from_tool_results(state, [first], "yes, email me the summary")
    assert "pending" in first_reply.casefold()

    second = execute_tool(
        "send_email_summary",
        {"summary_text": "summary", "confirmed": True},
        state,
    )
    assert second.ok is True
    assert second.data["status"] == "approved"
    assert state.email_status == "approved"
    assert state.phase == Phase.DONE
    second_reply = reply_from_tool_results(state, [second], "did it send?")
    assert "margaret@email.com" in second_reply.casefold() or "approved" in second_reply.casefold()


def test_timeout_stays_pending():
    state = _margaret_post(email_scenario="timeout")
    last = None
    for _ in range(3):
        last = execute_tool(
            "send_email_summary",
            {"summary_text": "summary", "confirmed": True},
            state,
        )
        assert last.ok is True
        assert state.email_status == "pending"
        assert state.phase == Phase.POST_PROCESS
    assert last is not None
    assert last.data["status"] == "pending"


def test_summary_includes_grounded_claim_facts():
    state = _margaret_post()
    claim = execute_tool("get_claim", {"case_id": "CL-2048"}, state)
    emailed = execute_tool(
        "send_email_summary",
        {"summary_text": "Claim CL-2048 denied, missing pathology report", "confirmed": True},
        state,
    )
    reply = reply_from_tool_results(state, [claim, emailed], "send the email")
    assert "CL-2048" in reply
    assert "pathology" in reply.casefold() or "denied" in reply.casefold() or "pending" in reply.casefold()
