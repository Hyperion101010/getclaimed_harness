from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from apps.insurance_claims.context import TurnContext, emit_reply, recent_transcript
from apps.insurance_claims.escalation import mark_escalated, should_escalate
from apps.insurance_claims.state import ConsentChoice, Phase, Sentiment, SessionState, TurnExtraction
from apps.insurance_claims.tools import execute_tool


class PostProcessOutcome(BaseModel):
    coarse_status: str
    reply: str = ""
    advanced: bool = False


def _load_claim(state: SessionState) -> dict[str, Any] | None:
    if not state.selected_case_id:
        return None
    result = execute_tool("get_claim", {"case_id": state.selected_case_id}, state)
    if not result.ok:
        return None
    return result.data.get("claim")


def _summary_text(state: SessionState, claim: dict[str, Any] | None) -> str:
    parts: list[str] = []
    if claim:
        parts.append(
            f"Claim {claim.get('case_id')} ({claim.get('case_type')}) status {claim.get('status')}"
        )
        if claim.get("denial_reason"):
            parts.append(f"Denial reason: {claim['denial_reason']}")
        if claim.get("documents_needed"):
            parts.append("Documents needed: " + ", ".join(claim["documents_needed"]))
        if claim.get("appeal_deadline"):
            parts.append(f"Appeal deadline: {claim['appeal_deadline']}")
    if state.intent:
        parts.append(f"Discussed: {state.intent.value}")
    return ". ".join(parts) if parts else "Conversation summary for the verified caller."


def _instruction(coarse_status: str) -> str:
    if coarse_status == "offer":
        return (
            "Offer to email a summary of what was discussed, claim outcome, and next steps. "
            "The caller must choose send or skip. Do not say an email was sent."
        )
    if coarse_status == "skipped":
        return "The caller declined the email. Confirm skip and close politely."
    if coarse_status == "sent_pending":
        return (
            "Consent was given and the email was submitted. Report the tool status as pending. "
            "Do not claim it was delivered."
        )
    if coarse_status == "sent_approved":
        return (
            "The email tool reports approved. Confirm it was sent to the address in tool_results "
            "and close."
        )
    if coarse_status == "sent_timeout":
        return (
            "The email tool is still pending after polling. Say it has not completed and a human "
            "can follow up. Do not invent a delivery."
        )
    if coarse_status == "out_of_scope":
        return (
            "Out of scope. Decline and ask whether to send or skip the email summary."
        )
    return "Ask whether to send or skip the email summary."


def _reply(
    state: SessionState,
    extraction: TurnExtraction,
    coarse_status: str,
    tool_results: list[dict[str, Any]],
    user_message: str,
    *,
    use_llm_reply: bool,
) -> str:
    context = TurnContext(
        phase=Phase.POST_PROCESS,
        identity_verified=True,
        instruction=_instruction(coarse_status),
        user_message=user_message,
        sentiment=state.sentiment.value,
        case_hint=state.case_hint.model_dump() if state.case_hint else None,
        intent=state.intent.value if state.intent else None,
        selected_case_id=state.selected_case_id,
        tool_results=tool_results,
        recent_transcript=recent_transcript(state),
        in_scope=extraction.in_scope,
    )
    return emit_reply(context, use_llm=use_llm_reply)


def _poll_email(state: SessionState, summary_text: str) -> dict[str, Any]:
    result = execute_tool(
        "send_email_summary",
        {
            "summary_text": summary_text,
            "scenario": state.email_scenario,
        },
        state,
    )
    return result.model_dump()


def run_post_process_turn(
    state: SessionState,
    extraction: TurnExtraction,
    user_message: str = "",
    *,
    use_llm_reply: bool = True,
) -> PostProcessOutcome:
    state.merge_extraction(extraction)

    if not extraction.in_scope:
        state.counters.out_of_scope += 1
    if extraction.sentiment == Sentiment.REFUSING:
        state.counters.refusals += 1

    if should_escalate(state):
        return PostProcessOutcome(coarse_status="escalated", reply=mark_escalated(state))

    if not extraction.in_scope and state.consent_choice == ConsentChoice.UNSET:
        reply = _reply(state, extraction, "out_of_scope", [], user_message, use_llm_reply=use_llm_reply)
        return PostProcessOutcome(coarse_status="out_of_scope", reply=reply)

    claim = _load_claim(state)
    summary = _summary_text(state, claim)
    tool_results: list[dict[str, Any]] = []
    if claim:
        tool_results.append({"claim": claim})

    if state.consent_choice == ConsentChoice.SKIP:
        state.phase = Phase.DONE
        state.done = True
        reply = _reply(
            state, extraction, "skipped", tool_results, user_message, use_llm_reply=use_llm_reply
        )
        return PostProcessOutcome(coarse_status="skipped", reply=reply, advanced=True)

    if state.consent_choice == ConsentChoice.SEND:
        email_result = _poll_email(state, summary)
        tool_results.append({"email": email_result})
        status = (email_result.get("data") or {}).get("status") if email_result.get("ok") else None

        if status == "approved":
            state.phase = Phase.DONE
            state.done = True
            reply = _reply(
                state,
                extraction,
                "sent_approved",
                tool_results,
                user_message,
                use_llm_reply=use_llm_reply,
            )
            return PostProcessOutcome(
                coarse_status="sent_approved", reply=reply, advanced=True
            )

        if state.email_scenario == "timeout" and (state.email_status_index or 0) >= 3:
            coarse = "sent_timeout"
        else:
            coarse = "sent_pending"
        reply = _reply(
            state, extraction, coarse, tool_results, user_message, use_llm_reply=use_llm_reply
        )
        return PostProcessOutcome(coarse_status=coarse, reply=reply)

    reply = _reply(
        state, extraction, "offer", tool_results, user_message, use_llm_reply=use_llm_reply
    )
    return PostProcessOutcome(coarse_status="offer", reply=reply)
