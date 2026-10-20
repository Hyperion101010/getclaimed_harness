from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from apps.insurance_claims.context import TurnContext, emit_reply, recent_transcript
from apps.insurance_claims.escalation import mark_escalated, should_escalate
from apps.insurance_claims.fixtures_loader import get_fixtures
from apps.insurance_claims.state import CallerIntent, Phase, Sentiment, SessionState, TurnExtraction
from apps.insurance_claims.tools import execute_tool

DOC_INTENTS = {
    CallerIntent.DENIAL_QUESTION,
    CallerIntent.DOCUMENT_SUBMISSION,
    CallerIntent.NEXT_STEPS,
    CallerIntent.GENERAL_CLAIM_QUESTION,
}


class ProcessCaseOutcome(BaseModel):
    coarse_status: str
    reply: str = ""
    advanced: bool = False


def _load_claim(state: SessionState, case_id: str) -> dict[str, Any] | None:
    result = execute_tool("get_claim", {"case_id": case_id}, state)
    if not result.ok:
        return None
    return result.data.get("claim")


def _document_guidance(state: SessionState, claim: dict[str, Any]) -> list[dict[str, Any]]:
    guidance = []
    for name in claim.get("documents_needed") or []:
        result = execute_tool(
            "get_document_guidance",
            {"case_type": claim.get("case_type"), "document_name": name},
            state,
        )
        if result.ok:
            guidance.append({"document": name, **result.data})
    return guidance


def _instruction(coarse_status: str) -> str:
    if coarse_status == "out_of_scope":
        return (
            "The last message is out of scope. Politely decline and offer to continue "
            "with the selected claim only."
        )
    if coarse_status == "no_case":
        return "No claim is selected. Ask which claim to look at."
    if coarse_status == "done":
        return (
            "The caller is finished with this claim. Confirm wrap-up and mention that "
            "an email summary can be offered next. Do not invent extra outcomes."
        )
    return (
        "Answer using tool_results only: claim status, denial reason, documents needed, "
        "appeal deadline, amounts, and document guidance. Do not invent facts. "
        "If amounts appear, use field_descriptions for meaning. "
        "If list_items is present, those are the documents needed; do not inline them."
    )


def _reply(
    state: SessionState,
    extraction: TurnExtraction,
    coarse_status: str,
    tool_results: list[dict[str, Any]],
    user_message: str,
    *,
    use_llm_reply: bool,
) -> str:
    documents: list[str] = []
    for payload in tool_results:
        claim = payload.get("claim") if isinstance(payload, dict) else None
        if claim:
            documents = list(claim.get("documents_needed") or [])
            break
    context = TurnContext(
        phase=Phase.PROCESS_CASE,
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
        list_style="bullet" if documents else None,
        list_items=documents,
    )
    return emit_reply(context, use_llm=use_llm_reply)


def run_process_case_turn(
    state: SessionState,
    extraction: TurnExtraction,
    user_message: str = "",
    *,
    use_llm_reply: bool = True,
) -> ProcessCaseOutcome:
    state.merge_extraction(extraction)

    if not extraction.in_scope:
        state.counters.out_of_scope += 1
    if extraction.sentiment == Sentiment.REFUSING:
        state.counters.refusals += 1

    if should_escalate(state):
        return ProcessCaseOutcome(coarse_status="escalated", reply=mark_escalated(state))

    if not extraction.in_scope:
        reply = _reply(state, extraction, "out_of_scope", [], user_message, use_llm_reply=use_llm_reply)
        return ProcessCaseOutcome(coarse_status="out_of_scope", reply=reply)

    if not state.selected_case_id:
        state.phase = Phase.RESOLVE_INTENT
        reply = _reply(state, extraction, "no_case", [], user_message, use_llm_reply=use_llm_reply)
        return ProcessCaseOutcome(coarse_status="no_case", reply=reply, advanced=True)

    claim = _load_claim(state, state.selected_case_id)
    if claim is None:
        state.selected_case_id = None
        state.phase = Phase.RESOLVE_INTENT
        reply = _reply(state, extraction, "no_case", [], user_message, use_llm_reply=use_llm_reply)
        return ProcessCaseOutcome(coarse_status="no_case", reply=reply, advanced=True)

    tool_results: list[dict[str, Any]] = [{"claim": claim}]
    intent = extraction.intent or state.intent
    if intent in DOC_INTENTS or claim.get("documents_needed"):
        guidance = _document_guidance(state, claim)
        if guidance:
            tool_results.append({"document_guidance": guidance})

    schema = get_fixtures().claim_schema
    tool_results.append(
        {"field_descriptions": schema.model_dump().get("field_descriptions", {})}
    )

    if extraction.done_with_case:
        state.phase = Phase.POST_PROCESS
        reply = _reply(
            state, extraction, "done", tool_results, user_message, use_llm_reply=use_llm_reply
        )
        return ProcessCaseOutcome(coarse_status="done", reply=reply, advanced=True)

    reply = _reply(
        state, extraction, "answer", tool_results, user_message, use_llm_reply=use_llm_reply
    )
    return ProcessCaseOutcome(coarse_status="answer", reply=reply)
