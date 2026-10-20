from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from apps.insurance_claims.context import (
    TurnContext,
    claim_list_item,
    emit_reply,
    recent_transcript,
)
from apps.insurance_claims.escalation import mark_escalated, should_escalate
from apps.insurance_claims.state import Phase, Sentiment, SessionState, TurnExtraction
from apps.insurance_claims.tools import execute_tool


class ResolveIntentOutcome(BaseModel):
    coarse_status: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    reply: str = ""
    advanced: bool = False


def _hint_filters(state: SessionState) -> dict[str, str]:
    hint = state.case_hint
    args: dict[str, str] = {}
    if hint is None:
        return args
    if hint.case_type:
        args["case_type"] = hint.case_type.value
    if hint.status:
        args["status"] = hint.status.value
    if hint.time_hint:
        args["time_hint"] = hint.time_hint.value
    return args


def _list_claims(state: SessionState, args: dict[str, str]) -> list[dict[str, Any]]:
    result = execute_tool("list_claims", args, state)
    if not result.ok:
        return []
    return list(result.data.get("claims") or [])


def filtered_claim_candidates(state: SessionState) -> list[dict[str, Any]]:
    filters = _hint_filters(state)
    if not filters:
        return []
    matched = _list_claims(state, filters)
    if matched:
        return matched
    without_time = {key: value for key, value in filters.items() if key != "time_hint"}
    if without_time != filters:
        matched = _list_claims(state, without_time)
        if len(matched) == 1:
            return matched
    case_only = {key: value for key, value in filters.items() if key == "case_type"}
    if case_only and case_only != without_time:
        matched = _list_claims(state, case_only)
        if len(matched) == 1:
            return matched
    return []


def has_unique_filtered_claim(state: SessionState) -> bool:
    return len(filtered_claim_candidates(state)) == 1


def _select_candidates(state: SessionState) -> tuple[list[dict[str, Any]], str]:
    filters = _hint_filters(state)
    if not filters:
        owned = _list_claims(state, {})
        if len(owned) == 1:
            return owned, "open"
        return [], "need_hint"

    matched = filtered_claim_candidates(state)
    if matched:
        return matched, "filtered"
    return [], "no_match"


def _instruction(coarse_status: str) -> str:
    if coarse_status == "resolved":
        return (
            "A single matching claim was selected. Confirm it using tool_results only "
            "and offer to explain status, denial reason, or next documents."
        )
    if coarse_status == "need_hint":
        return (
            "Identity is verified but there is not enough of a claim hint yet. Ask whether "
            "this is healthcare, dental, or auto, and whether it is denied, open, or closed. "
            "Do not list claim IDs or statuses."
        )
    if coarse_status == "no_match":
        return (
            "The remembered hint did not match a claim on file. Ask the caller to clarify "
            "claim type, status, or month. Do not list unrelated claims."
        )
    if coarse_status == "out_of_scope":
        return (
            "The last message is out of scope. Decline politely and ask which of their "
            "claims to look at, without listing claim IDs unless list_items is present."
        )
    return (
        "More than one claim still matches the hint. List only the candidates in list_items "
        "and ask which one to open. Do not invent extra claims."
    )


def _reply(
    state: SessionState,
    extraction: TurnExtraction,
    coarse_status: str,
    candidates: list[dict[str, Any]],
    user_message: str,
    *,
    use_llm_reply: bool,
) -> str:
    list_items: list[str] = []
    list_style = None
    if coarse_status == "ambiguous" and candidates:
        list_items = [claim_list_item(claim) for claim in candidates]
        list_style = "numbered"
    tool_results: list[dict[str, Any]] = []
    if coarse_status == "resolved" and candidates:
        tool_results = [{"claim": candidates[0]}]
    elif candidates:
        tool_results = [{"claims": candidates}]
    context = TurnContext(
        phase=Phase.RESOLVE_INTENT,
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
        list_style=list_style,
        list_items=list_items,
    )
    return emit_reply(context, use_llm=use_llm_reply)


def run_resolve_intent_turn(
    state: SessionState,
    extraction: TurnExtraction,
    user_message: str = "",
    *,
    use_llm_reply: bool = True,
) -> ResolveIntentOutcome:
    state.merge_extraction(extraction)

    if not extraction.in_scope:
        state.counters.out_of_scope += 1
    if extraction.sentiment == Sentiment.REFUSING:
        state.counters.refusals += 1
        state.counters.persuasion_attempts += 1
    if extraction.wants_human and extraction.sentiment == Sentiment.REFUSING:
        reply = mark_escalated(state)
        return ResolveIntentOutcome(coarse_status="escalated", reply=reply)

    if should_escalate(state):
        reply = mark_escalated(state)
        return ResolveIntentOutcome(coarse_status="escalated", reply=reply)

    if not extraction.in_scope:
        reply = _reply(
            state,
            extraction,
            "out_of_scope",
            [],
            user_message,
            use_llm_reply=use_llm_reply,
        )
        return ResolveIntentOutcome(coarse_status="out_of_scope", reply=reply)

    candidates, origin = _select_candidates(state)
    if len(candidates) == 1 and origin in {"filtered", "open"}:
        return _resolve_to(
            state,
            extraction,
            candidates,
            user_message,
            use_llm_reply=use_llm_reply,
        )

    coarse = origin if origin in {"need_hint", "no_match"} else "ambiguous"
    reply = _reply(
        state,
        extraction,
        coarse,
        candidates,
        user_message,
        use_llm_reply=use_llm_reply,
    )
    return ResolveIntentOutcome(
        coarse_status=coarse,
        candidates=candidates,
        reply=reply,
        advanced=False,
    )


def _resolve_to(
    state: SessionState,
    extraction: TurnExtraction,
    candidates: list[dict[str, Any]],
    user_message: str,
    *,
    use_llm_reply: bool,
) -> ResolveIntentOutcome:
    chosen = candidates[0]
    state.selected_case_id = chosen["case_id"]
    state.phase = Phase.PROCESS_CASE
    reply = _reply(
        state,
        extraction,
        "resolved",
        candidates,
        user_message,
        use_llm_reply=use_llm_reply,
    )
    return ResolveIntentOutcome(
        coarse_status="resolved",
        candidates=candidates,
        reply=reply,
        advanced=True,
    )
