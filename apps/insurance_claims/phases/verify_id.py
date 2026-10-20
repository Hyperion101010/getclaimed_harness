from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from apps.insurance_claims.context import (
    TurnContext,
    collected_pii_present,
    emit_reply,
    identity_list_items,
    recent_transcript,
)
from apps.insurance_claims.escalation import mark_escalated, should_escalate
from apps.insurance_claims.phases.resolve_intent import has_unique_filtered_claim
from apps.insurance_claims.state import (
    Phase,
    Sentiment,
    SessionState,
    TurnExtraction,
)
from apps.insurance_claims.verifier import VerificationResult, apply_verification, verify_identity

IDENTITY_FIELDS = ("name", "dob", "phone", "email", "id_last4")

IDENTITY_HELP = (
    "The caller can verify with any three of: full name, date of birth, phone, email, "
    "or the last four digits of SSN/national ID."
)


class VerifyIdOutcome(BaseModel):
    coarse_status: str
    missing_fields: list[str] = Field(default_factory=list)
    verification: Optional[VerificationResult] = None
    reply: str = ""
    advanced: bool = False


def missing_identity_fields(state: SessionState) -> list[str]:
    missing = []
    data = state.collected_pii.model_dump()
    for field in IDENTITY_FIELDS:
        if not data.get(field):
            missing.append(field)
    return missing


def filled_identity_count(state: SessionState) -> int:
    data = state.collected_pii.model_dump()
    return sum(1 for field in IDENTITY_FIELDS if data.get(field))


def build_coarse_status(
    state: SessionState,
    verification: VerificationResult,
    extraction: TurnExtraction,
) -> tuple[str, list[str]]:
    missing = missing_identity_fields(state)

    if state.escalated or state.phase == Phase.ESCALATED:
        return "escalated", missing
    if not extraction.in_scope:
        return "out_of_scope", missing
    if extraction.sentiment == Sentiment.REFUSING:
        return "refusing", missing
    if verification.verified:
        return "verified", []
    if verification.reason == "policy_not_found":
        return "policy_not_found", missing
    if filled_identity_count(state) == 0:
        return "need_identity", missing
    if verification.matched_count > 0:
        return "need_more_matches", missing
    return "need_more_details", missing


def _instruction(coarse_status: str) -> str:
    if coarse_status == "verified":
        return (
            "Identity is verified. Acknowledge that and continue using any remembered "
            "claim hint. Do not quote claim file details yet."
        )
    if coarse_status == "out_of_scope":
        return (
            "The last message is out of scope. Politely decline and steer back to "
            "identity verification for insurance claims support."
        )
    if coarse_status == "refusing":
        return (
            f"The caller is refusing verification. Explain why it is required. {IDENTITY_HELP} "
            "Offer a human representative if they still will not continue."
        )
    if coarse_status == "policy_not_found":
        return (
            "The policy number did not match a record. Ask them to recheck it or provide "
            f"other identity fields. {IDENTITY_HELP}"
        )
    if coarse_status == "need_more_matches":
        return (
            "Some identity fields were collected but there are not yet three matching fields. "
            "Ask for more of the missing fields from list_items. Do not write them as a comma sentence."
        )
    return (
        "Identity is not verified yet. Ask for identity details before any claim information. "
        "Use list_items as a bullet list of allowed fields. Do not write them as a comma sentence."
    )


def _reply_from_context(
    state: SessionState,
    extraction: TurnExtraction,
    coarse_status: str,
    missing: list[str],
    user_message: str,
    *,
    use_llm_reply: bool,
) -> str:
    show_fields = (not state.verified) and bool(missing)
    context = TurnContext(
        phase=Phase.VERIFY_ID,
        identity_verified=state.verified,
        instruction=_instruction(coarse_status),
        user_message=user_message,
        sentiment=state.sentiment.value,
        missing_identity_fields=missing,
        collected_pii_present=collected_pii_present(state),
        case_hint=state.case_hint.model_dump() if state.case_hint else None,
        intent=state.intent.value if state.intent else None,
        recent_transcript=recent_transcript(state),
        in_scope=extraction.in_scope,
        list_style="bullet" if show_fields else None,
        list_items=identity_list_items(missing) if show_fields else [],
    )
    return emit_reply(context, use_llm=use_llm_reply)


def run_verify_id_turn(
    state: SessionState,
    extraction: TurnExtraction,
    user_message: str = "",
    *,
    use_llm_reply: bool = True,
) -> VerifyIdOutcome:
    state.merge_extraction(extraction)

    if not extraction.in_scope:
        state.counters.out_of_scope += 1
    if extraction.sentiment == Sentiment.REFUSING:
        state.counters.refusals += 1
        state.counters.persuasion_attempts += 1
    if extraction.sentiment in {Sentiment.FRUSTRATED, Sentiment.ANGRY}:
        state.counters.persuasion_attempts += 1

    if should_escalate(state):
        reply = mark_escalated(state)
        return VerifyIdOutcome(
            coarse_status="escalated",
            missing_fields=missing_identity_fields(state),
            reply=reply,
            advanced=False,
        )

    if extraction.wants_human and extraction.sentiment == Sentiment.REFUSING:
        reply = mark_escalated(state)
        return VerifyIdOutcome(
            coarse_status="escalated",
            missing_fields=missing_identity_fields(state),
            reply=reply,
            advanced=False,
        )

    verification = verify_identity(state.collected_pii)
    previously_verified = state.verified
    apply_verification(state, verification)

    if (
        not verification.verified
        and filled_identity_count(state) >= 3
        and verification.reason in {"insufficient_matches", "policy_not_found", "ambiguous_match"}
    ):
        state.counters.failed_verification += 1

    if should_escalate(state):
        reply = mark_escalated(state)
        return VerifyIdOutcome(
            coarse_status="escalated",
            missing_fields=missing_identity_fields(state),
            verification=verification,
            reply=reply,
            advanced=False,
        )

    advanced = False
    if verification.verified and not previously_verified:
        state.phase = Phase.RESOLVE_INTENT
        advanced = True
        if has_unique_filtered_claim(state):
            return VerifyIdOutcome(
                coarse_status="verified",
                missing_fields=[],
                verification=verification,
                reply="",
                advanced=True,
            )

    coarse_status, missing = build_coarse_status(state, verification, extraction)
    reply = _reply_from_context(
        state,
        extraction,
        coarse_status,
        missing,
        user_message,
        use_llm_reply=use_llm_reply,
    )

    if missing:
        state.last_prompted_field = missing[0]
    elif verification.verified:
        state.last_prompted_field = None

    return VerifyIdOutcome(
        coarse_status=coarse_status,
        missing_fields=missing,
        verification=verification,
        reply=reply,
        advanced=advanced,
    )
