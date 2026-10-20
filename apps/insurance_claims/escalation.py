from __future__ import annotations

from apps.insurance_claims.state import Phase, SessionState

FAILED_VERIFICATION_LIMIT = 3

ESCALATION_MESSAGE = (
    "I understand this has been difficult. I'm connecting you with a human "
    "representative who can help you further. Please stay on the line."
)


def should_escalate(state: SessionState) -> bool:
    if state.escalated or state.phase == Phase.ESCALATED:
        return True
    return state.counters.failed_verification >= FAILED_VERIFICATION_LIMIT


def mark_escalated(state: SessionState) -> str:
    state.escalated = True
    state.phase = Phase.ESCALATED
    state.done = True
    return ESCALATION_MESSAGE
