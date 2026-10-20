from __future__ import annotations

from apps.insurance_claims.state import Phase, SessionState

FAILED_VERIFICATION_LIMIT = 3
OUT_OF_SCOPE_LIMIT = 3
REFUSAL_LIMIT = 3
PERSUASION_LIMIT = 2

ESCALATION_MESSAGE = (
    "I understand this has been difficult. I'm connecting you with a human "
    "representative who can help you further. Please stay on the line."
)


def should_escalate(state: SessionState) -> bool:
    if state.escalated or state.phase == Phase.ESCALATED:
        return True
    c = state.counters
    if c.failed_verification >= FAILED_VERIFICATION_LIMIT:
        return True
    if c.out_of_scope >= OUT_OF_SCOPE_LIMIT:
        return True
    if c.refusals >= REFUSAL_LIMIT:
        return True
    if c.persuasion_attempts >= PERSUASION_LIMIT and c.refusals >= 1:
        return True
    return False


def mark_escalated(state: SessionState) -> str:
    state.escalated = True
    state.phase = Phase.ESCALATED
    state.done = True
    return ESCALATION_MESSAGE
