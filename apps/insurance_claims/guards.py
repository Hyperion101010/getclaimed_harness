from __future__ import annotations

from apps.insurance_claims.state import SessionState

SAFE_FALLBACK_UNVERIFIED = (
    "I can help once your identity is verified. Please share at least three of: "
    "full name, date of birth, phone, email, or the last four digits of your ID."
)


def apply_output_guard(reply: str, state: SessionState) -> tuple[str, bool]:
    if reply and reply.strip():
        return reply, False
    if state.verified:
        return reply, False
    return SAFE_FALLBACK_UNVERIFIED, True
