from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from apps.insurance_claims.llm import structured_llm
from apps.insurance_claims.state import SessionState, TurnExtraction

EXTRACTION_SYSTEM = """You extract structured slots for an insurance claims support agent.
The user message may be typed text OR noisy speech-to-text transcription. Interpret both.

Rules:
- Return only fields supported by the user message or a clear answer to the last agent question.
- Use null for anything not stated. Do not invent values.
- Do not decide verification. Do not claim the user is verified.
- Capture case hints even during identity verification, including when the same utterance also has PII.
- case_hint.case_type must be one of: healthcare, dental, auto (never a phrase like "denied healthcare claim").
- case_hint.status must be one of: denied, open, closed.
- case_hint.time_hint must be a lowercase month name (january, february, ...).
- intent must be one of: denial_question, status_inquiry, document_submission, next_steps, general_claim_question, other.
- Map consent: send/yes/email me -> send; skip/no/don't email -> skip.
- Unrelated topics (e.g. what is RL, reinforcement learning) -> in_scope=false.
- Sentiment: frustrated, anxious, angry, confused, refusing, or neutral.
- Bare answers after last_prompted_field fill that field (e.g. "4472" or "four four seven two" after SSN last four -> id_last4).

Transcription / spoken-form handling:
- Spoken dates: "March fifteenth eighty-five" or "March 15th 1985" -> dob 1985-03-15 (YYYY-MM-DD).
- Spelled or grouped digits: "four four seven two", "four thousand four hundred seventy two" -> 4472.
- Policy numbers may be spoken letter-by-letter: "P O L nine nine two one" / "policy 9921" -> POL-9921 when clearly a policy number.
- Ignore filler, false starts, and STT artifacts; extract the intended values.
- Normalize DOB to YYYY-MM-DD when possible. Digits-only last4 should be four digits."""


def _last_assistant_text(state: SessionState) -> str:
    for turn in reversed(state.transcript):
        if turn.role == "assistant":
            return turn.content
    return ""


def extract_turn(user_message: str, state: SessionState) -> TurnExtraction:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_SYSTEM),
            (
                "human",
                "Phase: {phase}\n"
                "Already collected PII (may be partial): {collected_pii}\n"
                "Last agent message: {last_agent_message}\n"
                "Last prompted field: {last_prompted_field}\n\n"
                "User message:\n{user_message}",
            ),
        ]
    )
    try:
        result = (prompt | structured_llm(TurnExtraction, temperature=0)).invoke(
            {
                "phase": state.phase.value,
                "collected_pii": state.collected_pii.model_dump_json(),
                "last_agent_message": _last_assistant_text(state),
                "last_prompted_field": state.last_prompted_field or "",
                "user_message": user_message,
            }
        )
    except Exception:
        return TurnExtraction()
    if isinstance(result, TurnExtraction):
        return result
    return TurnExtraction.model_validate(result)
