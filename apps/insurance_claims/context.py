from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from apps.insurance_claims.llm import structured_llm
from apps.insurance_claims.state import Phase, SessionState

_FORMATTING_NOTES_PATH = Path(__file__).with_name("formatting_notes.md")


def formatting_notes() -> str:
    try:
        return _FORMATTING_NOTES_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


REPLY_SYSTEM = """You are an insurance claims support agent.
Write the next user-facing message from the turn context only.

Hard rules (the system already enforced these; you must not contradict them):
- Do not claim the caller is verified unless identity_verified is true.
- Before verification, never mention claim IDs, amounts, denial reasons, documents, or statuses.
- After verification, only mention claim facts present in tool_results.
- Do not invent policy, claim, or document details.
- If identity_verified is false, ask for allowed identity fields still missing.
- If the caller is frustrated or refusing, acknowledge that first, then continue the required step.
- Stay in-scope: insurance claims support only.
- Do not skip required steps.

Formatting (the system will attach list_items; do not flatten them into a sentence):
- Put only intro/outro sentences in paragraphs.
- Do not write "any of the following: A, B, C, or D".
- Do not write inline numbered lists like "1. … 2. …" in paragraphs.
- If list_items is present, do not repeat those items in paragraphs.
- Identity fields: the UI shows a bullet list.
- Remaining claim candidates: the UI shows a numbered list.
- Documents needed: the UI shows a bullet list.

Be concise, natural, and specific to this turn.

""" + formatting_notes()

IDENTITY_FIELD_LABELS = {
    "name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "id_last4": "last 4 digits of SSN or national ID",
}


class AgentReply(BaseModel):
    paragraphs: list[str] = Field(
        default_factory=list,
        description="Intro and closing sentences only. Never put list items here.",
    )
    reply: str = Field(
        default="",
        description="Single paragraph fallback if paragraphs is empty",
    )


class TurnContext(BaseModel):
    """Facts the reply model may see. SOP code fills this; the LLM does not choose gates."""

    phase: Phase
    identity_verified: bool
    instruction: str
    user_message: str = ""
    sentiment: str = "neutral"
    missing_identity_fields: list[str] = Field(default_factory=list)
    collected_pii_present: list[str] = Field(default_factory=list)
    case_hint: Optional[dict[str, Any]] = None
    intent: Optional[str] = None
    selected_case_id: Optional[str] = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    recent_transcript: list[dict[str, str]] = Field(default_factory=list)
    in_scope: bool = True
    list_style: Optional[Literal["bullet", "numbered"]] = None
    list_items: list[str] = Field(default_factory=list)


def recent_transcript(state: SessionState, limit: int = 6) -> list[dict[str, str]]:
    turns = state.transcript[-limit:]
    return [{"role": t.role, "content": t.content} for t in turns]


def collected_pii_present(state: SessionState) -> list[str]:
    return [key for key, value in state.collected_pii.model_dump().items() if value]


def identity_list_items(fields: list[str]) -> list[str]:
    return [IDENTITY_FIELD_LABELS.get(field, field) for field in fields]


def claim_list_item(claim: dict[str, Any]) -> str:
    case_id = claim.get("case_id") or ""
    case_type = claim.get("case_type") or "claim"
    created = claim.get("created_at") or ""
    status = claim.get("status") or ""
    return f"**{case_id}**: {case_type} claim, opened {created}, {status}"


def format_list(style: Optional[str], items: list[str]) -> str:
    if not items:
        return ""
    if style == "numbered":
        return "\n".join(f"{index}. {item}" for index, item in enumerate(items, 1))
    return "\n".join(f"- {item}" for item in items)


def assemble_message(
    paragraphs: list[str],
    list_style: Optional[str] = None,
    list_items: Optional[list[str]] = None,
) -> str:
    chunks: list[str] = []
    cleaned = _paragraphs_without_lists(paragraphs)
    if cleaned:
        chunks.append("\n\n".join(cleaned))
    formatted = format_list(list_style, list_items or [])
    if formatted:
        chunks.append(formatted)
    return "\n\n".join(chunks).strip()


def _is_list_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith(("- ", "* ")):
        return True
    if len(stripped) >= 3 and stripped[0].isdigit() and stripped[1:3] in {". ", ") "}:
        return True
    return False


def _paragraphs_without_lists(texts: list[str]) -> list[str]:
    kept: list[str] = []
    for text in texts:
        if not text or not str(text).strip():
            continue
        lines = [
            line.strip()
            for line in str(text).splitlines()
            if line.strip() and not _is_list_line(line)
        ]
        if lines:
            kept.append(" ".join(lines))
    return kept


def fallback_reply(context: TurnContext) -> str:
    """Factual fallback if the reply model is unavailable. Not customer-scripted."""
    items = list(context.list_items)
    style = context.list_style
    if not items and context.missing_identity_fields:
        items = identity_list_items(context.missing_identity_fields)
        style = style or "bullet"
    parts = [context.instruction]
    formatted = format_list(style, items)
    if formatted:
        parts.append(formatted)
    if context.identity_verified and context.selected_case_id:
        parts.append(f"Selected claim: {context.selected_case_id}.")
    if context.identity_verified and context.tool_results:
        parts.append(str(context.tool_results))
    return "\n\n".join(part for part in parts if part).strip()


def compose_followup(context: TurnContext) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPLY_SYSTEM),
            (
                "human",
                "Turn context JSON:\n{context}\n\n"
                "Write paragraphs only. The system will attach list_items.",
            ),
        ]
    )
    try:
        chain = prompt | structured_llm(AgentReply, temperature=0.3)
        result = chain.invoke({"context": context.model_dump_json()})
    except Exception:
        return fallback_reply(context)
    parsed: AgentReply | None = None
    if isinstance(result, AgentReply):
        parsed = result
    else:
        try:
            parsed = AgentReply.model_validate(result)
        except Exception:
            parsed = None
    if parsed is None:
        return fallback_reply(context)
    paragraphs = list(parsed.paragraphs)
    if not paragraphs and parsed.reply.strip():
        paragraphs = [parsed.reply.strip()]
    assembled = assemble_message(paragraphs, context.list_style, context.list_items)
    return assembled or fallback_reply(context)


def emit_reply(context: TurnContext, *, use_llm: bool = True) -> str:
    if use_llm:
        return compose_followup(context)
    return fallback_reply(context)
