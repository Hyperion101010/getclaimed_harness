from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

_FORMATTING_NOTES_PATH = Path(__file__).with_name("formatting_notes.md")

IDENTITY_FIELD_LABELS = {
    "name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "id_last4": "last 4 digits of SSN or national ID",
}


def formatting_notes() -> str:
    try:
        return _FORMATTING_NOTES_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


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
