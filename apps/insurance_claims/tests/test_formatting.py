from __future__ import annotations

from apps.insurance_claims.context import assemble_message, claim_list_item, format_list


def test_claim_list_item_format():
    item = claim_list_item(
        {
            "case_id": "CL-2048",
            "case_type": "healthcare",
            "created_at": "2026-01-12",
            "status": "denied",
        }
    )
    assert item == "**CL-2048**: healthcare claim, opened 2026-01-12, denied"


def test_assemble_message_keeps_paragraphs_and_list():
    text = assemble_message(
        ["I found a matching claim."],
        list_style="numbered",
        list_items=["**CL-2048**: healthcare claim, opened 2026-01-12, denied"],
    )
    assert "I found a matching claim." in text
    assert "1. **CL-2048**" in text


def test_format_list_bullets():
    assert format_list("bullet", ["date of birth", "phone number"]) == (
        "- date of birth\n- phone number"
    )
