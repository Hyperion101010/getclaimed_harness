from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Policyholder(BaseModel):
    party_id: str
    name: str
    policy_number: str
    dob: str
    id_type: str
    id_last4: str
    phone: str
    email: str
    name_aliases: list[str] = Field(default_factory=list)
    phone_aliases: list[str] = Field(default_factory=list)
    email_aliases: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    case_id: str
    party_id: str
    case_type: str
    created_at: str
    status: str
    summary: str
    expected_reimbursement_amount: str
    allowed_max_amount: str
    net_pay: str
    net_fee: str
    denial_reason: Optional[str] = None
    documents_needed: list[str] = Field(default_factory=list)
    appeal_deadline: Optional[str] = None


class Representative(BaseModel):
    rep_name: str
    relationship: str
    buyer_name: str
    buyer_party_id: str


class ConsentScenario(BaseModel):
    status_sequence: list[str]


class ClaimSchemaField(BaseModel):
    type: str
    example: str
    description: str


class ClaimSchemaDoc(BaseModel):
    notes: list[str] = Field(default_factory=list)
    field_descriptions: dict[str, ClaimSchemaField] = Field(default_factory=dict)


class DocumentGuidelineBundle(BaseModel):
    """Loose model: guideline JSON is nested and topic-driven."""

    data: dict[str, Any]
