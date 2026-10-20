from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    DONE = "DONE"
    ESCALATED = "ESCALATED"


class ConsentChoice(str, Enum):
    UNSET = "unset"
    SEND = "send"
    SKIP = "skip"


class CaseType(str, Enum):
    HEALTHCARE = "healthcare"
    DENTAL = "dental"
    AUTO = "auto"


class ClaimStatus(str, Enum):
    DENIED = "denied"
    OPEN = "open"
    CLOSED = "closed"


class TimeHint(str, Enum):
    JANUARY = "january"
    FEBRUARY = "february"
    MARCH = "march"
    APRIL = "april"
    MAY = "may"
    JUNE = "june"
    JULY = "july"
    AUGUST = "august"
    SEPTEMBER = "september"
    OCTOBER = "october"
    NOVEMBER = "november"
    DECEMBER = "december"


class CollectedPII(BaseModel):
    name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    id_last4: Optional[str] = None
    policy_number: Optional[str] = None


class CaseHint(BaseModel):
    case_type: Optional[CaseType] = None
    status: Optional[ClaimStatus] = None
    time_hint: Optional[TimeHint] = None
    raw: Optional[str] = None


class FieldMatchCounts(BaseModel):
    name: bool = False
    dob: bool = False
    phone: bool = False
    email: bool = False
    id_last4: bool = False

    @property
    def matched_count(self) -> int:
        return sum([self.name, self.dob, self.phone, self.email, self.id_last4])


class TranscriptTurn(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phase: Optional[Phase] = None


class SessionCounters(BaseModel):
    failed_verification: int = 0


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    phase: Phase = Phase.VERIFY_ID
    verified: bool = False
    verified_party_id: Optional[str] = None
    collected_pii: CollectedPII = Field(default_factory=CollectedPII)
    match_counts: FieldMatchCounts = Field(default_factory=FieldMatchCounts)
    case_hint: Optional[CaseHint] = None
    selected_case_id: Optional[str] = None
    counters: SessionCounters = Field(default_factory=SessionCounters)
    consent_choice: ConsentChoice = ConsentChoice.UNSET
    email_status: Optional[str] = None
    email_status_index: int = 0
    email_scenario: str = "default"
    transcript: list[TranscriptTurn] = Field(default_factory=list)
    escalated: bool = False
    done: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def merge_pii(self, incoming: CollectedPII) -> None:
        data = self.collected_pii.model_dump()
        for key, value in incoming.model_dump().items():
            if value is not None and value != "":
                data[key] = value
        self.collected_pii = CollectedPII(**data)

    def append_turn(self, role: str, content: str) -> None:
        self.transcript.append(
            TranscriptTurn(role=role, content=content, phase=self.phase)
        )
        self.touch()
