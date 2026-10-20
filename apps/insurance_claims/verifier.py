from __future__ import annotations

from datetime import datetime
from typing import Optional

import phonenumbers
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from apps.insurance_claims.fixture_models import Policyholder
from apps.insurance_claims.fixtures_loader import FixtureBundle, get_fixtures
from apps.insurance_claims.state import CollectedPII, FieldMatchCounts, SessionState

NAME_MATCH_THRESHOLD = 88
MIN_MATCHING_FIELDS = 3


def _collapse_spaces(value: str) -> str:
    return " ".join(value.split())


def _only_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


_DOB_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
)


class VerificationResult(BaseModel):
    verified: bool = False
    party_id: Optional[str] = None
    policy_number: Optional[str] = None
    match_counts: FieldMatchCounts = Field(default_factory=FieldMatchCounts)
    matched_count: int = 0
    candidate_count: int = 0
    reason: str = ""


def normalize_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = _collapse_spaces(value.strip().casefold())
    return cleaned or None


def normalize_email(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().casefold()
    return cleaned or None


def normalize_dob(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    for fmt in _DOB_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    digits = _only_digits(raw)
    if len(digits) == 8:
        try:
            return datetime.strptime(digits, "%Y%m%d").date().isoformat()
        except ValueError:
            try:
                return datetime.strptime(digits, "%m%d%Y").date().isoformat()
            except ValueError:
                return None
    return None


def normalize_phone(value: Optional[str], default_region: str = "US") -> Optional[str]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = phonenumbers.parse(raw, default_region)
        if not phonenumbers.is_possible_number(parsed):
            return None
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        digits = _only_digits(raw)
        if len(digits) == 10:
            return f"+1{digits}"
        if len(digits) == 11 and digits.startswith("1"):
            return f"+{digits}"
        return None


def normalize_id_last4(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    digits = _only_digits(value.strip())
    if len(digits) < 4:
        return None
    return digits[-4:]


def normalize_policy_number(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().upper()
    return cleaned or None


def _name_matches(provided: Optional[str], policyholder: Policyholder) -> bool:
    left = normalize_name(provided)
    if left is None:
        return False
    candidates = [policyholder.name, *policyholder.name_aliases]
    for candidate in candidates:
        right = normalize_name(candidate)
        if right is None:
            continue
        if left == right:
            return True
        if fuzz.token_set_ratio(left, right) >= NAME_MATCH_THRESHOLD:
            return True
    return False


def _phone_matches(provided: Optional[str], policyholder: Policyholder) -> bool:
    left = normalize_phone(provided)
    if left is None:
        return False
    candidates = [policyholder.phone, *policyholder.phone_aliases]
    return any(normalize_phone(candidate) == left for candidate in candidates)


def _email_matches(provided: Optional[str], policyholder: Policyholder) -> bool:
    left = normalize_email(provided)
    if left is None:
        return False
    candidates = [policyholder.email, *policyholder.email_aliases]
    return any(normalize_email(candidate) == left for candidate in candidates)


def match_fields(pii: CollectedPII, policyholder: Policyholder) -> FieldMatchCounts:
    return FieldMatchCounts(
        name=_name_matches(pii.name, policyholder),
        dob=normalize_dob(pii.dob) == normalize_dob(policyholder.dob)
        if pii.dob
        else False,
        phone=_phone_matches(pii.phone, policyholder),
        email=_email_matches(pii.email, policyholder),
        id_last4=(
            normalize_id_last4(pii.id_last4) == normalize_id_last4(policyholder.id_last4)
            if pii.id_last4
            else False
        ),
    )


def verify_identity(
    pii: CollectedPII,
    fixtures: FixtureBundle | None = None,
    *,
    min_matches: int = MIN_MATCHING_FIELDS,
) -> VerificationResult:
    bundle = fixtures or get_fixtures()
    policy_number = normalize_policy_number(pii.policy_number)

    if policy_number:
        target = bundle.get_policyholder_by_policy(policy_number)
        if target is None:
            return VerificationResult(
                verified=False,
                reason="policy_not_found",
                candidate_count=0,
            )
        candidates = [target]
    else:
        candidates = list(bundle.policyholders)

    best: Policyholder | None = None
    best_matches = FieldMatchCounts()
    best_count = -1
    ties = 0

    for candidate in candidates:
        matches = match_fields(pii, candidate)
        count = matches.matched_count
        if count > best_count:
            best = candidate
            best_matches = matches
            best_count = count
            ties = 1
        elif count == best_count and count > 0:
            ties += 1

    if best is None or best_count < min_matches:
        return VerificationResult(
            verified=False,
            party_id=best.party_id if best and best_count > 0 else None,
            policy_number=best.policy_number if best and best_count > 0 else policy_number,
            match_counts=best_matches,
            matched_count=max(best_count, 0),
            candidate_count=len(candidates),
            reason="insufficient_matches",
        )

    if ties > 1 and not policy_number:
        return VerificationResult(
            verified=False,
            match_counts=best_matches,
            matched_count=best_count,
            candidate_count=len(candidates),
            reason="ambiguous_match",
        )

    return VerificationResult(
        verified=True,
        party_id=best.party_id,
        policy_number=best.policy_number,
        match_counts=best_matches,
        matched_count=best_count,
        candidate_count=len(candidates),
        reason="verified",
    )


def apply_verification(state: SessionState, result: VerificationResult) -> None:
    state.match_counts = result.match_counts
    if result.verified:
        state.verified = True
        state.verified_party_id = result.party_id
    else:
        state.verified = False
