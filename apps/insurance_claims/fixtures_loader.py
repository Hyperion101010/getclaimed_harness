from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from apps.insurance_claims.config import FIXTURES_DIR, get_settings
from apps.insurance_claims.fixture_models import (
    Claim,
    ClaimSchemaDoc,
    ConsentScenario,
    DocumentGuidelineBundle,
    Policyholder,
    Representative,
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class FixtureBundle:
    def __init__(
        self,
        policyholders: list[Policyholder],
        claims: list[Claim],
        representatives: list[Representative],
        consent_scenarios: dict[str, ConsentScenario],
        claim_schema: ClaimSchemaDoc,
        document_guidelines: DocumentGuidelineBundle,
    ) -> None:
        self.policyholders = policyholders
        self.claims = claims
        self.representatives = representatives
        self.consent_scenarios = consent_scenarios
        self.claim_schema = claim_schema
        self.document_guidelines = document_guidelines
        self._policyholders_by_id = {p.party_id: p for p in policyholders}
        self._policyholders_by_policy = {p.policy_number.upper(): p for p in policyholders}
        self._claims_by_id = {c.case_id: c for c in claims}

    def get_policyholder(self, party_id: str) -> Policyholder | None:
        return self._policyholders_by_id.get(party_id)

    def get_policyholder_by_policy(self, policy_number: str) -> Policyholder | None:
        return self._policyholders_by_policy.get(policy_number.upper())

    def get_claim(self, case_id: str) -> Claim | None:
        return self._claims_by_id.get(case_id)

    def claims_for_party(self, party_id: str) -> list[Claim]:
        return [c for c in self.claims if c.party_id == party_id]


def load_fixtures(fixtures_dir: Path | None = None) -> FixtureBundle:
    root = fixtures_dir or get_settings().fixtures_dir
    if not root.exists():
        root = FIXTURES_DIR

    policyholders = [
        Policyholder.model_validate(row)
        for row in _read_json(root / "policyholders.json")
    ]
    claims = [Claim.model_validate(row) for row in _read_json(root / "claims.json")]
    representatives = [
        Representative.model_validate(row)
        for row in _read_json(root / "representatives.json")
    ]
    consent_raw = _read_json(root / "consent_scenarios.json")
    consent_scenarios = {
        name: ConsentScenario.model_validate(payload)
        for name, payload in consent_raw.items()
    }
    claim_schema = ClaimSchemaDoc.model_validate(_read_json(root / "claim_schema.json"))
    document_guidelines = DocumentGuidelineBundle(
        data=_read_json(root / "required_document_guideline.json")
    )
    return FixtureBundle(
        policyholders=policyholders,
        claims=claims,
        representatives=representatives,
        consent_scenarios=consent_scenarios,
        claim_schema=claim_schema,
        document_guidelines=document_guidelines,
    )


@lru_cache
def get_fixtures() -> FixtureBundle:
    return load_fixtures()
