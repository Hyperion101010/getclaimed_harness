from __future__ import annotations

import pytest

from apps.insurance_claims.config import get_settings
from apps.insurance_claims.fixtures_loader import get_fixtures, load_fixtures
from apps.insurance_claims.state import CallerIntent, Phase, SessionState
from apps.insurance_claims.store import InMemorySessionStore, JsonFileSessionStore


def test_fixtures_load_and_validate():
    get_fixtures.cache_clear()
    bundle = load_fixtures()

    assert len(bundle.policyholders) == 4
    assert len(bundle.claims) == 5
    assert len(bundle.representatives) == 1
    assert "default" in bundle.consent_scenarios
    assert "timeout" in bundle.consent_scenarios
    assert "expected_reimbursement_amount" in bundle.claim_schema.field_descriptions
    assert "default_guidance" in bundle.document_guidelines.data

    margaret = bundle.get_policyholder_by_policy("POL-9921")
    assert margaret is not None
    assert margaret.name == "Margaret Chen"
    assert margaret.party_id == "P9"

    claim = bundle.get_claim("CL-2048")
    assert claim is not None
    assert claim.status == "denied"
    assert claim.party_id == "P9"

    party_claims = bundle.claims_for_party("P9")
    assert len(party_claims) == 4


def test_session_state_defaults_and_merge_pii():
    state = SessionState()
    assert state.phase == Phase.VERIFY_ID
    assert state.verified is False
    assert state.done is False

    state.merge_pii(
        state.collected_pii.model_copy(update={"name": "Margaret Chen", "dob": "1985-03-15"})
    )
    state.merge_pii(state.collected_pii.model_copy(update={"id_last4": "4472"}))
    assert state.collected_pii.name == "Margaret Chen"
    assert state.collected_pii.dob == "1985-03-15"
    assert state.collected_pii.id_last4 == "4472"

    state.append_turn("user", "hello")
    assert len(state.transcript) == 1
    assert state.transcript[0].phase == Phase.VERIFY_ID


def test_in_memory_session_store():
    store = InMemorySessionStore()
    created = store.create()
    loaded = store.get(created.session_id)
    assert loaded is not None
    assert loaded.session_id == created.session_id

    loaded.verified = True
    store.save(loaded)
    again = store.get(created.session_id)
    assert again is not None
    assert again.verified is True


def test_json_file_session_store(tmp_path):
    store = JsonFileSessionStore(directory=tmp_path)
    created = store.create()
    loaded = store.get(created.session_id)
    assert loaded is not None
    assert (tmp_path / f"{created.session_id}.json").exists()

    loaded.intent = CallerIntent.DENIAL_QUESTION
    store.save(loaded)
    again = store.get(created.session_id)
    assert again is not None
    assert again.intent == "denial_question"

    store.delete(created.session_id)
    assert store.get(created.session_id) is None


def test_settings_defaults():
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.openrouter_model
    assert settings.openrouter_base_url.endswith("/api/v1")
    assert settings.fixtures_dir.exists()


@pytest.mark.skipif(
    not get_settings().openrouter_api_key,
    reason="OPENROUTER_API_KEY not set; skipping live OpenRouter smoke call",
)
def test_openrouter_smoke_call():
    from apps.insurance_claims.llm import build_chat_model

    get_settings.cache_clear()
    llm = build_chat_model(temperature=0)
    result = llm.invoke("Reply with exactly: pong")
    assert "pong" in result.content.lower()
