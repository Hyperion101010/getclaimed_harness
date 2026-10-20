from __future__ import annotations

from apps.insurance_claims.fixtures_loader import get_fixtures
from apps.insurance_claims.state import CollectedPII
from apps.insurance_claims.verifier import (
    normalize_dob,
    normalize_phone,
    verify_identity,
)


def test_margaret_three_of_five_verifies():
    result = verify_identity(
        CollectedPII(
            name="Margaret Chen",
            policy_number="POL-9921",
            dob="1985-03-15",
            id_last4="4472",
        )
    )
    assert result.verified is True
    assert result.party_id == "P9"
    assert result.matched_count >= 3
    assert result.match_counts.name is True
    assert result.match_counts.dob is True
    assert result.match_counts.id_last4 is True


def test_two_of_five_does_not_verify():
    result = verify_identity(
        CollectedPII(
            name="Margaret Chen",
            policy_number="POL-9921",
            dob="1985-03-15",
        )
    )
    assert result.verified is False
    assert result.matched_count == 2
    assert result.reason == "insufficient_matches"


def test_ya_wen_li_name_alias_matches():
    result = verify_identity(
        CollectedPII(
            name="Yaven Li",
            policy_number="POL-7742",
            dob="1989-12-03",
            id_last4="5317",
        )
    )
    assert result.verified is True
    assert result.party_id == "P13"
    assert result.match_counts.name is True


def test_ma_tian_national_id_last4():
    result = verify_identity(
        CollectedPII(
            name="Ma Tian",
            policy_number="POL-8836",
            dob="1964-09-10",
            id_last4="6688",
        )
    )
    assert result.verified is True
    assert result.party_id == "P12"
    assert result.match_counts.id_last4 is True


def test_email_alias_and_phone_normalization():
    result = verify_identity(
        CollectedPII(
            name="Ya Wen Li",
            email="yawen.li@example.com",
            phone="650-521-2830",
            id_last4="5317",
        )
    )
    assert result.verified is True
    assert result.party_id == "P13"
    assert result.match_counts.email is True
    assert result.match_counts.phone is True


def test_wrong_policy_number_fails():
    result = verify_identity(
        CollectedPII(
            name="Margaret Chen",
            policy_number="POL-0000",
            dob="1985-03-15",
            id_last4="4472",
        )
    )
    assert result.verified is False
    assert result.reason == "policy_not_found"


def test_wrong_id_last4_blocks_third_match():
    result = verify_identity(
        CollectedPII(
            name="Margaret Chen",
            policy_number="POL-9921",
            dob="1985-03-15",
            id_last4="0000",
        )
    )
    assert result.verified is False
    assert result.match_counts.id_last4 is False
    assert result.matched_count == 2


def test_normalize_helpers():
    assert normalize_dob("03/15/1985") == "1985-03-15"
    assert normalize_phone("(650) 521-2836") == "+16505212836"
    fixtures = get_fixtures()
    assert fixtures.get_policyholder("P9") is not None
