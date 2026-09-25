import pytest

from survivor.data.team_keys import (
    ABBREVIATION_TO_FULL_NAME,
    FULL_NAME_TO_ABBREVIATION,
    to_abbreviation,
)


def test_exactly_32_teams():
    assert len(FULL_NAME_TO_ABBREVIATION) == 32
    assert len(ABBREVIATION_TO_FULL_NAME) == 32


def test_full_name_maps_to_abbreviation():
    assert to_abbreviation("Buffalo Bills") == "BUF"
    assert to_abbreviation("Kansas City Chiefs") == "KC"


def test_abbreviation_passes_through():
    assert to_abbreviation("BUF") == "BUF"


def test_alias_abbreviation_maps_to_canonical():
    # SurvivorGrid uses WSH where the odds/schedule sources use WAS.
    assert to_abbreviation("WSH") == "WAS"


def test_unknown_team_raises():
    with pytest.raises(KeyError):
        to_abbreviation("Springfield Isotopes")


def test_mapping_is_bijective():
    abbreviations = list(FULL_NAME_TO_ABBREVIATION.values())
    assert len(abbreviations) == len(set(abbreviations))
