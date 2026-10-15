import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from pick_rules import available_teams, validate_pick  # noqa: E402


def test_valid_pick_normalised():
    assert validate_pick(" kc ", []) == "KC"


def test_unknown_team_rejected():
    with pytest.raises(ValueError, match="not a recognised"):
        validate_pick("XYZ", [])


def test_used_team_rejected():
    with pytest.raises(ValueError, match="already used"):
        validate_pick("KC", ["KC", "BUF"])


def test_available_teams_excludes_used():
    teams = available_teams(["KC", "BUF"])
    assert len(teams) == 30 and "KC" not in teams and "BUF" not in teams
