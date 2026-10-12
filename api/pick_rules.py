"""Rules for a saved pick. Kept free of FastAPI so it can be unit tested."""
from __future__ import annotations

from survivor.data.team_keys import ALL_TEAMS, to_abbreviation


def validate_pick(team: str, used_before: list[str]) -> str:
    """Return the canonical abbreviation for `team`, or raise ValueError if it isn't a team
    or this entry already used it in an earlier week (survivor: one use per team)."""
    try:
        abbreviation = to_abbreviation(team.strip().upper())
    except KeyError:
        raise ValueError(f"{team!r} is not a recognised NFL team") from None
    if abbreviation in used_before:
        raise ValueError(f"{abbreviation} was already used by this entry in an earlier week")
    return abbreviation


def available_teams(used_before: list[str]) -> list[str]:
    return [t for t in ALL_TEAMS if t not in used_before]
