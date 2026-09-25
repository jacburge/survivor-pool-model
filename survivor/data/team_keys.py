"""Normalize team names across data sources to one shared key: the abbreviation.

The Odds API returns full names ("Buffalo Bills"); schedule and pick-percentage
sources may use other spellings. Every source's ingestion step should map
through here before anything downstream joins on team.
"""

from __future__ import annotations

# Canonical key is the standard 2-3 letter NFL abbreviation.
FULL_NAME_TO_ABBREVIATION: dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

ABBREVIATION_TO_FULL_NAME: dict[str, str] = {v: k for k, v in FULL_NAME_TO_ABBREVIATION.items()}

assert len(FULL_NAME_TO_ABBREVIATION) == 32, "expected exactly 32 NFL teams"


def to_abbreviation(team_name: str) -> str:
    """Map any known full team name to its abbreviation; passes through if already an abbreviation."""
    if team_name in ABBREVIATION_TO_FULL_NAME:
        return team_name
    try:
        return FULL_NAME_TO_ABBREVIATION[team_name]
    except KeyError as exc:
        raise KeyError(f"unrecognized team name: {team_name!r}") from exc
