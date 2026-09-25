"""Fetch the NFL schedule from ESPN's public scoreboard endpoint.

No API key, no relation to the odds API's quota, so this can be pulled
freely. It's an unofficial, undocumented endpoint (see ESPN API research),
so cache pulls and don't lean on it for anything more than schedule facts.

Each game locks at its own kickoff, but Splash's rule adds a hard Sunday
1:00 PM Eastern cutoff for every remaining game that week -- including
Sunday/Monday night games. effective_lock_time below is
min(kickoff, that week's Sunday 1pm ET cutoff).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from survivor.data.team_keys import FULL_NAME_TO_ABBREVIATION, to_abbreviation

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
EASTERN = ZoneInfo("America/New_York")
REGULAR_SEASON = 2


def fetch_week_schedule(week: int, year: int, season_type: int = REGULAR_SEASON) -> dict[str, Any]:
    """One live pull of a single week's schedule. Raises on any non-2xx response."""
    response = requests.get(
        SCOREBOARD_URL,
        params={"week": week, "seasontype": season_type, "year": year},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def parse_week_schedule(raw: dict[str, Any], week: int) -> pd.DataFrame:
    """Flatten one week's raw scoreboard JSON into a tidy frame.

    Joins on displayName (e.g. "Cleveland Browns") through team_keys, the
    same normalization the odds client uses, rather than ESPN's own
    abbreviations -- ESPN spells a few differently (e.g. WSH vs WAS).
    """
    rows = []
    for event in raw.get("events", []):
        competition = event["competitions"][0]
        home = next(c for c in competition["competitors"] if c["homeAway"] == "home")
        away = next(c for c in competition["competitors"] if c["homeAway"] == "away")
        rows.append(
            {
                "week": week,
                "game_id": event["id"],
                "commence_time": event["date"],
                "home_team": to_abbreviation(home["team"]["displayName"]),
                "away_team": to_abbreviation(away["team"]["displayName"]),
            }
        )
    return pd.DataFrame(rows, columns=["week", "game_id", "commence_time", "home_team", "away_team"])


def compute_bye_teams(week_games: pd.DataFrame) -> list[str]:
    """Teams absent from a week's schedule."""
    playing = set(week_games["home_team"]) | set(week_games["away_team"])
    all_teams = set(FULL_NAME_TO_ABBREVIATION.values())
    return sorted(all_teams - playing)


def _sunday_cutoff_utc(week_games: pd.DataFrame) -> pd.Timestamp | None:
    """The week's Sunday 1:00 PM Eastern cutoff, as a UTC timestamp."""
    commence = pd.to_datetime(week_games["commence_time"], utc=True)
    eastern_times = commence.dt.tz_convert(EASTERN)
    sunday_dates = eastern_times[eastern_times.dt.weekday == 6].dt.date
    if sunday_dates.empty:
        return None
    sunday_date = sorted(sunday_dates)[0]
    cutoff_local = datetime.combine(sunday_date, time(13, 0), tzinfo=EASTERN)
    return pd.Timestamp(cutoff_local).tz_convert(timezone.utc)


def add_effective_lock_times(week_games: pd.DataFrame) -> pd.DataFrame:
    """Add effective_lock_time = min(own kickoff, that week's Sunday 1pm ET cutoff).

    A game kicking off before the Sunday cutoff (e.g. an early London game)
    locks at its own earlier kickoff. Sunday/Monday night games are still
    bound by the earlier Sunday 1pm cutoff, per the pool's rules.
    """
    week_games = week_games.copy()
    commence = pd.to_datetime(week_games["commence_time"], utc=True)
    cutoff = _sunday_cutoff_utc(week_games)
    if cutoff is None:
        week_games["effective_lock_time"] = commence
    else:
        week_games["effective_lock_time"] = commence.clip(upper=cutoff)
    return week_games


def fetch_full_schedule(weeks: range, year: int) -> pd.DataFrame:
    """Pull and combine schedules for a range of weeks (e.g. range(4, 19))."""
    frames = []
    for week in weeks:
        raw = fetch_week_schedule(week, year)
        frames.append(parse_week_schedule(raw, week))
    combined = pd.concat(frames, ignore_index=True)
    return add_effective_lock_times(combined)
