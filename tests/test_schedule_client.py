import pandas as pd
import pytest

from survivor.data.schedule_client import (
    add_effective_lock_times,
    compute_bye_teams,
    parse_week_schedule,
)


def _competitor(display_name: str, home_away: str) -> dict:
    return {"homeAway": home_away, "team": {"displayName": display_name}}


def _event(event_id: str, date: str, home: str, away: str) -> dict:
    return {
        "id": event_id,
        "date": date,
        "competitions": [
            {
                "competitors": [
                    _competitor(home, "home"),
                    _competitor(away, "away"),
                ]
            }
        ],
    }


# Mirrors the real Week 4, 2026 slate observed from a live pull: Thursday,
# an early Sunday game, the main Sunday slate, and a Monday night game.
SAMPLE_WEEK_4_RAW = {
    "events": [
        _event("1", "2026-10-02T00:15Z", "Cleveland Browns", "Pittsburgh Steelers"),  # Thu 8:15pm ET
        _event("2", "2026-10-04T13:30Z", "Washington Commanders", "Indianapolis Colts"),  # Sun 9:30am ET
        _event("3", "2026-10-04T17:00Z", "Buffalo Bills", "New England Patriots"),  # Sun 1:00pm ET
        _event("4", "2026-10-06T00:15Z", "New Orleans Saints", "Atlanta Falcons"),  # Mon 8:15pm ET
    ]
}


def test_parse_week_schedule_normalizes_teams_and_tags_week():
    df = parse_week_schedule(SAMPLE_WEEK_4_RAW, week=4)
    assert len(df) == 4
    assert (df["week"] == 4).all()
    assert set(df["home_team"]) == {"CLE", "WAS", "BUF", "NO"}
    assert set(df["away_team"]) == {"PIT", "IND", "NE", "ATL"}


def test_bye_teams_are_teams_absent_from_the_week():
    games = pd.DataFrame(
        {
            "home_team": ["BUF", "KC"],
            "away_team": ["NE", "LV"],
        }
    )
    byes = compute_bye_teams(games)
    assert "BUF" not in byes
    assert "NE" not in byes
    assert "KC" not in byes
    assert "LV" not in byes
    assert "DAL" in byes  # not scheduled this week
    assert len(byes) == 32 - 4


def test_early_sunday_game_locks_at_its_own_earlier_kickoff():
    df = parse_week_schedule(SAMPLE_WEEK_4_RAW, week=4)
    df = add_effective_lock_times(df)
    early_game = df[df["game_id"] == "2"].iloc[0]
    assert early_game["effective_lock_time"] == pd.Timestamp("2026-10-04T13:30:00Z")


def test_thursday_game_locks_at_its_own_kickoff():
    df = parse_week_schedule(SAMPLE_WEEK_4_RAW, week=4)
    df = add_effective_lock_times(df)
    thursday_game = df[df["game_id"] == "1"].iloc[0]
    assert thursday_game["effective_lock_time"] == pd.Timestamp("2026-10-02T00:15:00Z")


def test_monday_night_game_is_capped_at_sunday_cutoff_not_its_own_kickoff():
    df = parse_week_schedule(SAMPLE_WEEK_4_RAW, week=4)
    df = add_effective_lock_times(df)
    monday_game = df[df["game_id"] == "4"].iloc[0]

    expected_cutoff = pd.Timestamp("2026-10-04 13:00", tz="America/New_York").tz_convert("UTC")
    assert monday_game["effective_lock_time"] == expected_cutoff
    assert monday_game["effective_lock_time"] < pd.Timestamp(monday_game["commence_time"])


def test_main_sunday_slate_locks_at_the_sunday_cutoff():
    df = parse_week_schedule(SAMPLE_WEEK_4_RAW, week=4)
    df = add_effective_lock_times(df)
    main_slate_game = df[df["game_id"] == "3"].iloc[0]
    expected_cutoff = pd.Timestamp("2026-10-04 13:00", tz="America/New_York").tz_convert("UTC")
    assert main_slate_game["effective_lock_time"] == expected_cutoff


def test_week_with_no_sunday_game_falls_back_to_each_games_own_kickoff():
    thursday_and_monday_only = {
        "events": [
            SAMPLE_WEEK_4_RAW["events"][0],
            SAMPLE_WEEK_4_RAW["events"][3],
        ]
    }
    df = parse_week_schedule(thursday_and_monday_only, week=4)
    df = add_effective_lock_times(df)
    assert (df["effective_lock_time"] == pd.to_datetime(df["commence_time"], utc=True)).all()
