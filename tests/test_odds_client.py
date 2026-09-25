import pytest

from survivor.data.odds_client import parse_odds_events

SAMPLE_EVENT = {
    "id": "abc123",
    "sport_key": "americanfootball_nfl",
    "commence_time": "2026-10-05T17:00:00Z",
    "home_team": "Buffalo Bills",
    "away_team": "Miami Dolphins",
    "bookmakers": [
        {
            "key": "draftkings",
            "title": "DraftKings",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": 1.50},
                        {"name": "Miami Dolphins", "price": 2.80},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": 1.91, "point": -7.5},
                        {"name": "Miami Dolphins", "price": 1.91, "point": 7.5},
                    ],
                },
            ],
        },
        {
            "key": "fanduel",
            "title": "FanDuel",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": 1.53},
                        {"name": "Miami Dolphins", "price": 2.70},
                    ],
                },
            ],
        },
    ],
}


def test_parse_produces_one_row_per_outcome():
    df = parse_odds_events([SAMPLE_EVENT])
    # draftkings: 2 h2h + 2 spreads, fanduel: 2 h2h = 6 rows
    assert len(df) == 6


def test_parse_normalizes_team_names_to_abbreviations():
    df = parse_odds_events([SAMPLE_EVENT])
    assert set(df["team"]) == {"BUF", "MIA"}
    assert (df["home_team"] == "BUF").all()
    assert (df["away_team"] == "MIA").all()


def test_parse_keeps_point_for_spreads_and_none_for_h2h():
    df = parse_odds_events([SAMPLE_EVENT])
    h2h = df[df["market"] == "h2h"]
    spreads = df[df["market"] == "spreads"]
    assert h2h["point"].isna().all()
    assert not spreads["point"].isna().any()


def test_parse_multiple_bookmakers_present():
    df = parse_odds_events([SAMPLE_EVENT])
    assert set(df["bookmaker"]) == {"draftkings", "fanduel"}


def test_parse_empty_events_returns_empty_frame_with_columns():
    df = parse_odds_events([])
    assert len(df) == 0
    assert "game_id" in df.columns
    assert "price" in df.columns


def test_parse_unrecognized_team_raises():
    bad_event = dict(SAMPLE_EVENT, home_team="Not A Real Team")
    with pytest.raises(KeyError):
        parse_odds_events([bad_event])
