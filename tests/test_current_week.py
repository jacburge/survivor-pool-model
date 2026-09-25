import pandas as pd
import pytest

from survivor.probability.current_week import (
    compute_current_week_probabilities,
    compute_current_week_spreads,
    compute_game_probabilities,
    to_team_survival_probabilities,
)

TWO_BOOK_GAME = pd.DataFrame(
    [
        {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
         "bookmaker": "draftkings", "market": "h2h", "team": "BUF", "price": 1.25},
        {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
         "bookmaker": "draftkings", "market": "h2h", "team": "NE", "price": 4.20},
        {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
         "bookmaker": "fanduel", "market": "h2h", "team": "BUF", "price": 1.27},
        {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
         "bookmaker": "fanduel", "market": "h2h", "team": "NE", "price": 4.00},
    ]
)

# fanduel is missing the away side entirely -- should be skipped, not crash.
GAME_WITH_INCOMPLETE_BOOK = pd.concat(
    [
        TWO_BOOK_GAME,
        pd.DataFrame(
            [
                {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
                 "bookmaker": "betrivers", "market": "h2h", "team": "BUF", "price": 1.26},
            ]
        ),
    ],
    ignore_index=True,
)

TWO_GAME_WEEK = pd.concat(
    [
        TWO_BOOK_GAME,
        pd.DataFrame(
            [
                {"game_id": "g2", "commence_time": "2026-10-04T17:00:00Z", "home_team": "KC", "away_team": "LV",
                 "bookmaker": "draftkings", "market": "h2h", "team": "KC", "price": 1.40},
                {"game_id": "g2", "commence_time": "2026-10-04T17:00:00Z", "home_team": "KC", "away_team": "LV",
                 "bookmaker": "draftkings", "market": "h2h", "team": "LV", "price": 3.10},
                # a spreads row that should be ignored by compute_current_week_probabilities
                {"game_id": "g2", "commence_time": "2026-10-04T17:00:00Z", "home_team": "KC", "away_team": "LV",
                 "bookmaker": "draftkings", "market": "spreads", "team": "KC", "price": 1.91},
            ]
        ),
    ],
    ignore_index=True,
)


def test_survival_probabilities_sum_to_one_with_tie():
    result = compute_game_probabilities(TWO_BOOK_GAME, tie_probability=0.003)
    total = result["home_survival_probability"] + result["away_survival_probability"] + result["tie_probability"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_favorite_has_higher_survival_probability():
    result = compute_game_probabilities(TWO_BOOK_GAME)
    assert result["home_survival_probability"] > result["away_survival_probability"]


def test_num_books_counts_only_complete_books():
    result = compute_game_probabilities(TWO_BOOK_GAME)
    assert result["num_books"] == 2


def test_incomplete_book_is_skipped_not_counted():
    result = compute_game_probabilities(GAME_WITH_INCOMPLETE_BOOK)
    assert result["num_books"] == 2  # betrivers (one-sided) excluded


def test_no_complete_book_raises():
    one_sided_only = TWO_BOOK_GAME[TWO_BOOK_GAME["team"] == "BUF"]
    with pytest.raises(ValueError):
        compute_game_probabilities(one_sided_only)


def test_shin_method_also_sums_to_one():
    result = compute_game_probabilities(TWO_BOOK_GAME, method="shin")
    total = result["home_survival_probability"] + result["away_survival_probability"] + result["tie_probability"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        compute_game_probabilities(TWO_BOOK_GAME, method="not_a_real_method")


def test_current_week_ignores_non_h2h_markets_and_covers_all_games():
    result = compute_current_week_probabilities(TWO_GAME_WEEK)
    assert len(result) == 2
    assert set(result["game_id"]) == {"g1", "g2"}


def test_current_week_all_rows_sum_to_one():
    result = compute_current_week_probabilities(TWO_GAME_WEEK, tie_probability=0.003)
    totals = (
        result["home_survival_probability"] + result["away_survival_probability"] + result["tie_probability"]
    )
    assert (totals.round(9) == 1.0).all()


def test_to_team_survival_probabilities_reshapes_by_team():
    current_week = compute_current_week_probabilities(TWO_GAME_WEEK)
    by_team = to_team_survival_probabilities(current_week)
    assert set(by_team.index) == {"BUF", "NE", "KC", "LV"}
    assert by_team["BUF"] == pytest.approx(current_week.loc[current_week.home_team == "BUF", "home_survival_probability"].iloc[0])


SPREADS_ODDS = pd.DataFrame(
    [
        {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
         "bookmaker": "draftkings", "market": "spreads", "team": "BUF", "price": 1.91, "point": -7.0},
        {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
         "bookmaker": "draftkings", "market": "spreads", "team": "NE", "price": 1.91, "point": 7.0},
        {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
         "bookmaker": "fanduel", "market": "spreads", "team": "BUF", "price": 1.95, "point": -7.5},
        {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
         "bookmaker": "fanduel", "market": "spreads", "team": "NE", "price": 1.87, "point": 7.5},
        # h2h rows for the same game should be ignored entirely by compute_current_week_spreads
        {"game_id": "g1", "commence_time": "2026-10-04T17:00:00Z", "home_team": "BUF", "away_team": "NE",
         "bookmaker": "draftkings", "market": "h2h", "team": "BUF", "price": 1.25},
    ]
)


def test_compute_current_week_spreads_averages_across_books():
    result = compute_current_week_spreads(SPREADS_ODDS)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["home_team"] == "BUF"
    assert row["home_spread"] == pytest.approx((-7.0 + -7.5) / 2)
    assert row["num_books"] == 2


def test_compute_current_week_spreads_ignores_h2h_rows():
    result = compute_current_week_spreads(SPREADS_ODDS)
    # only spreads rows should count -- if h2h leaked in, num_books or the
    # averaged point would be wrong (h2h rows have no point at all)
    assert not result["home_spread"].isna().any()
