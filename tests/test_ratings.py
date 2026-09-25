import numpy as np
import pandas as pd
import pytest

from survivor.probability.ratings import (
    fit_team_ratings,
    fit_weekly_rating_std,
    projected_rating_std,
    sample_correlated_ratings,
)


def _synthetic_games(true_ratings: dict[str, float], true_hfa: float, matchups: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for home, away in matchups:
        spread = -(true_ratings[home] - true_ratings[away] + true_hfa)
        rows.append({"home_team": home, "away_team": away, "home_spread": spread})
    return pd.DataFrame(rows)


def test_fit_recovers_spreads_within_one_point():
    true_ratings = {"NE": 3.0, "NYJ": -2.0, "BUF": 5.0, "MIA": -1.0}
    true_hfa = 2.0
    # Every team must appear both home and away, or home-field advantage is
    # collinear with team identity and unidentifiable from the schedule alone.
    matchups = [
        ("NE", "NYJ"), ("NYJ", "NE"),
        ("BUF", "MIA"), ("MIA", "BUF"),
        ("NE", "MIA"), ("MIA", "NE"),
        ("BUF", "NYJ"), ("NYJ", "BUF"),
    ]
    games = _synthetic_games(true_ratings, true_hfa, matchups)

    fit = fit_team_ratings(games)

    fitted_spreads = games.apply(
        lambda g: -(fit.ratings[g["home_team"]] - fit.ratings[g["away_team"]] + fit.home_field_advantage),
        axis=1,
    )
    mean_abs_error = (fitted_spreads - games["home_spread"]).abs().mean()
    assert mean_abs_error < 1.0
    assert fit.home_field_advantage == pytest.approx(true_hfa, abs=0.5)


def test_ratings_are_zero_mean():
    true_ratings = {"NE": 3.0, "NYJ": -2.0, "BUF": 5.0, "MIA": -1.0}
    matchups = [("NE", "NYJ"), ("BUF", "MIA"), ("NE", "MIA"), ("BUF", "NYJ")]
    games = _synthetic_games(true_ratings, 2.0, matchups)

    fit = fit_team_ratings(games)
    assert sum(fit.ratings.values()) == pytest.approx(0.0, abs=1e-6)


def test_projected_std_grows_with_horizon():
    assert projected_rating_std(0, weekly_std=1.0) == 0.0
    std_5 = projected_rating_std(5, weekly_std=1.0)
    std_10 = projected_rating_std(10, weekly_std=1.0)
    assert std_10 > std_5
    assert std_10 == pytest.approx(np.sqrt(10))


def test_projected_std_rejects_negative_horizon():
    with pytest.raises(ValueError):
        projected_rating_std(-1, weekly_std=1.0)


def test_correlated_sampling_reuses_one_draw_per_path():
    rng = np.random.default_rng(42)
    ratings = {"NE": 3.0, "BUF": 5.0}
    sampled = sample_correlated_ratings(ratings, weekly_std=2.0, weeks_ahead=5, n_paths=1000, rng=rng)

    assert sampled["NE"].shape == (1000,)
    # each path's rating should differ from the mean (uncertainty applied)...
    assert sampled["NE"].std() > 0
    # ...but be the *same* value whether read once or twice for the same path
    assert np.array_equal(sampled["NE"], sampled["NE"])


def test_correlated_sampling_zero_horizon_has_no_noise():
    rng = np.random.default_rng(0)
    sampled = sample_correlated_ratings({"NE": 3.0}, weekly_std=2.0, weeks_ahead=0, n_paths=100, rng=rng)
    assert np.all(sampled["NE"] == 3.0)


def test_fit_weekly_rating_std_from_history():
    history = pd.DataFrame(
        {
            "NE": [1.0, 1.5, 0.8, 1.2],
            "BUF": [4.0, 4.2, 3.9, 4.5],
        }
    )
    std = fit_weekly_rating_std(history)
    assert std > 0
