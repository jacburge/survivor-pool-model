import numpy as np
import pandas as pd
import pytest

from survivor.probability.ratings import (
    DEFAULT_RIDGE,
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


def test_fit_weekly_rating_std_does_not_diff_across_season_boundary():
    # A huge jump between the last week of one season and the first week of
    # the next must not be treated as a "one week" change.
    season_a = pd.DataFrame({"NE": [1.0, 1.1, 0.9]}, index=[16, 17, 18])
    season_b = pd.DataFrame({"NE": [50.0, 50.1, 49.9]}, index=[1, 2, 3])  # unrelated roster/rating scale

    combined_std = fit_weekly_rating_std([season_a, season_b])
    within_season_only = fit_weekly_rating_std(season_a)
    # the huge cross-season jump (~49 points) must not leak into the pooled
    # estimate -- it should look like within-season noise, not blow up
    assert combined_std < 5 * within_season_only


def test_fit_weekly_rating_std_pools_multiple_seasons():
    season_a = pd.DataFrame({"NE": [1.0, 2.0, 3.0]}, index=[1, 2, 3])
    season_b = pd.DataFrame({"NE": [1.0, 2.0, 3.0]}, index=[1, 2, 3])
    pooled = fit_weekly_rating_std([season_a, season_b])
    single = fit_weekly_rating_std(season_a)
    # identical seasons pooled should give the same std as either one alone
    assert pooled == pytest.approx(single)


def test_default_ridge_is_zero():
    # Regression guard for the finding in ratings.py: ridge only ever hurt
    # accuracy on real data, since minimum-norm already handles disconnected
    # components on its own.
    assert DEFAULT_RIDGE == 0.0


def test_minimum_norm_centers_disconnected_components_without_ridge():
    # Two disconnected components: an isolated pair (A, B), and a connected
    # three-team chain (C-D-E). Nothing ties one component's level to the
    # other's, yet minimum-norm least squares should still center each
    # component's own mean rating at 0, unaided by ridge.
    games = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_spread": -20.0},
            {"home_team": "C", "away_team": "D", "home_spread": -3.0},
            {"home_team": "D", "away_team": "E", "home_spread": 2.0},
        ]
    )
    fit = fit_team_ratings(games, ridge=0.0)
    assert (fit.ratings["A"] + fit.ratings["B"]) / 2 == pytest.approx(0.0, abs=1e-9)
    assert (fit.ratings["C"] + fit.ratings["D"] + fit.ratings["E"]) / 3 == pytest.approx(0.0, abs=1e-9)


def test_ridge_shrinks_ratings_toward_zero():
    games = pd.DataFrame([{"home_team": "A", "away_team": "B", "home_spread": -14.0}])
    unregularized = fit_team_ratings(games, ridge=0.0)
    regularized = fit_team_ratings(games, ridge=1.0)
    assert abs(regularized.ratings["A"]) < abs(unregularized.ratings["A"])


def test_ridge_trades_off_in_sample_fit_quality():
    true_ratings = {"NE": 3.0, "NYJ": -2.0, "BUF": 5.0, "MIA": -1.0}
    matchups = [
        ("NE", "NYJ"), ("NYJ", "NE"),
        ("BUF", "MIA"), ("MIA", "BUF"),
        ("NE", "MIA"), ("MIA", "NE"),
        ("BUF", "NYJ"), ("NYJ", "BUF"),
    ]
    games = _synthetic_games(true_ratings, 2.0, matchups)

    no_ridge_mae = fit_team_ratings(games, ridge=0.0).residuals.abs().mean()
    with_ridge_mae = fit_team_ratings(games, ridge=1.0).residuals.abs().mean()
    assert with_ridge_mae > no_ridge_mae


def test_uniform_weights_match_unweighted_fit():
    true_ratings = {"NE": 3.0, "NYJ": -2.0, "BUF": 5.0, "MIA": -1.0}
    matchups = [
        ("NE", "NYJ"), ("NYJ", "NE"),
        ("BUF", "MIA"), ("MIA", "BUF"),
        ("NE", "MIA"), ("MIA", "NE"),
        ("BUF", "NYJ"), ("NYJ", "BUF"),
    ]
    games = _synthetic_games(true_ratings, 2.0, matchups)

    unweighted = fit_team_ratings(games)
    uniformly_weighted = fit_team_ratings(games, weights=np.ones(len(games)))
    for team in true_ratings:
        assert uniformly_weighted.ratings[team] == pytest.approx(unweighted.ratings[team])
    assert uniformly_weighted.home_field_advantage == pytest.approx(unweighted.home_field_advantage)


def test_weights_rejects_wrong_length():
    games = _synthetic_games({"NE": 1.0, "BUF": -1.0}, 0.0, [("NE", "BUF"), ("BUF", "NE")])
    with pytest.raises(ValueError):
        fit_team_ratings(games, weights=[1.0, 2.0, 3.0])  # 3 weights, 2 games


def test_higher_weight_pulls_the_fit_toward_that_games_own_spread():
    # A simple cycle (A-B, B-C, C-A) is always fit exactly regardless of
    # weights -- the shared home-field-advantage term absorbs any
    # "inconsistency" around a cycle (verified separately: residuals are
    # ~0 for a bare 3-cycle no matter the target spreads). Adding a 4th,
    # redundant A-vs-C game with a target that conflicts with what the
    # first three already imply creates genuine overdetermination: no
    # rating/HFA choice can satisfy all four games at once. Weighting the
    # A-vs-B game heavily should then pull the fitted A-vs-B spread much
    # closer to its own -3.0 than an equal-weight fit would.
    games = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_spread": -3.0},
            {"home_team": "B", "away_team": "C", "home_spread": -2.0},
            {"home_team": "C", "away_team": "A", "home_spread": -2.0},
            {"home_team": "A", "away_team": "C", "home_spread": -10.0},
        ]
    )

    def implied_ab_spread(fit):
        return -(fit.ratings["A"] - fit.ratings["B"] + fit.home_field_advantage)

    unweighted = fit_team_ratings(games)
    heavily_weighted = fit_team_ratings(games, weights=[100.0, 1.0, 1.0, 1.0])

    assert abs(implied_ab_spread(heavily_weighted) - (-3.0)) < abs(implied_ab_spread(unweighted) - (-3.0))
