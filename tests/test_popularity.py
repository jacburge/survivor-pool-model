import math

import numpy as np
import pytest

from survivor.decision.popularity import (
    cross_entropy,
    fit_beta_to_weeks,
    pick_probabilities,
    proportional_to_win_probability,
    weekly_cross_entropy,
)


def test_probabilities_sum_to_one():
    teams = ["BUF", "NE", "MIA"]
    win_prob = {"BUF": 0.85, "NE": 0.60, "MIA": 0.55}
    probs = pick_probabilities(teams, win_prob, future_value={}, beta=5.0, gamma=0.0)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_higher_win_probability_gets_more_popularity():
    teams = ["BUF", "NE"]
    win_prob = {"BUF": 0.85, "NE": 0.55}
    probs = pick_probabilities(teams, win_prob, future_value={}, beta=5.0, gamma=0.0)
    assert probs["BUF"] > probs["NE"]


def test_zero_beta_gamma_is_uniform():
    teams = ["BUF", "NE", "MIA"]
    win_prob = {"BUF": 0.85, "NE": 0.55, "MIA": 0.40}
    probs = pick_probabilities(teams, win_prob, future_value={}, beta=0.0, gamma=0.0)
    for p in probs.values():
        assert p == pytest.approx(1 / 3)


def test_future_value_shifts_popularity_away_from_favorite():
    teams = ["BUF", "NE"]
    win_prob = {"BUF": 0.85, "NE": 0.80}
    # NE has much higher future value (e.g. a team worth saving), BUF none
    future_value = {"BUF": 0.0, "NE": 5.0}
    probs_no_future = pick_probabilities(teams, win_prob, future_value={}, beta=5.0, gamma=0.0)
    probs_with_future = pick_probabilities(teams, win_prob, future_value, beta=5.0, gamma=1.0)
    assert probs_with_future["NE"] > probs_no_future["NE"]


def test_empty_available_teams_raises():
    with pytest.raises(ValueError):
        pick_probabilities([], {}, {}, beta=1.0, gamma=0.0)


def test_proportional_to_win_probability_sums_to_one_and_ranks_correctly():
    win_prob = {"BUF": 0.8, "NE": 0.4, "MIA": 0.2}
    probs = proportional_to_win_probability(win_prob)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["BUF"] > probs["NE"] > probs["MIA"]


def test_cross_entropy_of_distribution_with_itself_equals_its_entropy():
    dist = {"BUF": 0.7, "NE": 0.3}
    entropy = -sum(p * math.log(p) for p in dist.values())
    assert cross_entropy(dist, dist) == pytest.approx(entropy)


def test_cross_entropy_penalizes_confident_wrong_predictions():
    observed = {"BUF": 0.9, "NE": 0.1}
    good_prediction = {"BUF": 0.85, "NE": 0.15}
    bad_prediction = {"BUF": 0.1, "NE": 0.9}
    assert cross_entropy(good_prediction, observed) < cross_entropy(bad_prediction, observed)


def test_weekly_cross_entropy_renormalizes_pick_percentage():
    win_prob = {"BUF": 0.8, "NE": 0.5}
    # pick percentages that don't sum to exactly 1 (rounding, as real data has)
    pick_pct = {"BUF": 0.60, "NE": 0.38}
    loss = weekly_cross_entropy(win_prob, pick_pct, beta=5.0)
    assert loss > 0
    assert np.isfinite(loss)


def test_fit_beta_recovers_known_temperature():
    # Generate "observed" pick percentages from a known beta, then check the
    # fit recovers something close to it.
    true_beta = 8.0
    weeks = []
    for win_prob in [
        {"BUF": 0.8, "NE": 0.5, "MIA": 0.3},
        {"KC": 0.9, "LV": 0.4, "DEN": 0.2},
        {"SF": 0.7, "SEA": 0.6, "ARI": 0.35},
    ]:
        observed = pick_probabilities(list(win_prob), win_prob, {}, beta=true_beta, gamma=0.0)
        weeks.append({"win_probability": win_prob, "pick_percentage": observed})

    fitted_beta = fit_beta_to_weeks(weeks)
    assert fitted_beta == pytest.approx(true_beta, rel=0.05)


def test_fitted_beta_beats_baselines_on_held_out_week():
    true_beta = 6.0
    train_weeks = []
    for win_prob in [
        {"BUF": 0.8, "NE": 0.5, "MIA": 0.3},
        {"KC": 0.9, "LV": 0.4, "DEN": 0.2},
    ]:
        observed = pick_probabilities(list(win_prob), win_prob, {}, beta=true_beta, gamma=0.0)
        train_weeks.append({"win_probability": win_prob, "pick_percentage": observed})
    fitted_beta = fit_beta_to_weeks(train_weeks)

    held_out_win_prob = {"SF": 0.75, "SEA": 0.55, "ARI": 0.3}
    held_out_observed = pick_probabilities(list(held_out_win_prob), held_out_win_prob, {}, beta=true_beta, gamma=0.0)

    fitted_loss = weekly_cross_entropy(held_out_win_prob, held_out_observed, beta=fitted_beta)
    uniform_loss = weekly_cross_entropy(held_out_win_prob, held_out_observed, beta=0.0)
    proportional_loss = cross_entropy(proportional_to_win_probability(held_out_win_prob), held_out_observed)

    assert fitted_loss < uniform_loss
    assert fitted_loss < proportional_loss
