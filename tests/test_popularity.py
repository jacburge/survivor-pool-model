import pytest

from survivor.decision.popularity import pick_probabilities


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
