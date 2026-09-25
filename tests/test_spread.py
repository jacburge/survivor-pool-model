import pytest

from survivor.probability.spread import rating_diff_to_spread, spread_to_win_prob


def test_pick_em_is_fifty_fifty():
    assert spread_to_win_prob(0.0) == pytest.approx(0.5)


def test_favorite_has_higher_win_probability():
    # favored by 7 points
    p = spread_to_win_prob(-7.0)
    assert p > 0.5


def test_underdog_has_lower_win_probability():
    p = spread_to_win_prob(7.0)
    assert p < 0.5


def test_symmetric_around_pick_em():
    fav = spread_to_win_prob(-3.0)
    dog = spread_to_win_prob(3.0)
    assert fav + dog == pytest.approx(1.0)


def test_large_favorite_approaches_certainty():
    p = spread_to_win_prob(-21.0)
    assert p > 0.9


def test_rating_diff_to_spread_matches_fit_target():
    # r_home - r_away + h = -s  =>  s = -(r_home - r_away + h)
    s = rating_diff_to_spread(home_rating=5.0, away_rating=2.0, home_field_advantage=1.5)
    assert s == pytest.approx(-(5.0 - 2.0 + 1.5))
