import numpy as np
import pytest

from survivor.probability.devig import (
    apply_tie_adjustment,
    consensus_probabilities,
    implied_probabilities,
    power_devig,
    shin_devig,
)


def test_implied_probabilities_sum_greater_than_one_with_vig():
    # -110 / -110 standard NFL two-way market: decimal odds 1.909 each side
    odds = np.array([1.9091, 1.9091])
    q = implied_probabilities(odds)
    assert q.sum() > 1.0
    assert q.sum() == pytest.approx(1.0476, abs=1e-3)


def test_power_devig_sums_to_one():
    odds = np.array([1.9091, 1.9091])
    p, k = power_devig(odds)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert p[0] == pytest.approx(0.5, abs=1e-9)  # symmetric market stays 50/50
    assert k > 0


def test_power_devig_heavy_favorite():
    # -400 favorite (decimal 1.25) vs +320 underdog (decimal 4.20)
    odds = np.array([1.25, 4.20])
    p, _ = power_devig(odds)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert p[0] > 0.75  # favorite still clearly favored after devig


def test_shin_devig_sums_to_one():
    odds = np.array([1.9091, 1.9091])
    p, z = shin_devig(odds)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert p[0] == pytest.approx(0.5, abs=1e-9)
    assert 0 <= z < 1


def test_shin_and_power_agree_closely_near_even_games():
    odds = np.array([1.95, 1.95])
    p_power, _ = power_devig(odds)
    p_shin, _ = shin_devig(odds)
    assert p_power[0] == pytest.approx(p_shin[0], abs=1e-6)


def test_shin_and_power_diverge_on_heavy_favorites():
    odds = np.array([1.10, 8.50])  # heavy favorite
    p_power, _ = power_devig(odds)
    p_shin, _ = shin_devig(odds)
    # both valid probabilities, but not identical on a heavy favorite
    assert p_power.sum() == pytest.approx(1.0, abs=1e-9)
    assert p_shin.sum() == pytest.approx(1.0, abs=1e-9)
    assert abs(p_power[0] - p_shin[0]) > 1e-4


def test_tie_adjustment_reduces_win_probability():
    win_probs = np.array([0.75, 0.25])
    adjusted = apply_tie_adjustment(win_probs, tie_probability=0.003)
    assert np.all(adjusted < win_probs)
    assert adjusted[0] == pytest.approx(0.75 * 0.997)


def test_full_game_probabilities_sum_to_one_including_tie():
    odds = np.array([1.9091, 1.9091])
    p, _ = power_devig(odds)
    tie_prob = 0.003
    survival = apply_tie_adjustment(p, tie_prob)
    assert survival.sum() + tie_prob == pytest.approx(1.0, abs=1e-9)


def test_consensus_across_books_renormalizes():
    book_a = np.array([0.55, 0.45])
    book_b = np.array([0.53, 0.47])
    consensus = consensus_probabilities([book_a, book_b])
    assert consensus.sum() == pytest.approx(1.0, abs=1e-9)
    assert consensus[0] == pytest.approx(0.54, abs=1e-9)


def test_power_devig_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        power_devig(np.array([1.0, 2.0]))  # odds of 1.0 implies probability of 1
