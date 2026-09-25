import pytest

from survivor.decision.payout import expected_payout, portfolio_expected_payout, split_payout


def test_split_payout_divides_pot_equally():
    payouts = split_payout(9000.0, ["entry_1", "entry_2", "entry_3"])
    assert payouts["entry_1"] == pytest.approx(3000.0)
    assert sum(payouts.values()) == pytest.approx(9000.0)


def test_split_payout_single_winner_takes_all():
    payouts = split_payout(9000.0, ["entry_1"])
    assert payouts["entry_1"] == pytest.approx(9000.0)


def test_split_payout_empty_winners_raises():
    with pytest.raises(ValueError):
        split_payout(9000.0, [])


def test_expected_payout_averages_across_paths():
    # entry_1 wins alone on path 1, splits 3 ways on path 2, loses on path 3
    paths = [{"entry_1"}, {"entry_1", "entry_2", "entry_3"}, {"entry_2"}]
    value = expected_payout(9000.0, paths, "entry_1")
    expected = (9000.0 + 3000.0 + 0.0) / 3
    assert value == pytest.approx(expected)


def test_expected_payout_zero_when_never_wins():
    paths = [{"entry_2"}, {"entry_3"}]
    assert expected_payout(9000.0, paths, "entry_1") == 0.0


def test_portfolio_sums_across_entries_on_shared_paths():
    # entries 1 and 2 both picked the same team and share every path's outcome
    paths = [{"entry_1", "entry_2"}, {"entry_1", "entry_2"}, set()]
    total = portfolio_expected_payout(9000.0, paths, ["entry_1", "entry_2"])
    per_entry = expected_payout(9000.0, paths, "entry_1")
    assert total == pytest.approx(2 * per_entry)
