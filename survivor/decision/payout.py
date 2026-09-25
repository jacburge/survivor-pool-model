"""Payout objective: expected share of the pot, not survival probability.

V_i = E[ pot * 1{i in W} / |W| ], where W is the winner set on a simulated
path (last survivors, an entire final week's field eliminated together, or
Week 18 survivors). Portfolio value sums V over entries sharing a path.
"""

from __future__ import annotations

from collections.abc import Iterable


def split_payout(pot: float, winners: Iterable[str]) -> dict[str, float]:
    """Equal split of the pot among winners on one simulated path."""
    winners = list(winners)
    if not winners:
        raise ValueError("winners must be non-empty")
    share = pot / len(winners)
    return {entry: share for entry in winners}


def expected_payout(pot: float, path_winners: list[set[str]], entry: str) -> float:
    """Monte Carlo estimate of V_i for one entry across simulated paths.

    path_winners: one winner set per simulated path. Paths where the entry
    doesn't appear in the winner set contribute zero.
    """
    if not path_winners:
        raise ValueError("no simulated paths provided")
    total = 0.0
    for winners in path_winners:
        if entry in winners:
            total += pot / len(winners)
    return total / len(path_winners)


def portfolio_expected_payout(pot: float, path_winners: list[set[str]], entries: Iterable[str]) -> float:
    """Sum of expected payout across a set of entries sharing the same paths.

    Entries picking the same team on the same path both land in that path's
    winner set (or both get eliminated together), which is why joint
    allocation across entries matters -- see Phase 6.
    """
    return sum(expected_payout(pot, path_winners, entry) for entry in entries)
