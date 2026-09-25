"""Convert bookmaker moneylines into fair win probabilities.

Bookmaker odds embed a margin (the vig) so implied probabilities sum to
more than 1. Devigging removes it. The power method and Shin's method
agree closely on near-even games but diverge on heavy favorites -- exactly
where survivor picks cluster -- so both are provided for comparison.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def implied_probabilities(decimal_odds: np.ndarray) -> np.ndarray:
    """Raw bookmaker-implied probabilities (sum to more than 1)."""
    decimal_odds = np.asarray(decimal_odds, dtype=float)
    return 1.0 / decimal_odds


def power_devig(decimal_odds: np.ndarray, tol: float = 1e-12, max_iter: int = 200) -> tuple[np.ndarray, float]:
    """Power method: find exponent k with sum(q_i^k) = 1, p_i = q_i^k.

    Removes proportionally more margin from longshots than favorites,
    matching favorite-longshot bias. q_i are the raw implied probabilities.
    """
    q = implied_probabilities(decimal_odds)
    if np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("implied probabilities must lie strictly between 0 and 1")

    def total_at(k: float) -> float:
        return np.sum(q**k) - 1.0

    # sum(q^k) is strictly decreasing in k since 0 < q_i < 1.
    lo, hi = 0.1, 10.0
    while total_at(hi) > 0:
        hi *= 2
        if hi > 1e6:
            raise RuntimeError("power devig failed to converge on an exponent")
    k = brentq(total_at, lo, hi, xtol=tol, maxiter=max_iter)
    p = q**k
    return p / p.sum(), k


def shin_devig(decimal_odds: np.ndarray, tol: float = 1e-12, max_iter: int = 200) -> tuple[np.ndarray, float]:
    """Shin's method: model a fraction z of stake as informed, solve for z.

    p_i = (sqrt(z^2 + 4(1 - z) * q_i^2 / S) - z) / (2 * (1 - z))

    where q_i are raw implied probabilities and S = sum(q_i). Solved so
    that sum(p_i) = 1. z=0 has a removable singularity at p_i = q_i / S.
    """
    q = implied_probabilities(decimal_odds)
    if np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("implied probabilities must lie strictly between 0 and 1")
    s = q.sum()
    if s <= 1.0:
        # no margin to remove
        return q / s, 0.0

    def shin_probs(z: float) -> np.ndarray:
        if z <= 1e-10:
            return q / s
        return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / s) - z) / (2 * (1 - z))

    def total_at(z: float) -> float:
        return shin_probs(z).sum() - 1.0

    lo, hi = 0.0, 1.0 - 1e-9
    if total_at(lo) < 0:
        # already sums to <=1 at z=0 (can happen with tiny margins); fall back
        return q / s, 0.0
    z = brentq(total_at, lo, hi, xtol=tol, maxiter=max_iter)
    p = shin_probs(z)
    return p / p.sum(), z


def apply_tie_adjustment(win_probabilities: np.ndarray, tie_probability: float) -> np.ndarray:
    """Convert win probability to survival probability: p(win) * (1 - p(tie)).

    tie_probability is per game (roughly 0.002-0.005 for NFL), not per team.
    """
    win_probabilities = np.asarray(win_probabilities, dtype=float)
    return win_probabilities * (1.0 - tie_probability)


def consensus_probabilities(devigged_by_book: list[np.ndarray]) -> np.ndarray:
    """Average devigged probabilities across books, then renormalize to sum to 1."""
    stacked = np.vstack(devigged_by_book)
    mean = stacked.mean(axis=0)
    return mean / mean.sum()
