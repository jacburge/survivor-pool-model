"""Convert point spreads into win probabilities for future weeks.

The normal approximation is the default per the plan. It is off by a point
or two near small spreads because NFL margins cluster at 3 and 7 -- an
empirical spread-to-moneyline mapping fit to historical closing lines is
the flagged upgrade once historical data is ingested (Phase 3 technical
note), not implemented here yet.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

DEFAULT_MARGIN_STD = 13.5


def spread_to_win_prob(spread: float | np.ndarray, sigma: float = DEFAULT_MARGIN_STD) -> float | np.ndarray:
    """Win probability for a team given its own spread (negative = favorite).

    Assumes final margin ~ Normal(-spread, sigma^2), i.e. a team favored by
    s points is expected to win by s on average.
    """
    return norm.cdf(-np.asarray(spread, dtype=float) / sigma)


def rating_diff_to_spread(home_rating: float, away_rating: float, home_field_advantage: float) -> float:
    """Expected home margin from ratings: matches the fit target -s = r_home - r_away + h."""
    return -(home_rating - away_rating + home_field_advantage)
