"""Predict which teams rival entries pick, conditioned on availability.

Week 4 uses published public pick percentages directly. From Week 4
Thursday onward this softmax is the fallback for any rival whose pick
isn't directly observed by the rival tracker (Phase 8).
"""

from __future__ import annotations

import numpy as np


def pick_probabilities(
    available_teams: list[str],
    win_probability: dict[str, float],
    future_value: dict[str, float],
    beta: float,
    gamma: float,
) -> dict[str, float]:
    """Softmax over available teams: P(j) = exp(beta*p_j + gamma*f_j) / sum(...).

    future_value defaults to 0 for any team missing from the dict, matching
    a gamma near zero meaning the public ignores future value.
    """
    if not available_teams:
        raise ValueError("no available teams to pick from")

    scores = np.array(
        [beta * win_probability[t] + gamma * future_value.get(t, 0.0) for t in available_teams]
    )
    scores -= scores.max()  # numerical stability, does not change softmax output
    weights = np.exp(scores)
    probabilities = weights / weights.sum()
    return dict(zip(available_teams, probabilities))


def fit_softmax_temperature(
    observed_picks: list[str],
    available_teams_per_pick: list[list[str]],
    win_probability_per_pick: list[dict[str, float]],
    future_value_per_pick: list[dict[str, float]],
    beta_init: float = 5.0,
    gamma_init: float = 0.0,
) -> tuple[float, float]:
    """Fit beta, gamma by maximum likelihood on historical public picks.

    Each of the four lists must be the same length and aligned by index:
    one entry per historical (rival, week) pick observation.
    """
    from scipy.optimize import minimize

    def neg_log_likelihood(params: np.ndarray) -> float:
        beta, gamma = params
        total = 0.0
        for pick, teams, win_prob, future_val in zip(
            observed_picks, available_teams_per_pick, win_probability_per_pick, future_value_per_pick
        ):
            probs = pick_probabilities(teams, win_prob, future_val, beta, gamma)
            total -= np.log(max(probs[pick], 1e-12))
        return total

    result = minimize(neg_log_likelihood, x0=[beta_init, gamma_init], method="Nelder-Mead")
    beta, gamma = result.x
    return float(beta), float(gamma)
