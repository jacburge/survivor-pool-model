"""Predict which teams rival entries pick, conditioned on availability.

Week 4 uses published public pick percentages directly. From Week 4
Thursday onward this softmax is the fallback for any rival whose pick
isn't directly observed by the rival tracker (Phase 8).
"""

from __future__ import annotations

import numpy as np

# Fit on 72 weeks of real SurvivorGrid pick data, 2022-2025
# (scripts/validate_popularity_model.py; historical pull via
# scripts/collect_historical_pick_data.py). gamma fixed at 0, matching the
# plan's own expectation that the public largely ignores future value --
# fitting only beta was already enough to clear Phase 4's bar: on 2025 held
# out entirely (trained on 2022-2024), mean log loss was 2.34 for the
# fitted softmax vs. 3.41 uniform and 2.99 proportional-to-win-probability.
DEFAULT_BETA = 10.1
DEFAULT_GAMMA = 0.0


def pick_probabilities(
    available_teams: list[str],
    win_probability: dict[str, float],
    future_value: dict[str, float],
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
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


def proportional_to_win_probability(win_probability: dict[str, float]) -> dict[str, float]:
    """Baseline: pick share proportional to win probability, no temperature at all."""
    total = sum(win_probability.values())
    return {team: p / total for team, p in win_probability.items()}


def cross_entropy(predicted: dict[str, float], observed_distribution: dict[str, float]) -> float:
    """-sum(observed * log(predicted)), i.e. average log loss if observed were per-person choice frequencies.

    observed_distribution should already sum to ~1 over the same team set as predicted.
    """
    return -sum(observed_distribution[team] * np.log(max(predicted[team], 1e-12)) for team in observed_distribution)


def weekly_cross_entropy(
    win_probability: dict[str, float],
    pick_percentage: dict[str, float],
    beta: float,
    gamma: float = 0.0,
    future_value: dict[str, float] | None = None,
) -> float:
    """Cross entropy between the softmax model and one week's observed pick percentages.

    Only real market/pick data goes in here -- aggregate percentages, not
    individual rival choices, since that's what's actually available from a
    public pick-percentage source (e.g. SurvivorGrid). pick_percentage is
    renormalized to sum to 1 first (published percentages round to ~99-101%).
    """
    teams = list(win_probability.keys())
    predicted = pick_probabilities(teams, win_probability, future_value or {}, beta, gamma)
    total = sum(pick_percentage.values())
    observed = {team: pick_percentage[team] / total for team in teams}
    return cross_entropy(predicted, observed)


def fit_beta_to_weeks(weeks: list[dict], beta_bounds: tuple[float, float] = (0.01, 50.0)) -> float:
    """Fit a single beta (gamma fixed at 0) minimizing mean cross entropy across many historical weeks.

    Each week is a dict with 'win_probability' and 'pick_percentage', both
    {team: value} for that week's available teams. gamma=0 matches the
    plan's own expectation that the public largely ignores future value.
    """
    from scipy.optimize import minimize_scalar

    def objective(beta: float) -> float:
        return float(np.mean([weekly_cross_entropy(w["win_probability"], w["pick_percentage"], beta) for w in weeks]))

    result = minimize_scalar(objective, bounds=beta_bounds, method="bounded")
    return float(result.x)
