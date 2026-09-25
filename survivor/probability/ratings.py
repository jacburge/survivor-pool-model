"""Fit market-implied team ratings and project them forward with uncertainty.

Ratings are identified only up to an additive constant, so the fit pins
the mean rating to zero. Uncertainty grows with weeks-ahead as a random
walk, and -- per the plan's most important modeling rule -- each Monte
Carlo path must sample one true rating per team and reuse it for every
remaining week, not resample per game. sample_correlated_ratings below is
what enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RatingFit:
    ratings: dict[str, float]
    home_field_advantage: float
    residuals: pd.Series  # fitted spread minus observed spread, per game


# 0.0 by default. ridge exists as an escape hatch, not a default need: numpy's
# minimum-norm lstsq solution already centers each disconnected schedule-graph
# component's mean rating at exactly 0 on its own (verified on real Week 3,
# 2026 data with a 6-team/24-team split -- the cross-component gap is ~0.000
# at every ridge level tested, including 0). Adding ridge on top of that only
# shrank well-determined within-component ratings, worsening both in-sample
# fit and Week 4 holdout accuracy at every positive value tried
# (scripts/validate_ratings.py). Keep ridge=0 until there's a component so
# sparsely connected (few games relative to teams) that variance reduction
# actually earns its keep, or an external prior (e.g. preseason power
# ratings) to regularize toward instead of a flat 0.
DEFAULT_RIDGE = 0.0


def fit_team_ratings(games: pd.DataFrame, ridge: float = 0.0) -> RatingFit:
    """Least squares fit of one rating per team plus home-field advantage.

    games must have columns: home_team, away_team, home_spread (bookmaker
    spread for the home team; negative means the home team is favored).
    Fits -home_spread ~= r_home - r_away + h.

    A single week's games is a perfect matching -- 32 teams paired into 16
    disjoint games with no edges between pairs -- so within a pair, only
    the rating *difference* is pinned by that game; nothing ties one pair's
    level to another's. Even two weeks combined can leave the schedule
    graph split into disconnected components (real Week 3+4, 2026 data
    does: a 6-team cluster and a 24-team cluster). This turns out to be
    less of a problem than it looks: numpy's minimum-norm solution already
    centers each disconnected component's mean rating at exactly 0 on its
    own (verified on the real 6/24 split -- the cross-component gap is
    ~0.000), which is the same "assume unknown teams are average" prior
    ridge would otherwise add. ridge > 0 additionally shrinks every team's
    rating toward 0, trading fit quality for variance reduction; on real
    data this only ever made both in-sample fit and Week 4 holdout accuracy
    worse (see DEFAULT_RIDGE), since a single-book-consensus spread is
    already a fairly reliable data point, not a noisy one. Keep ridge=0
    unless a component is so sparsely connected that shrinkage's variance
    reduction is worth its bias, or there's an external prior to shrink
    toward instead of a flat 0.
    """
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    team_index = {team: i for i, team in enumerate(teams)}
    n_games = len(games)
    n_teams = len(teams)

    # Columns: one per team (+1 home, -1 away), plus one for home-field advantage.
    design = np.zeros((n_games + 1, n_teams + 1))
    target = np.zeros(n_games + 1)

    for row, (_, game) in enumerate(games.iterrows()):
        design[row, team_index[game["home_team"]]] = 1.0
        design[row, team_index[game["away_team"]]] = -1.0
        design[row, -1] = 1.0
        target[row] = -game["home_spread"]

    # Extra row pins sum(ratings) = 0 to resolve the additive degeneracy.
    design[n_games, :n_teams] = 1.0
    target[n_games] = 0.0

    if ridge > 0:
        ridge_rows = np.zeros((n_teams, n_teams + 1))
        ridge_rows[:, :n_teams] = np.eye(n_teams) * np.sqrt(ridge)
        design = np.vstack([design, ridge_rows])
        target = np.concatenate([target, np.zeros(n_teams)])

    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    ratings = {team: solution[team_index[team]] for team in teams}
    home_field_advantage = solution[-1]

    fitted = games.apply(
        lambda g: -(ratings[g["home_team"]] - ratings[g["away_team"]] + home_field_advantage),
        axis=1,
    )
    residuals = fitted - games["home_spread"]

    return RatingFit(ratings=ratings, home_field_advantage=home_field_advantage, residuals=residuals)


def projected_rating_std(weeks_ahead: int, weekly_std: float) -> float:
    """Random-walk uncertainty: variance grows linearly with weeks ahead."""
    if weeks_ahead < 0:
        raise ValueError("weeks_ahead must be non-negative")
    return weekly_std * np.sqrt(weeks_ahead)


def sample_correlated_ratings(
    ratings: dict[str, float],
    weekly_std: float,
    weeks_ahead: dict[str, int] | int,
    n_paths: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample each team's true rating once per Monte Carlo path.

    weeks_ahead can be a single int (same horizon for all teams) or a dict
    per team, e.g. when projecting to different future weeks. Returns
    {team: array of shape (n_paths,)}. Reuse the same array for every week
    that path simulates for that team -- do not resample per game.
    """
    result: dict[str, np.ndarray] = {}
    for team, rating in ratings.items():
        horizon = weeks_ahead if isinstance(weeks_ahead, int) else weeks_ahead[team]
        std = projected_rating_std(horizon, weekly_std)
        noise = rng.standard_normal(n_paths) * std if std > 0 else np.zeros(n_paths)
        result[team] = rating + noise
    return result


def fit_weekly_rating_std(rating_history: pd.DataFrame) -> float:
    """Fit weekly random-walk std from historical week-over-week rating changes.

    rating_history: wide DataFrame indexed by week, one column per team,
    containing that team's fitted rating each week (NaN for bye weeks).
    """
    weekly_changes = rating_history.diff().stack()
    return float(weekly_changes.std())
