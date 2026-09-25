"""Wire the devig math to real moneyline odds: one win/survival probability per game.

Takes the tidy odds frame from survivor.data.odds_client.parse_odds_events
(market == "h2h"), devigs each book independently, takes the consensus
across books, then applies the tie adjustment. This is Phase 2's output:
the single most important input to each week's decision.

Default method is Shin, not power, based on validating both against
SurvivorGrid's published consensus on real Week 3, 2026 odds
(scripts/validate_current_week.py): Shin matched within 0.5 percentage
points on every team (mean 0.21), while power diverged by up to 2.76
points on heavy favorites (mean 1.07) -- power's extra margin-shifting
toward favorites overshoots what the market consensus actually reflects.
Exactly the divergence the plan flagged as worth comparing; pass
method="power" to use it anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from survivor.probability.devig import (
    apply_tie_adjustment,
    consensus_probabilities,
    power_devig,
    shin_devig,
)

DEFAULT_TIE_PROBABILITY = 0.003
DEFAULT_DEVIG_METHOD = "shin"

_DEVIG_METHODS = {"power": power_devig, "shin": shin_devig}


def compute_game_probabilities(
    game_odds: pd.DataFrame,
    method: str = DEFAULT_DEVIG_METHOD,
    tie_probability: float = DEFAULT_TIE_PROBABILITY,
) -> dict:
    """Consensus win/survival probability for one game from its h2h rows across books.

    game_odds must be pre-filtered to market == "h2h" and a single game_id.
    Books missing a price for either side are skipped (can't devig a
    one-sided market); at least one complete book is required.
    """
    if method not in _DEVIG_METHODS:
        raise ValueError(f"unknown devig method {method!r}, expected one of {list(_DEVIG_METHODS)}")
    devig = _DEVIG_METHODS[method]

    home_team = game_odds["home_team"].iloc[0]
    away_team = game_odds["away_team"].iloc[0]

    per_book_probs = []
    for _, book_rows in game_odds.groupby("bookmaker"):
        prices_by_team = dict(zip(book_rows["team"], book_rows["price"]))
        if home_team not in prices_by_team or away_team not in prices_by_team:
            continue
        prices = np.array([prices_by_team[home_team], prices_by_team[away_team]])
        probs, _ = devig(prices)
        per_book_probs.append(probs)

    if not per_book_probs:
        raise ValueError(f"no book had complete two-sided pricing for {home_team} vs {away_team}")

    consensus = consensus_probabilities(per_book_probs)
    survival = apply_tie_adjustment(consensus, tie_probability)

    return {
        "game_id": game_odds["game_id"].iloc[0],
        "commence_time": game_odds["commence_time"].iloc[0],
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": consensus[0],
        "away_win_probability": consensus[1],
        "home_survival_probability": survival[0],
        "away_survival_probability": survival[1],
        "tie_probability": tie_probability,
        "num_books": len(per_book_probs),
    }


def compute_current_week_probabilities(
    odds_df: pd.DataFrame,
    method: str = DEFAULT_DEVIG_METHOD,
    tie_probability: float = DEFAULT_TIE_PROBABILITY,
) -> pd.DataFrame:
    """One row per game: consensus devigged win and survival probabilities."""
    h2h = odds_df[odds_df["market"] == "h2h"]
    rows = [
        compute_game_probabilities(group, method=method, tie_probability=tie_probability)
        for _, group in h2h.groupby("game_id")
    ]
    columns = [
        "game_id", "commence_time", "home_team", "away_team",
        "home_win_probability", "away_win_probability",
        "home_survival_probability", "away_survival_probability",
        "tie_probability", "num_books",
    ]
    return pd.DataFrame(rows, columns=columns)


def to_team_survival_probabilities(current_week: pd.DataFrame) -> pd.Series:
    """Reshape from one-row-per-game to one-row-per-team, indexed by team abbreviation."""
    home = current_week.set_index("home_team")["home_survival_probability"]
    away = current_week.set_index("away_team")["away_survival_probability"]
    return pd.concat([home, away]).sort_index()
