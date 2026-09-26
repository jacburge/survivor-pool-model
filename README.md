# Survivor

Decision pipeline for the NFL survivor pool. See [plan.md](plan.md) for the
full design (goal, method, phases, technical notes) and its "Progress"
section at the top for phase-by-phase status.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # fill in THE_ODDS_API_KEY (thin free tier: 500 req/month)
```

## Run tests

```bash
.venv/bin/pytest
```

## Pull real data

Schedule (ESPN) and pick percentages (SurvivorGrid) are free and keyless.
Odds (The Odds API) is quota-limited and only pulled when asked. Everything
lands under `data_store/` (git-ignored, not committed) as timestamped raw
pulls plus a `latest.csv` per source.

```bash
.venv/bin/python scripts/refresh_all.py                 # schedule + pick percentages
.venv/bin/python scripts/refresh_all.py --odds           # + a live odds pull (uses API quota)
.venv/bin/python scripts/refresh_all.py --pick-week 5
```

Other one-off scripts under `scripts/`: `backfill_rating_history.py` and
`collect_historical_*.py` pull multi-season SurvivorGrid history for
calibration/validation; `backtest_week1.py` replays the pipeline blind
against a past week; the `validate_*.py` scripts check each phase's
acceptance criteria against whatever is cached in `data_store/` (no new API
calls). Run these after `refresh_all.py` — several expect `data_store/`
to already be populated.

## Layout

- `survivor/data/` — ingestion and storage (Phase 1): odds, schedule, and
  SurvivorGrid clients, team-key normalization, rating history, rival
  tracker.
- `survivor/probability/` — devigging, spread-to-probability, team rating
  fit and projection (Phases 2-3).
- `survivor/decision/` — pick popularity model, payout objective (Phases 4,
  6).
- `survivor/simulation/` — field simulator, base-policy assignment, and
  rollout scoring (Phase 5).

## Status

Phases 0 through 5 are done and validated against real data — data
ingestion, current- and future-week win probabilities, the pick popularity
model, and the Monte Carlo field simulator with rollout scoring all wired
up and tested (121 tests passing). A blind Week 1, 2026 backtest
(`scripts/backtest_week1.py`) ran the full pipeline end to end successfully.

One open validation gap: Phase 3's lookahead cross-check (projected spreads
within ~1.5 points of real lookahead lines) hadn't passed as of its first
commit — worth rechecking as more lookahead weeks of real data build up.

Not yet built: Phase 6 (joint 10-entry portfolio allocation — the next
task, needed before October 3), Phase 7 (lock-day submission), and most of
Phase 8 (only the rival tracker's storage layer exists so far; popularity
refinement from the tracked field, split-decision logic, and the exact
endgame solver are still open). See [plan.md](plan.md)'s "Progress" section
for the full phase-by-phase breakdown.

This checkout has no `.env` or `data_store/` populated (both git-ignored) —
run the Setup and "Pull real data" steps above before running scripts that
touch real data.
