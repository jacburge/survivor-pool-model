# Survivor

Decision pipeline for the NFL survivor pool. See [plan.md](plan.md) for the
full design (goal, method, phases, technical notes).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run tests

```bash
.venv/bin/pytest
```

## Layout

- `survivor/probability/` — devigging, spread-to-probability, team rating fit
  and projection (Phases 2-3).
- `survivor/decision/` — pick popularity model, payout objective (Phases 4, 6).
- `survivor/data/` — ingestion and storage (Phase 1). Not yet built.
- `survivor/simulation/` — field simulator and rollout scoring (Phase 5). Not
  yet built.

## Status

Core probability-layer and decision-objective math is implemented and
tested against synthetic data (`tests/`). Not yet wired to real odds, schedule,
or pick-percentage data — that's Phase 1, gated on picking an odds provider
(see plan.md's Technical dependencies section).
