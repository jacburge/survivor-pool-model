import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
API = str(Path(__file__).resolve().parent.parent / "api")


@pytest.fixture
def api(tmp_path, monkeypatch):
    """models/ingest bound to a throwaway SQLite file that has the pre-summary recommendations schema."""
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE recommendations (id INTEGER PRIMARY KEY, league_id INT NOT NULL, week INT NOT NULL, "
        "entry_id TEXT NOT NULL, team TEXT NOT NULL, used_teams_before TEXT, expected_total_payout FLOAT, "
        "payout_se FLOAT, generated_at DATETIME)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.syspath_prepend(API)
    for name in ("models", "ingest"):
        sys.modules.pop(name, None)
    models, ingest = importlib.import_module("models"), importlib.import_module("ingest")
    yield models, ingest
    for name in ("models", "ingest"):
        sys.modules.pop(name, None)


def test_summary_round_trip_and_column_migration(api, tmp_path):
    models, ingest = api
    models.init_db()
    models.init_db()  # idempotent
    (tmp_path / "latest.csv").write_text("entry_id,team,used_teams_before\nentry_1,KC,\n")
    (tmp_path / "latest_summary.json").write_text(json.dumps(
        {"week": 5, "objective": "win_probability", "expected_total_payout": 10.0, "payout_se": 0.5, "win_probability": 0.4}
    ))
    s = ingest.load_summary(tmp_path, 5)
    db = models.SessionLocal()
    ingest.ingest_recommendation(db, 1, 5, tmp_path / "latest.csv", s["expected_total_payout"], s["payout_se"],
                                 s["objective"], s["win_probability"])
    db.commit()
    row = db.query(models.Recommendation).one()
    assert (row.team, row.objective, row.win_probability) == ("KC", "win_probability", 0.4)


def test_stale_or_missing_summary_refused(api, tmp_path):
    _, ingest = api
    with pytest.raises(SystemExit, match="no run summary"):
        ingest.load_summary(tmp_path, 5)
    (tmp_path / "latest_summary.json").write_text(json.dumps({"week": 4}))
    with pytest.raises(SystemExit, match="stale"):
        ingest.load_summary(tmp_path, 5)
