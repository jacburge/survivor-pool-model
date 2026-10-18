import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

API = str(Path(__file__).resolve().parent.parent / "api")

# Shaped like ESPN's scoreboard calendar: weeks run Wednesday 07:00Z to the next Wednesday 06:59Z.
RAW = {"leagues": [{"calendar": [
    {"value": "1", "entries": [{"label": "Hall of Fame Weekend", "value": "1",
                                "startDate": "2026-08-01T07:00Z", "endDate": "2026-08-08T06:59Z"}]},
    {"value": "2", "entries": [
        {"label": "Week 5", "value": "5", "startDate": "2026-10-07T07:00Z", "endDate": "2026-10-14T06:59Z"},
        {"label": "Week 6", "value": "6", "startDate": "2026-10-14T07:00Z", "endDate": "2026-10-21T06:59Z"},
        {"label": "Week 7", "value": "7", "startDate": "2026-10-21T07:00Z", "endDate": "2026-10-28T06:59Z"},
    ]},
]}]}


@pytest.fixture
def nfl(monkeypatch):
    monkeypatch.syspath_prepend(API)
    sys.modules.pop("nfl", None)
    return importlib.import_module("nfl")


def utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


class FakeDB:
    commits = 0

    def commit(self):
        self.commits += 1


def league(current_week, week_end=18):
    return SimpleNamespace(current_week=current_week, week_end=week_end, year=2026)


def test_parse_calendar_keeps_regular_season_only(nfl):
    weeks = nfl.parse_calendar(RAW)
    assert [w for w, _, _ in weeks] == [5, 6, 7]
    assert weeks[1][1] == utc(2026, 10, 14, 7)


def test_week_turns_tuesday_after_monday_night(nfl):
    cal = nfl.parse_calendar(RAW)
    # Monday Oct 12, 11pm ET (Tue 03:00Z) -- Monday night's game may still be on: still week 5
    assert nfl.pool_week_at(cal, utc(2026, 10, 13, 3)) == 5
    # Tuesday Oct 13, 3:30am ET (07:30Z): week 6, a day before ESPN's own Wednesday flip
    assert nfl.pool_week_at(cal, utc(2026, 10, 13, 7, 30)) == 6
    # Wednesday noon ET, when the weekly job runs: week 6
    assert nfl.pool_week_at(cal, utc(2026, 10, 14, 16)) == 6


def test_offseason_is_none(nfl):
    assert nfl.pool_week_at(nfl.parse_calendar(RAW), utc(2027, 3, 1, 12)) is None


def test_advance_moves_forward(nfl):
    lg, db = league(5), FakeDB()
    assert nfl.advance_league_week(db, lg, now=utc(2026, 10, 14, 16), calendar=nfl.parse_calendar(RAW))
    assert lg.current_week == 6 and db.commits == 1


def test_advance_never_moves_back(nfl):
    lg, db = league(7), FakeDB()  # set ahead by hand
    assert not nfl.advance_league_week(db, lg, now=utc(2026, 10, 14, 16), calendar=nfl.parse_calendar(RAW))
    assert lg.current_week == 7 and db.commits == 0


def test_advance_stops_at_last_week(nfl):
    lg = league(5, week_end=5)
    assert not nfl.advance_league_week(FakeDB(), lg, now=utc(2026, 10, 14, 16), calendar=nfl.parse_calendar(RAW))
    assert lg.current_week == 5


def test_failed_pull_leaves_week_alone(nfl, monkeypatch):
    def boom(year):
        raise ConnectionError("ESPN down")
    monkeypatch.setattr(nfl, "season_calendar", boom)
    lg = league(5)
    assert not nfl.advance_league_week(FakeDB(), lg, now=utc(2026, 10, 14, 16))
    assert lg.current_week == 5
