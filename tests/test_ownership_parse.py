import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from ownership_parse import parse_ownership_text  # noqa: E402


def _row(text):
    (row,) = parse_ownership_text(text)
    return row


def test_count_only():
    assert _row("Seahawks 40") == {"team": "SEA", "pct": None, "pick_count": 40}


def test_decimal_percent_only():
    assert _row("Lions 8.3%") == {"team": "DET", "pct": 8.3, "pick_count": None}


def test_decimal_percent_and_count():
    assert _row("Chiefs 12.5% 34") == {"team": "KC", "pct": 12.5, "pick_count": 34}


def test_integer_percent_and_thousands_count():
    assert _row("Bills 12% 1,234") == {"team": "BUF", "pct": 12.0, "pick_count": 1234}


def test_full_name_and_digit_nickname():
    assert _row("Kansas City Chiefs 5") ["team"] == "KC"
    assert _row("49ers 7%")["team"] == "SF"


def test_bad_lines_skipped():
    assert [r["team"] for r in parse_ownership_text("garbage\n\nJets 3")] == ["NYJ"]
