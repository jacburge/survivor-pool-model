"""Parse a pasted Splash ownership table into rows. Kept free of FastAPI so it can be unit tested."""
from __future__ import annotations

import re

try:
    from survivor.data.team_keys import to_abbreviation
except ImportError:
    to_abbreviation = None  # her package not installed in this environment -- fall back to raw text

# Her to_abbreviation only knows full names ("Kansas City Chiefs") and
# existing abbreviations -- not bare nicknames. Splash's own export uses
# nicknames only ("Chiefs", "Seahawks"), so without this map almost every
# real paste would silently fail to resolve and never match a team on the
# board. Kept here, not in her package -- this is an ownership-paste
# concern, not a general team-name one.
NICKNAME_TO_ABBREVIATION = {
    "CARDINALS": "ARI", "FALCONS": "ATL", "RAVENS": "BAL", "BILLS": "BUF",
    "PANTHERS": "CAR", "BEARS": "CHI", "BENGALS": "CIN", "BROWNS": "CLE",
    "COWBOYS": "DAL", "BRONCOS": "DEN", "LIONS": "DET", "PACKERS": "GB",
    "TEXANS": "HOU", "COLTS": "IND", "JAGUARS": "JAX", "JAGS": "JAX",
    "CHIEFS": "KC", "RAIDERS": "LV", "CHARGERS": "LAC", "RAMS": "LAR",
    "DOLPHINS": "MIA", "VIKINGS": "MIN", "PATRIOTS": "NE", "SAINTS": "NO",
    "GIANTS": "NYG", "JETS": "NYJ", "EAGLES": "PHI", "STEELERS": "PIT",
    "49ERS": "SF", "NINERS": "SF", "SEAHAWKS": "SEA", "BUCCANEERS": "TB",
    "BUCS": "TB", "TITANS": "TEN", "COMMANDERS": "WAS",
}

# TEAM, then an optional percentage (decimals allowed, % required so it can't be
# mistaken for a count), then an optional count. At least one of the two is required.
LINE_RE = re.compile(
    r"^(?P<team>.+?)[\s,]+(?:(?P<pct>\d+(?:\.\d+)?)\s*%)?\s*(?P<count>\d[\d,]*)?\s*$"
)


def parse_ownership_text(text: str) -> list[dict]:
    """Handles 'TEAM COUNT', 'TEAM PCT%', and 'TEAM PCT% COUNT' (Splash's
    own Stats-tab order) per line; PCT may be decimal ('12.5%'). When both a
    percentage and a count are present, both are kept. Lines that don't
    parse are skipped, not errored on, so one bad paste doesn't block the
    rest."""
    rows = []
    for line in text.strip().splitlines():
        m = LINE_RE.match(line.strip())
        if not m or (m.group("pct") is None and m.group("count") is None):
            continue
        team_raw = m.group("team").strip()
        pct = float(m.group("pct")) if m.group("pct") is not None else None
        count = int(m.group("count").replace(",", "")) if m.group("count") else None
        team = team_raw.upper()
        if to_abbreviation:
            try:
                team = to_abbreviation(team_raw)
            except KeyError:
                team = NICKNAME_TO_ABBREVIATION.get(team.replace(" ", ""), team)
        rows.append({"team": team, "pct": pct, "pick_count": count})
    return rows
