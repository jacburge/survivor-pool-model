import pytest

from survivor.data.availability import parse_availability
from survivor.simulation.rival_field import RivalFieldState

# a real Week 3 table from another pool: 2,463 live entries, each having made 2 picks
SPLASH_TABLE = """
It shows how many of the 2,463 live entries can still pick each team this week.

#	Team	Available	%
1	49ers	1,123 / 2,463	46%
2	Jaguars	1,683 / 2,463	68%
3	Eagles	1,688 / 2,463	69%
4	Lions	1,914 / 2,463	78%
5	Steelers	2,064 / 2,463	84%
6	Raiders	2,280 / 2,463	93%
7	Chiefs	2,319 / 2,463	94%
8	Seahawks	2,373 / 2,463	96%
9	Patriots	2,381 / 2,463	97%
10	Rams	2,382 / 2,463	97%
11	Bengals	2,385 / 2,463	97%
12	Bears	2,386 / 2,463	97%
13	Packers	2,389 / 2,463	97%
14	Ravens	2,403 / 2,463	98%
15	Cowboys	2,405 / 2,463	98%
16	Panthers	2,407 / 2,463	98%
17	Bills	2,417 / 2,463	98%
18	Jets	2,435 / 2,463	99%
19	Vikings	2,448 / 2,463	99%
20	Broncos	2,457 / 2,463	100%
21	Giants	2,458 / 2,463	100%
2232	Buccaneers, Chargers, Saints, Browns, Cardinals, Falcons, Titans, Commanders, Texans, Colts, Dolphins	2,463 / 2,463	100%
Takeaway: The 49ers are already burned by more than half the field, and the Jaguars and Eagles by about a third. Any team at 100% is still open to every live entry.

Source: Splash Sports  DV/BL availability
"""


def test_parses_the_live_entry_count_and_every_team():
    n_alive, available = parse_availability(SPLASH_TABLE)

    assert n_alive == 2463
    assert len(available) == 32
    assert available["SF"] == 1123
    assert available["JAX"] == 1683
    assert available["NYG"] == 2458


def test_grouped_row_assigns_the_same_count_to_each_team_in_it():
    _, available = parse_availability(SPLASH_TABLE)

    for team in ["TB", "LAC", "NO", "CLE", "ARI", "ATL", "TEN", "WAS", "HOU", "IND", "MIA"]:
        assert available[team] == 2463


def test_parsed_table_satisfies_the_two_picks_each_identity():
    n_alive, available = parse_availability(SPLASH_TABLE)

    state = RivalFieldState.from_availability(n_alive, picks_made=2, available=available)  # raises if counts don't sum to 2 * n_alive

    assert state.used_counts.sum() == 2 * 2463


def test_wrong_picks_made_is_rejected():
    n_alive, available = parse_availability(SPLASH_TABLE)

    with pytest.raises(ValueError, match="sum to"):
        RivalFieldState.from_availability(n_alive, picks_made=3, available=available)


def test_space_separated_rows_parse_the_same_as_tab_separated():
    spaced = SPLASH_TABLE.replace("\t", " ")

    assert parse_availability(spaced) == parse_availability(SPLASH_TABLE)


def test_truncated_table_is_rejected_rather_than_read_as_unused_teams():
    truncated = "\n".join(line for line in SPLASH_TABLE.splitlines() if "Buccaneers" not in line)

    with pytest.raises(ValueError, match="missing teams"):
        parse_availability(truncated)


def test_unrecognized_team_names_the_row():
    with pytest.raises(ValueError, match="Grizzlies"):
        parse_availability("1\tGrizzlies\t10 / 10\t100%")


def test_inconsistent_live_entry_counts_are_rejected():
    bad = SPLASH_TABLE.replace("2,457 / 2,463", "2,457 / 2,400")

    with pytest.raises(ValueError, match="disagree"):
        parse_availability(bad)


def test_no_rows_is_rejected():
    with pytest.raises(ValueError, match="no availability rows"):
        parse_availability("just some text")


# Brent's report format: "Team: available/live (pct%)", listing only the 19 least-open teams
BRENT_TABLE = """
Vikings: 387/610 (63%)
Bengals: 407/610 (67%)
Ravens: 419/610 (69%)
Texans: 455/610 (75%)
Jaguars: 502/610 (82%)
Seahawks: 534/610 (88%)
Lions: 553/610 (91%)
Bears: 572/610 (94%)
Patriots: 580/610 (95%)
Chiefs: 582/610 (95%)
Packers: 588/610 (96%)
Broncos: 593/610 (97%)
Steelers: 594/610 (97%)
Colts: 594/610 (97%)
49ers: 596/610 (98%)
Rams: 601/610 (99%)
Jets: 604/610 (99%)
Giants: 606/610 (99%)
Cowboys: 608/610 (100%)
"""


def test_parses_the_colon_and_parenthesis_report_format():
    n_alive, available = parse_availability(BRENT_TABLE, allow_missing=True)

    assert n_alive == 610
    assert len(available) == 19
    assert available["MIN"] == 387 and available["DAL"] == 608


def test_partial_table_is_rejected_unless_allowed():
    with pytest.raises(ValueError, match="missing teams"):
        parse_availability(BRENT_TABLE)


def test_fill_missing_teams_spreads_the_used_shortfall_over_unlisted_teams():
    from survivor.data.availability import fill_missing_teams

    n_alive, available = parse_availability(BRENT_TABLE, allow_missing=True)
    filled = fill_missing_teams(n_alive, picks_made=2, available=available)

    assert len(filled) == 32
    assert sum(n_alive - c for c in filled.values()) == 2 * n_alive
    RivalFieldState.from_availability(n_alive, picks_made=2, available=filled)
    unlisted = set(filled) - set(available)
    assert all(n_alive - 1 <= filled[t] <= n_alive for t in unlisted)


def test_fill_missing_teams_rejects_an_overfull_table():
    from survivor.data.availability import fill_missing_teams

    n_alive, available = parse_availability(BRENT_TABLE, allow_missing=True)
    with pytest.raises(ValueError):
        fill_missing_teams(n_alive, picks_made=1, available=available)
