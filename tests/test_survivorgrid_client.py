import pandas as pd
import pytest

from survivor.data.survivorgrid_client import parse_pick_grid, parse_schedule_grid

# A minimal fixture mirroring survivorgrid.com's real table structure (verified
# via a live raw-HTML pull): four <td class="dist"> for EV/W%/P%, a
# <td class="teamname"> that may carry a trailing result span, then future
# game cells we don't need. One row uses WSH to check the alias mapping.
SAMPLE_GRID_HTML = """
<table class="datatable" id="grid">
  <thead><tr><th>EV</th><th>W%</th><th>P%</th><th>Team</th></tr></thead>
  <tbody>
    <tr id="t1">
      <td class="dist">1.14</td>
      <td class="dist">78.0%</td>
      <td class="dist">15.3%</td>
      <td class="teamname">PHI<span class="resultW">&nbsp;(W)</span></td>
    </tr>
    <tr id="t2">
      <td class="dist">0.62</td>
      <td class="dist">62.5%</td>
      <td class="dist">3.1%</td>
      <td class="teamname">KC</td>
    </tr>
    <tr id="t3">
      <td class="dist">0.40</td>
      <td class="dist">55.0%</td>
      <td class="dist">11.1%</td>
      <td class="teamname">WSH</td>
    </tr>
    <tr id="t4">
      <td class="dist">-</td>
      <td class="dist">-</td>
      <td class="dist">-</td>
      <td class="teamname">MIN</td>
    </tr>
    <tr id="t5">
      <td class="dist">0.30</td>
      <td class="dist">22.0%</td>
      <td class="dist">1.0%</td>
      <td class="teamname">DAL<span class="resultL">&nbsp;(L)</span></td>
    </tr>
  </tbody>
</table>
"""


def test_parses_one_row_per_team():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    assert len(df) == 5
    assert set(df["team"]) == {"PHI", "KC", "WAS", "MIN", "DAL"}


def test_strips_result_marker_from_played_week():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    row = df[df["team"] == "PHI"].iloc[0]
    assert row["win_probability"] == pytest.approx(0.78)
    assert row["pick_percentage"] == pytest.approx(0.153)
    assert row["expected_value"] == pytest.approx(1.14)


def test_alias_team_normalized_to_canonical():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    row = df[df["team"] == "WAS"].iloc[0]
    assert row["pick_percentage"] == pytest.approx(0.111)


def test_missing_values_become_none():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    row = df[df["team"] == "MIN"].iloc[0]
    assert pd.isna(row["win_probability"])
    assert pd.isna(row["pick_percentage"])
    assert pd.isna(row["expected_value"])


def test_pick_percentages_are_plausible_fractions():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    valid = df["pick_percentage"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_missing_table_raises():
    with pytest.raises(ValueError):
        parse_pick_grid("<html><body>no grid here</body></html>")


SCHEDULE_GRID_HTML = """
<table class="datatable" id="grid">
  <thead><tr><th>EV</th><th>W%</th><th>P%</th><th>Team</th><th>1</th><th>2</th><th>3</th></tr></thead>
  <tbody>
    <tr id="t1">
      <td class="dist">1.07</td>
      <td class="dist">79.8%</td>
      <td class="dist">26.8%</td>
      <td class="teamname">LAC</td>
      <td class="gc g19">
        ARI<br>
        <span class="spread">-9.5</span>
      </td>
      <td class="gc rd">
        @BUF<br>
        <span class="spread">3</span>
      </td>
      <td class="gc bye">BYE</td>
      <td class="fv" data-sort-value="0.47"><div class="starrating"></div></td>
    </tr>
    <tr id="t2">
      <td class="dist">0.80</td>
      <td class="dist">60.0%</td>
      <td class="dist">5.0%</td>
      <td class="teamname">SF</td>
      <td class="gc g6 rd">
        <span style="font-size: 9px;" title="Neutral Field">(n)</span>MIN<br>
        <span class="spread">-3</span>
      </td>
      <td class="gc g2">
        DEN<br>
        <span class="spread">-3.5</span>
      </td>
      <td class="gc rd dv">
        @SEA<br>
        <span class="spread">3</span>
      </td>
      <td class="fv" data-sort-value="0.5"><div class="starrating"></div></td>
    </tr>
  </tbody>
</table>
"""


def test_schedule_grid_parses_home_and_away_with_own_spread():
    df = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=1)
    lac_week1 = df[(df.team == "LAC") & (df.week == 1)].iloc[0]
    assert lac_week1["opponent"] == "ARI"
    assert lac_week1["is_home"] is True
    assert lac_week1["spread"] == pytest.approx(-9.5)

    lac_week2 = df[(df.team == "LAC") & (df.week == 2)].iloc[0]
    assert lac_week2["opponent"] == "BUF"
    assert lac_week2["is_home"] is False
    assert lac_week2["spread"] == pytest.approx(3.0)


def test_schedule_grid_marks_bye_week():
    df = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=1)
    lac_week3 = df[(df.team == "LAC") & (df.week == 3)].iloc[0]
    assert bool(lac_week3["is_bye"])
    assert pd.isna(lac_week3["opponent"])
    assert pd.isna(lac_week3["spread"])


def test_schedule_grid_handles_neutral_site_marker():
    df = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=1)
    sf_week1 = df[(df.team == "SF") & (df.week == 1)].iloc[0]
    assert sf_week1["opponent"] == "MIN"
    assert sf_week1["is_home"] is False  # "rd" class present despite the (n) marker


def test_schedule_grid_week_numbers_offset_from_start_week():
    df = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=1)
    assert set(df[df.team == "LAC"]["week"]) == {1, 2, 3}

    df_offset = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=5)
    assert set(df_offset[df_offset.team == "LAC"]["week"]) == {5, 6, 7}


def test_extracts_win_result():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    assert df[df["team"] == "PHI"].iloc[0]["result"] == "W"


def test_extracts_loss_result():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    assert df[df["team"] == "DAL"].iloc[0]["result"] == "L"


def test_upcoming_week_has_no_result():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    assert pd.isna(df[df["team"] == "KC"].iloc[0]["result"])
