import pandas as pd
import pytest

from survivor.data.survivorgrid_client import parse_pick_grid

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
  </tbody>
</table>
"""


def test_parses_one_row_per_team():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    assert len(df) == 4
    assert set(df["team"]) == {"PHI", "KC", "WAS", "MIN"}


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
