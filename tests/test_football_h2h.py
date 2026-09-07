"""Descriptive head-to-head and form — football/h2h.py. No model here.

These feed the top of the Pronóstico tab: "how are these two arriving at this
fixture". The one property worth pinning is `as_of`: form "going into" a match
must not see the match itself or anything after it.
"""

import pandas as pd
import pytest

from football.common import outcome_from_goals
from football.h2h import head_to_head, team_form


def _frame(rows):
    out = pd.DataFrame(rows, columns=["ds", "home_team", "away_team", "home_goals", "away_goals"])
    out["ds"] = pd.to_datetime(out["ds"])
    out["outcome"] = [outcome_from_goals(h, a) for h, a in zip(out["home_goals"], out["away_goals"])]
    return out.sort_values("ds").reset_index(drop=True)


MATCHES = _frame([
    ("2024-01-01", "A", "B", 2, 0),   # A win
    ("2024-01-08", "C", "A", 1, 1),   # draw
    ("2024-01-15", "A", "D", 0, 3),   # A loss
    ("2024-01-22", "B", "A", 1, 2),   # A win (away)
    ("2024-02-01", "A", "B", 0, 0),   # draw, h2h A-B
    ("2024-02-08", "B", "A", 3, 1),   # B win, h2h
])


def test_form_counts_results_from_the_team_perspective():
    form = team_form(MATCHES, "A", last_n=10)
    assert form["played"] == 6
    assert (form["wins"], form["draws"], form["losses"]) == (2, 2, 2)
    assert form["goals_for"] == 2 + 1 + 0 + 2 + 0 + 1
    assert form["points"] == 2 * 3 + 2 * 1
    assert form["results"][-1] == "L"  # most recent: lost at B 1-3


def test_form_last_n_takes_only_the_most_recent():
    form = team_form(MATCHES, "A", last_n=2)
    assert form["played"] == 2
    assert form["results"] == ["D", "L"]  # 0-0 vs B, then 1-3 at B


def test_form_as_of_excludes_the_fixture_and_everything_after():
    form = team_form(MATCHES, "A", last_n=10, as_of="2024-01-22")
    assert form["played"] == 3  # only matches strictly before 22 Jan
    assert "D" == team_form(MATCHES, "A", last_n=1, as_of="2024-01-10")["results"][0]


def test_head_to_head_orients_counts_to_the_given_home_and_away():
    h2h = head_to_head(MATCHES, "A", "B")
    assert h2h["meetings"] == 4
    # A vs B history: A 2-0 home (2024-01-01), B 1-2 home (2024-01-22, A away win),
    # A 0-0 home, B 3-1 home (2024-02-08, A away loss). Oriented to home=A/away=B:
    assert h2h["home_wins"] == 2   # the 2024-01-01 and 2024-01-22 A-wins
    assert h2h["draws"] == 1
    assert h2h["away_wins"] == 1
    assert h2h["avg_goals"] == pytest.approx(9 / 4)


def test_head_to_head_symmetric_in_meetings_count():
    assert head_to_head(MATCHES, "A", "B")["meetings"] == head_to_head(MATCHES, "B", "A")["meetings"]


def test_unknown_team_returns_zero_played_rather_than_raising():
    form = team_form(MATCHES, "Nobody")
    assert form["played"] == 0 and form["results"] == []
