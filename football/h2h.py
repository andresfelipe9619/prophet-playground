"""Head-to-head record and recent form, straight off the match frame.

Descriptive only — there is no model in this file. It answers "how are these
two teams arriving at this fixture" for the top of the dashboard's two-team
view: last-N results, goals, points, the home/away split, and the history
between the pair.

`as_of` is the one subtle parameter. The forecast view shows form *going into*
a fixture, so it must be able to ask for "everything strictly before this date"
— never the fixture itself, never anything after it. Leaking a single future
match here would make the form panel quietly disagree with what was knowable
at kick-off.
"""

import numpy as np
import pandas as pd

from football.common import DRAW


def _before(matches, as_of):
    if as_of is None:
        return matches
    return matches[matches["ds"] < pd.Timestamp(as_of)]


def _involving(matches, team):
    return matches[(matches["home_team"] == team) | (matches["away_team"] == team)]


def team_form(matches, team, last_n=5, as_of=None):
    """The team's last `last_n` matches before `as_of`, from its own perspective."""
    played = _involving(_before(matches, as_of), team).sort_values("ds").tail(last_n)

    results, goals_for, goals_against, points = [], 0, 0, 0
    split = {"home": _empty_split(), "away": _empty_split()}
    for row in played.itertuples():
        at_home = row.home_team == team
        gf, ga = (row.home_goals, row.away_goals) if at_home else (row.away_goals, row.home_goals)
        goals_for += int(gf)
        goals_against += int(ga)
        letter = "D" if gf == ga else ("W" if gf > ga else "L")
        results.append(letter)
        points += {"W": 3, "D": 1, "L": 0}[letter]
        bucket = split["home" if at_home else "away"]
        bucket["played"] += 1
        bucket["wins" if letter == "W" else "draws" if letter == "D" else "losses"] += 1
        bucket["goals_for"] += int(gf)
        bucket["goals_against"] += int(ga)

    return {
        "team": team,
        "played": len(results),
        "results": results,
        "wins": results.count("W"),
        "draws": results.count("D"),
        "losses": results.count("L"),
        "goals_for": goals_for,
        "goals_against": goals_against,
        "points": points,
        "home": split["home"],
        "away": split["away"],
    }


def _empty_split():
    return {"played": 0, "wins": 0, "draws": 0, "losses": 0, "goals_for": 0, "goals_against": 0}


def head_to_head(matches, home_team, away_team, as_of=None):
    """Every past meeting between the pair, counts oriented to the given home/away."""
    window = _before(matches, as_of)
    pair = window[
        ((window["home_team"] == home_team) & (window["away_team"] == away_team))
        | ((window["home_team"] == away_team) & (window["away_team"] == home_team))
    ].sort_values("ds")

    home_wins = draws = away_wins = 0
    total_goals = 0
    for row in pair.itertuples():
        total_goals += int(row.home_goals) + int(row.away_goals)
        if row.outcome == DRAW:
            draws += 1
        elif row.home_team == home_team:
            home_wins += 1 if row.home_goals > row.away_goals else 0
            away_wins += 1 if row.home_goals < row.away_goals else 0
        else:  # venue reversed: the historical home side is our away_team
            away_wins += 1 if row.home_goals > row.away_goals else 0
            home_wins += 1 if row.home_goals < row.away_goals else 0

    last = [
        {"ds": row.ds, "home_team": row.home_team, "away_team": row.away_team,
         "home_goals": int(row.home_goals), "away_goals": int(row.away_goals)}
        for row in pair.tail(5).itertuples()
    ]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "meetings": len(pair),
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "avg_goals": float(total_goals / len(pair)) if len(pair) else float("nan"),
        "last": last,
    }
