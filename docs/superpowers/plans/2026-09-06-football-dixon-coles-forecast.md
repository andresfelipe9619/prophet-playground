# Football Dixon-Coles Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the football dashboard a two-team view that shows head-to-head stats plus a Dixon-Coles 1X2 and scoreline forecast next to the closing-line market and a measured "does it beat the market" verdict, running on real European and Colombian data.

**Architecture:** Six new domain modules under `football/` (scoring rules, the Dixon-Coles model, head-to-head stats, the `new/COL.csv` "extra" contract, a paired market-comparison test, a walk-forward backtest) plus a fourth dashboard tab. `core/` is untouched — the domain supplies the null (the market) and the scoring rule, exactly as `lottery/` does. Every surface showing model probabilities also shows the market and links the verdict.

**Tech Stack:** Python 3.11+, pandas ≥ 2.2, numpy ≥ 1.26, scipy ≥ 1.13 (`scipy.optimize.minimize`, `scipy.stats.poisson`), streamlit + plotly (dashboard only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-football-dixon-coles-forecast-design.md` — read it alongside this plan.

## Global Constraints

- **`core/` stays domain-free.** Nothing mentioning a team, match, goal or odds goes in `core/`. Copied verbatim from `CLAUDE.md`.
- **Outcome order is `("H", "D", "A")`**, from `football/common.py:OUTCOMES`. Every probability vector, score and column triple is `(home, draw, away)` in that order. Never re-declare it. RPS depends on this order being the natural one.
- **Model probabilities are never shown or scored without the market beside them** and the measured verdict linked. A surface that implies a forecast is good without beating the closing line is a spec violation.
- **One odds source per frame, or none.** Opening and closing odds never mix. Extra (`new/COL.csv`) files are opening-odds-only and `odds_are_closing` is always `False` for them.
- **Chance/market comparisons are one-sided.** Only `p_value_greater` may back a "beats the market" claim.
- **Every evaluation surface reports the naive and the corrected verdict together** and points the reader at the corrected one.
- **Language:** English for code, docstrings, comments, docs, commits. Spanish only for user-facing dashboard strings.
- **New test dependency ⇒ add it to BOTH `requirements.txt` and `requirements-test.txt`.** (None is expected — scipy is already in both.)
- **Tests are deterministic**, built from `football/sample_data.py` (seeded) or hand-built frames. Slow model fits are marked `@pytest.mark.slow`.
- **Charts go through `dashboard/ui.py:chart(fig, title, key)` / `section(title, key)`** — never a bare `st.plotly_chart`. Every `HELP` string says what the chart does *not* mean.
- Commit after every green test cycle. Commit messages end with the trailer:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT`

---

## File Structure

| File | Responsibility |
| --- | --- |
| `football/scoring.py` (new) | Brier, RPS, log-loss, per-match contributions, skill score. Pure. |
| `football/h2h.py` (new) | `team_form`, `head_to_head` — descriptive only, no model. Pure. |
| `football/dixon_coles.py` (new) | The Dixon-Coles model: fit by MLE, scoreline matrix, 1X2 / scores / O-U / BTTS. |
| `football/extra_processor.py` (new) | The `new/COL.csv` contract → the same tidy shape as `processor.py`, opening odds only. |
| `football/downloader.py` (modify) | `--extra` flag: fetch `new/{CODE}.csv`, validate through `extra_processor`, write. |
| `football/evaluation.py` (new) | `beats_market_test` — paired proper-score difference vs the market, through `core/significance.py`. |
| `football/backtest.py` (new) | Walk-forward and date-cutoff evaluation of Dixon-Coles vs the market. |
| `dashboard/football_page.py` (modify) | Source toggle, the Pronóstico tab, the Resultados evaluation section. |
| `dashboard/ui.py` (modify) | New `fb_*` `HELP` keys. |
| `tests/test_football_scoring.py` (new) | Scoring-rule invariants. |
| `tests/test_football_h2h.py` (new) | Form / h2h / `as_of` cutoff. |
| `tests/test_football_dixon_coles.py` (new) | Fit recovers the truth; matrix sums to 1; rho affects low scores; unknown team raises. |
| `tests/test_football_extra_processor.py` (new) | `new/COL.csv` mapping, opening-only source, league filter. |
| `tests/fixtures/new_COL_sample.csv` (new) | Saved-style extra file for the parser test. |
| `tests/test_football_evaluation.py` (new) | Paired test wiring; both verdict keys; correction tightens. |
| `tests/test_football_backtest.py` (new, `slow`) | Beats a soft market, does not beat a sharp one. |
| `docs/football.md`, `docs/data-pipeline.md`, `docs/models.md`, `docs/evaluation.md`, `docs/dashboard.md`, `CLAUDE.md` (modify) | Keep the doc-map in sync. |

---

## Task 1: `football/scoring.py` — proper scoring rules

**Files:**
- Create: `football/scoring.py`
- Test: `tests/test_football_scoring.py`

**Interfaces:**
- Consumes: `football.common.OUTCOMES`, `football.common.outcome_index`.
- Produces:
  - `brier_score(probs, outcomes) -> float`
  - `ranked_probability_score(probs, outcomes) -> float`
  - `log_loss(probs, outcomes) -> float`
  - `per_match_scores(probs, outcomes, metric="rps") -> np.ndarray` (shape `(n,)`, the term before the mean)
  - `skill_score(model_probs, baseline_probs, outcomes, metric="rps") -> float`
  - `METRICS = ("brier", "rps", "log_loss")`
  - `probs` is array-like `(n, 3)` or `(3,)` in `OUTCOMES` order; `outcomes` is a sequence of `"H"/"D"/"A"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_football_scoring.py
"""Proper scoring rules for the three-way football outcome — football/scoring.py.

The outcome is ORDERED (H < D < A as a rank of "how much the home side won by"),
so the headline metric is RPS, which charges less for a near miss than for a
far one. Brier is symmetric and does not. Every function here takes probs in
OUTCOMES order; getting that order wrong silently corrupts RPS.
"""

import numpy as np
import pytest

from football.scoring import (
    METRICS,
    brier_score,
    log_loss,
    per_match_scores,
    ranked_probability_score,
    skill_score,
)


def test_perfect_forecast_scores_zero_on_every_metric():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = ["H", "D", "A"]
    assert brier_score(probs, outcomes) == pytest.approx(0.0)
    assert ranked_probability_score(probs, outcomes) == pytest.approx(0.0)
    assert log_loss(probs, outcomes) == pytest.approx(0.0)


def test_uniform_forecast_has_known_brier_and_rps():
    probs = np.full((4, 3), 1 / 3)
    outcomes = ["H", "D", "A", "H"]
    # Brier per match: (1-1/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 0.6667
    assert brier_score(probs, outcomes) == pytest.approx(2 / 3)
    # RPS for a uniform forecast on a 3-way outcome is 1/9 regardless of which
    # outcome occurred, by symmetry of the cumulative distribution.
    assert ranked_probability_score(probs, outcomes) == pytest.approx(1 / 9)


def test_rps_rewards_the_near_miss_but_brier_does_not():
    # Away win occurred. Forecast X put all mass on Draw (adjacent);
    # forecast Y put all mass on Home win (far). RPS should prefer X; Brier ties.
    near = np.array([[0.0, 1.0, 0.0]])
    far = np.array([[1.0, 0.0, 0.0]])
    outcomes = ["A"]
    assert ranked_probability_score(near, outcomes) < ranked_probability_score(far, outcomes)
    assert brier_score(near, outcomes) == pytest.approx(brier_score(far, outcomes))


def test_per_match_scores_mean_equals_the_aggregate():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=20)
    outcomes = rng.choice(["H", "D", "A"], size=20)
    for metric in METRICS:
        agg = {"brier": brier_score, "rps": ranked_probability_score, "log_loss": log_loss}[metric]
        assert per_match_scores(probs, outcomes, metric).mean() == pytest.approx(agg(probs, outcomes))


def test_skill_score_positive_when_model_beats_baseline():
    outcomes = ["H", "H", "H", "H"]
    good = np.full((4, 3), 0.0) + np.array([0.9, 0.05, 0.05])
    weak = np.full((4, 3), 1 / 3)
    assert skill_score(good, weak, outcomes, metric="rps") > 0
    assert skill_score(weak, good, outcomes, metric="rps") < 0


def test_skill_score_drops_rows_where_either_side_is_nan():
    outcomes = ["H", "D", "A"]
    model = np.array([[0.8, 0.1, 0.1], [np.nan, np.nan, np.nan], [0.2, 0.3, 0.5]])
    market = np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3], [np.nan, np.nan, np.nan]])
    # Only row 0 survives in both; the call must not raise and must score 1 match.
    value = skill_score(model, market, outcomes, metric="brier")
    assert np.isfinite(value)


def test_single_forecast_metric_raises_on_nan():
    with pytest.raises(ValueError):
        brier_score(np.array([[np.nan, np.nan, np.nan]]), ["H"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_football_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'football.scoring'`

- [ ] **Step 3: Write the implementation**

```python
# football/scoring.py
"""Proper scoring rules for the three-way match outcome.

This is the football counterpart of the lottery's set-based hit counting: the
number that says how good a probabilistic forecast is. The outcome is treated
as ORDERED — home win, draw, away win, in that order — so the default metric
is the ranked probability score, which penalises a forecast that put its mass
on the draw when the away side won less than one that put its mass on the home
win. Brier is offered too but is symmetric across outcomes and does not make
that distinction; log-loss is offered for the confidently-wrong case.

`skill_score(model, baseline, ...)` is the headline "vs the market" figure:
positive means the model scored better than the baseline on the same matches.

Every function takes `probs` as `(n, 3)` (or `(3,)`) in `football.common.OUTCOMES`
order. Getting that order wrong corrupts RPS silently, which is why the order
lives in one place and is imported, never redeclared.
"""

import numpy as np

from football.common import OUTCOMES, outcome_index

METRICS = ("brier", "rps", "log_loss")
_LOG_CLIP = 1e-15


def _as_matrix(probs):
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    if probs.ndim != 2 or probs.shape[1] != len(OUTCOMES):
        raise ValueError(
            f"Expected probabilities shaped (n, {len(OUTCOMES)}) in {OUTCOMES} order, "
            f"got {probs.shape}."
        )
    return probs


def _onehot(outcomes):
    idx = np.array([outcome_index(o) for o in outcomes])
    out = np.zeros((idx.size, len(OUTCOMES)))
    out[np.arange(idx.size), idx] = 1.0
    return out


def _require_finite(probs):
    if np.isnan(probs).any():
        raise ValueError(
            "probs contains NaN. The single-forecast metrics score every row; for a "
            "frame with partial market coverage use skill_score, which drops unmatched rows."
        )


def per_match_scores(probs, outcomes, metric="rps"):
    """The per-match score contribution — the quantity averaged by the aggregate metrics.

    `evaluation.py` needs these one-per-match so it can form a paired difference
    between the model and the market and test its mean against zero.
    """
    p = _as_matrix(probs)
    y = _onehot(outcomes)
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"{p.shape[0]} probability rows but {y.shape[0]} outcomes.")
    if metric == "brier":
        return np.sum((p - y) ** 2, axis=1)
    if metric == "rps":
        cp = np.cumsum(p, axis=1)[:, :-1]
        cy = np.cumsum(y, axis=1)[:, :-1]
        return np.sum((cp - cy) ** 2, axis=1) / (len(OUTCOMES) - 1)
    if metric == "log_loss":
        return -np.sum(y * np.log(np.clip(p, _LOG_CLIP, 1.0)), axis=1)
    raise ValueError(f"Unknown metric {metric!r}. Available: {METRICS}.")


def brier_score(probs, outcomes):
    """Mean squared error between the probability vector and the outcome indicator."""
    _require_finite(_as_matrix(probs))
    return float(per_match_scores(probs, outcomes, "brier").mean())


def ranked_probability_score(probs, outcomes):
    """Mean squared error between the cumulative forecast and cumulative outcome.

    0 is perfect. For a 3-way outcome a uniform forecast scores 1/9. Lower is
    better, and a near miss (mass on the adjacent outcome) costs less than a
    far one — the property Brier lacks.
    """
    _require_finite(_as_matrix(probs))
    return float(per_match_scores(probs, outcomes, "rps").mean())


def log_loss(probs, outcomes):
    """Mean negative log probability assigned to the outcome that happened."""
    _require_finite(_as_matrix(probs))
    return float(per_match_scores(probs, outcomes, "log_loss").mean())


_AGGREGATE = {"brier": brier_score, "rps": ranked_probability_score, "log_loss": log_loss}


def skill_score(model_probs, baseline_probs, outcomes, metric="rps"):
    """`1 - score(model) / score(baseline)` on the matches both can score.

    Positive means the model beat the baseline. Rows where either side has a
    NaN (a match with no market price, say) are dropped from both before
    scoring, so the two are always compared on the same matches.
    """
    if metric not in _AGGREGATE:
        raise ValueError(f"Unknown metric {metric!r}. Available: {METRICS}.")
    model = _as_matrix(model_probs)
    baseline = _as_matrix(baseline_probs)
    outcomes = np.asarray(list(outcomes))
    keep = ~(np.isnan(model).any(axis=1) | np.isnan(baseline).any(axis=1))
    if not keep.any():
        return float("nan")
    fn = _AGGREGATE[metric]
    base = fn(baseline[keep], outcomes[keep])
    if base == 0:
        return float("nan")
    return float(1.0 - fn(model[keep], outcomes[keep]) / base)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_football_scoring.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add football/scoring.py tests/test_football_scoring.py
git commit -m "Add football/scoring.py: Brier, RPS, log-loss and skill score

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT"
```

---

## Task 2: `football/h2h.py` — head-to-head and form

**Files:**
- Create: `football/h2h.py`
- Test: `tests/test_football_h2h.py`

**Interfaces:**
- Consumes: a tidy match frame with `MATCH_COLUMNS` (`ds, home_team, away_team, home_goals, away_goals, outcome`).
- Produces:
  - `team_form(matches, team, last_n=5, as_of=None) -> dict` with keys
    `team, played, results` (list of `"W"/"D"/"L"` most-recent-last),
    `wins, draws, losses, goals_for, goals_against, points, home, away`
    (`home`/`away` are dicts `{played, wins, draws, losses, goals_for, goals_against}`).
  - `head_to_head(matches, home_team, away_team, as_of=None) -> dict` with keys
    `meetings` (int), `home_team, away_team, home_wins, draws, away_wins`
    (counts oriented to the *given* home/away teams regardless of venue in the
    historical meeting), `avg_goals`, `last` (list of dicts
    `{ds, home_team, away_team, home_goals, away_goals}` most-recent-last, ≤ 5).
  - `as_of` (a date-like or `None`): only matches with `ds < as_of` are used.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_football_h2h.py
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
    assert "W" == team_form(MATCHES, "A", last_n=1, as_of="2024-01-10")["results"][0]


def test_head_to_head_orients_counts_to_the_given_home_and_away():
    h2h = head_to_head(MATCHES, "A", "B")
    assert h2h["meetings"] == 3
    # A vs B history: A 2-0 (2024-01-22 was B home vs A -> counts as away_team A win),
    # 0-0 draw, B 3-1 (away_team A loss). Oriented to home=A/away=B:
    assert h2h["home_wins"] == 1   # the 2024-01-22 B-home A-win, A is 'home_team' arg
    assert h2h["draws"] == 1
    assert h2h["away_wins"] == 1
    assert h2h["avg_goals"] == pytest.approx((3 + 0 + 4) / 3)


def test_head_to_head_symmetric_in_meetings_count():
    assert head_to_head(MATCHES, "A", "B")["meetings"] == head_to_head(MATCHES, "B", "A")["meetings"]


def test_unknown_team_returns_zero_played_rather_than_raising():
    form = team_form(MATCHES, "Nobody")
    assert form["played"] == 0 and form["results"] == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_football_h2h.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'football.h2h'`

- [ ] **Step 3: Write the implementation**

```python
# football/h2h.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_football_h2h.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add football/h2h.py tests/test_football_h2h.py
git commit -m "Add football/h2h.py: recent form and head-to-head record

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT"
```

---

## Task 3: `football/dixon_coles.py` — the model

**Files:**
- Create: `football/dixon_coles.py`
- Test: `tests/test_football_dixon_coles.py`

**Interfaces:**
- Consumes: `football.common.OUTCOMES`; a tidy match frame (`ds, home_team, away_team, home_goals, away_goals`); `scipy.optimize.minimize`, `scipy.stats.poisson`.
- Produces:
  - `class UnknownTeamError(ValueError)`
  - `class DixonColes`:
    - `DixonColes.fit(matches, half_life=None, max_iter=200) -> DixonColes` (classmethod)
    - `.teams -> tuple[str, ...]`
    - `.params -> dict` with `mu, home_advantage, rho, attack (dict), defence (dict)`
    - `.scoreline_matrix(home_team, away_team, max_goals=10) -> np.ndarray` `(max_goals+1, max_goals+1)`, sums to 1
    - `.predict_outcome(home_team, away_team) -> np.ndarray` shape `(3,)` in `OUTCOMES` order
    - `.most_likely_scores(home_team, away_team, n=5) -> list[tuple[tuple[int,int], float]]`
    - `.over_under(home_team, away_team, line=2.5) -> tuple[float, float]` `(p_over, p_under)`
    - `.both_teams_to_score(home_team, away_team) -> float`
    - `.predict_matches(matches) -> np.ndarray` shape `(n, 3)`; raises `UnknownTeamError` if any team is unseen
  - `independent_poisson_matrix(lambda_home, lambda_away, max_goals=10) -> np.ndarray` (module function, the rho=0 baseline, used by a test)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_football_dixon_coles.py
"""The Dixon-Coles model — football/dixon_coles.py.

Two independent Poisson goal counts, plus the low-score correction (rho) that
pulls probability toward 0-0/1-0/0-1/1-1 to fix the draw deficit the plain
model has. Fitted by maximum likelihood.

The checks that matter: a model fitted on data generated from a known truth
recovers that truth on average; the scoreline grid is a proper distribution;
the rho term actually moves the low-score cells; an unseen team is refused
loudly rather than scored on nothing.
"""

import numpy as np
import pytest

from football.dixon_coles import (
    DixonColes,
    UnknownTeamError,
    independent_poisson_matrix,
)
from football.sample_data import load_sample_and_preprocess

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fitted():
    matches = load_sample_and_preprocess(n_teams=14, seed=1)
    return DixonColes.fit(matches), matches


def test_scoreline_matrix_is_a_distribution(fitted):
    model, matches = fitted
    grid = model.scoreline_matrix(model.teams[0], model.teams[1], max_goals=10)
    assert grid.shape == (11, 11)
    assert (grid >= 0).all()
    assert grid.sum() == pytest.approx(1.0, abs=1e-9)


def test_predict_outcome_sums_to_one_and_is_in_hda_order(fitted):
    model, _ = fitted
    p = model.predict_outcome(model.teams[2], model.teams[5])
    assert p.shape == (3,)
    assert p.sum() == pytest.approx(1.0)
    # Home advantage is positive in the generator, so on average p_home > p_away
    # across a slate of neutral fixtures.
    slate = np.mean([model.predict_outcome(h, a)
                     for h in model.teams for a in model.teams if h != a], axis=0)
    assert slate[0] > slate[2]


def test_attack_effects_are_centred(fitted):
    model, _ = fitted
    assert sum(model.params["attack"].values()) == pytest.approx(0.0, abs=1e-6)
    assert sum(model.params["defence"].values()) == pytest.approx(0.0, abs=1e-6)


def test_fit_recovers_the_generative_outcome_probabilities(fitted):
    model, matches = fitted
    truth = load_sample_and_preprocess(n_teams=14, seed=1)[["p_true_home", "p_true_draw", "p_true_away"]].to_numpy()
    pred = model.predict_matches(matches)
    # Mean absolute error across all matches and outcomes, well under 5 points.
    assert np.abs(pred - truth).mean() < 0.05


def test_rho_moves_the_low_score_cells_only():
    lh, la = 1.4, 1.1
    base = independent_poisson_matrix(lh, la, max_goals=8)
    model = DixonColes._from_params(
        teams=("X", "Y"), mu=np.log(1.25), home_advantage=0.0,
        attack={"X": 0.0, "Y": 0.0}, defence={"X": 0.0, "Y": 0.0}, rho=-0.15,
    )
    # Force the lambdas by construction: mu chosen so exp(mu) ~ 1.25; use a helper.
    grid = model._grid(lh, la, max_goals=8)
    changed = ~np.isclose(grid / grid.sum(), base)
    # Only (0,0),(0,1),(1,0),(1,1) differ before renormalisation shifts everything;
    # after renormalisation every cell moves a little, but the four corners move most.
    corners = np.array([grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]])
    base_corners = np.array([base[0, 0], base[0, 1], base[1, 0], base[1, 1]])
    assert np.abs(corners / grid.sum() - base_corners).max() > 1e-3
    # Negative rho lifts the exact-draw mass relative to plain Poisson.
    assert np.trace(grid) / grid.sum() > np.trace(base)


def test_unknown_team_raises(fitted):
    model, _ = fitted
    with pytest.raises(UnknownTeamError):
        model.predict_outcome("Nonexistent FC", model.teams[0])


def test_half_life_changes_the_fit(fitted):
    model, matches = fitted
    decayed = DixonColes.fit(matches, half_life=90)
    assert decayed.params["rho"] != model.params["rho"] or (
        decayed.params["attack"] != model.params["attack"]
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_football_dixon_coles.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'football.dixon_coles'`

- [ ] **Step 3: Write the implementation**

```python
# football/dixon_coles.py
"""Dixon-Coles: two Poisson goal counts with a low-score dependence correction.

The plain independent-Poisson model is the standard first pass at football
scores and is known to under-produce draws and low-scoring correlated
scorelines. Dixon & Coles (1997) fix that with one extra parameter, rho, that
reweights the four cells 0-0, 1-0, 0-1, 1-1. From one fitted model every
market falls out: 1X2, correct score, over/under, both teams to score.

Parameters, fitted by maximum likelihood over past matches:

    lambda_home = exp(mu + home_advantage + attack[home] + defence[away])
    lambda_away = exp(mu +                  attack[away] + defence[home])

`attack` and `defence` are each constrained to sum to zero for identifiability
against the intercept `mu`. `half_life` (in days) applies an exponential
time-decay weight to older matches so recent form counts for more; None means
every match counts equally.

This model does NOT know about the market. It produces a forecast;
`football/evaluation.py` is what decides whether that forecast is worth
anything, by scoring it against the closing line.
"""

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from football.common import OUTCOMES  # noqa: F401  (kept for the order contract)

# rho must keep every tau cell positive over realistic lambdas; this bound is
# wide enough for real data and narrow enough that the optimiser stays sane.
_RHO_BOUNDS = (-0.4, 0.4)


class UnknownTeamError(ValueError):
    """A prediction was asked for a team the model was never fitted on."""


def _tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Dixon-Coles low-score correction, broadcast over goal-count arrays."""
    h = np.asarray(home_goals)
    a = np.asarray(away_goals)
    out = np.ones(np.broadcast(h, a).shape, dtype=float)
    out = np.where((h == 0) & (a == 0), 1.0 - lambda_home * lambda_away * rho, out)
    out = np.where((h == 0) & (a == 1), 1.0 + lambda_home * rho, out)
    out = np.where((h == 1) & (a == 0), 1.0 + lambda_away * rho, out)
    out = np.where((h == 1) & (a == 1), 1.0 - rho, out)
    return out


def independent_poisson_matrix(lambda_home, lambda_away, max_goals=10):
    """The rho = 0 scoreline grid: the plain independent-Poisson baseline."""
    goals = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))


class DixonColes:
    def __init__(self, teams, mu, home_advantage, attack, defence, rho):
        self.teams = tuple(teams)
        self._index = {team: i for i, team in enumerate(self.teams)}
        self.params = {
            "mu": float(mu),
            "home_advantage": float(home_advantage),
            "rho": float(rho),
            "attack": dict(attack),
            "defence": dict(defence),
        }

    # -- construction -----------------------------------------------------
    @classmethod
    def _from_params(cls, teams, mu, home_advantage, attack, defence, rho):
        return cls(teams, mu, home_advantage, attack, defence, rho)

    @classmethod
    def fit(cls, matches, half_life=None, max_iter=200):
        """Maximum-likelihood fit on a tidy match frame."""
        teams = tuple(sorted(set(matches["home_team"]) | set(matches["away_team"])))
        idx = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        home_idx = matches["home_team"].map(idx).to_numpy()
        away_idx = matches["away_team"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy(dtype=float)
        ag = matches["away_goals"].to_numpy(dtype=float)

        if half_life:
            age_days = (matches["ds"].max() - matches["ds"]).dt.days.to_numpy(dtype=float)
            weights = 0.5 ** (age_days / float(half_life))
        else:
            weights = np.ones(len(matches))

        # Parameter vector: [mu, home_adv, attack[0..n-2], defence[0..n-2], rho].
        # The last attack/defence value is fixed at minus the sum of the rest.
        def unpack(theta):
            mu, home_adv = theta[0], theta[1]
            a_free = theta[2:2 + n - 1]
            d_free = theta[2 + n - 1:2 + 2 * (n - 1)]
            rho = theta[-1]
            attack = np.append(a_free, -a_free.sum())
            defence = np.append(d_free, -d_free.sum())
            return mu, home_adv, attack, defence, rho

        def neg_log_likelihood(theta):
            mu, home_adv, attack, defence, rho = unpack(theta)
            lam_h = np.exp(mu + home_adv + attack[home_idx] + defence[away_idx])
            lam_a = np.exp(mu + attack[away_idx] + defence[home_idx])
            ll = (poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a)
                  + np.log(np.clip(_tau(hg, ag, lam_h, lam_a, rho), 1e-12, None)))
            return -float(np.sum(weights * ll))

        theta0 = np.concatenate([[np.log(max(hg.mean(), 0.5)), 0.25],
                                 np.zeros(n - 1), np.zeros(n - 1), [0.0]])
        bounds = [(None, None), (None, None)] + [(None, None)] * (2 * (n - 1)) + [_RHO_BOUNDS]
        result = minimize(neg_log_likelihood, theta0, method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": max_iter})
        if not result.success:
            warnings.warn(f"Dixon-Coles fit did not converge: {result.message}", stacklevel=2)

        mu, home_adv, attack, defence, rho = unpack(result.x)
        if abs(rho - _RHO_BOUNDS[0]) < 1e-4 or abs(rho - _RHO_BOUNDS[1]) < 1e-4:
            warnings.warn(
                f"rho hit its bound ({rho:.3f}); the low-score correction is at its limit.",
                stacklevel=2,
            )
        return cls(
            teams=teams, mu=mu, home_advantage=home_adv, rho=rho,
            attack={t: float(attack[i]) for t, i in idx.items()},
            defence={t: float(defence[i]) for t, i in idx.items()},
        )

    # -- prediction -----------------------------------------------------
    def _team(self, name):
        if name not in self._index:
            raise UnknownTeamError(
                f"{name!r} is not in this model. Fitted teams: {', '.join(self.teams)}."
            )
        return name

    def _lambdas(self, home_team, away_team):
        p = self.params
        lam_h = np.exp(p["mu"] + p["home_advantage"]
                       + p["attack"][home_team] + p["defence"][away_team])
        lam_a = np.exp(p["mu"] + p["attack"][away_team] + p["defence"][home_team])
        return float(lam_h), float(lam_a)

    def _grid(self, lambda_home, lambda_away, max_goals=10):
        grid = independent_poisson_matrix(lambda_home, lambda_away, max_goals)
        goals = np.arange(max_goals + 1)
        h, a = np.meshgrid(goals, goals, indexing="ij")
        grid = grid * _tau(h, a, lambda_home, lambda_away, self.params["rho"])
        return np.clip(grid, 0.0, None)

    def scoreline_matrix(self, home_team, away_team, max_goals=10):
        lam_h, lam_a = self._lambdas(self._team(home_team), self._team(away_team))
        grid = self._grid(lam_h, lam_a, max_goals)
        return grid / grid.sum()

    def predict_outcome(self, home_team, away_team):
        grid = self.scoreline_matrix(home_team, away_team)
        return np.array([
            float(np.tril(grid, -1).sum()),   # home win
            float(np.trace(grid)),            # draw
            float(np.triu(grid, 1).sum()),    # away win
        ])

    def most_likely_scores(self, home_team, away_team, n=5):
        grid = self.scoreline_matrix(home_team, away_team)
        flat = np.dstack(np.unravel_index(np.argsort(grid, axis=None)[::-1], grid.shape))[0]
        return [((int(h), int(a)), float(grid[h, a])) for h, a in flat[:n]]

    def over_under(self, home_team, away_team, line=2.5):
        grid = self.scoreline_matrix(home_team, away_team)
        goals = np.arange(grid.shape[0])
        totals = goals[:, None] + goals[None, :]
        p_over = float(grid[totals > line].sum())
        return p_over, 1.0 - p_over

    def both_teams_to_score(self, home_team, away_team):
        grid = self.scoreline_matrix(home_team, away_team)
        return float(grid[1:, 1:].sum())

    def predict_matches(self, matches):
        return np.array([
            self.predict_outcome(row.home_team, row.away_team)
            for row in matches.itertuples()
        ])
```

Note for the implementer: `test_rho_moves_the_low_score_cells_only` constructs a
model with `_from_params` and then calls the private `_grid(lh, la, ...)`
directly — that is intentional, it isolates the correction math from the fit.
Keep `_grid` taking explicit lambdas.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_football_dixon_coles.py -q`
Expected: PASS (7 passed). If `test_fit_recovers_the_generative_outcome_probabilities` is borderline, widen `n_teams`/match count in the fixture, not the tolerance beyond 0.05.

- [ ] **Step 5: Run the fast suite to confirm nothing else broke**

Run: `python -m pytest -m "not slow" -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add football/dixon_coles.py tests/test_football_dixon_coles.py
git commit -m "Add football/dixon_coles.py: MLE fit, scoreline grid, 1X2 / O-U / BTTS

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT"
```

---

## Task 4: `football/extra_processor.py` + `--extra` download

**Files:**
- Create: `football/extra_processor.py`
- Create: `tests/fixtures/new_COL_sample.csv`
- Modify: `football/downloader.py` (add `--extra`, `EXTRA_LEAGUES`, `download_extra`)
- Test: `tests/test_football_extra_processor.py`
- Modify docs: `docs/data-pipeline.md`

**Interfaces:**
- Consumes: `football.common` (`MATCH_COLUMNS`, `ODDS_COLUMNS`, `outcome_from_goals`); `football.processor` (`MatchFormatError`, `_parse_dates`, `check_match_format`).
- Produces:
  - `EXTRA_ODDS_SOURCES` — tuple of `(name, (home, draw, away))`, all opening.
  - `preprocess_extra(df, league=None, validate=True) -> pd.DataFrame` — same shape as `preprocess_matches`, `attrs["odds_source"]` a `extra_*_opening` name, `attrs["odds_are_closing"] == False`.
  - `load_extra(path, league=None, validate=True) -> pd.DataFrame`
  - `available_leagues(path) -> list[str]`
  - In `downloader.py`: `EXTRA_LEAGUES` dict (`{"COL": "Colombia — Primera A", ...}`), `EXTRA_URL = "https://www.football-data.co.uk/new/{code}.csv"`, `download_extra(codes, out_dir=DEFAULT_DATA_DIR, dry_run=False, force=False) -> list[dict]`.

- [ ] **Step 1: Create the fixture**

`tests/fixtures/new_COL_sample.csv` (header + a few rows in football-data's
"extra" shape; two leagues so the filter is exercised):

```csv
Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PH,PD,PA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA
Colombia,Colombia Primera A,2023,04/02/2023,00:00,Millonarios,Nacional,2,1,H,2.35,3.10,3.20,2.45,3.25,3.40,2.30,3.05,3.10
Colombia,Colombia Primera A,2023,05/02/2023,00:00,Junior,America de Cali,0,0,D,2.10,3.20,3.80,2.20,3.35,3.95,2.05,3.15,3.70
Colombia,Colombia Primera A,2023,11/02/2023,00:00,Tolima,Millonarios,1,2,A,2.55,3.05,2.95,2.65,3.15,3.10,2.50,3.00,2.85
Colombia,Colombia Primera B,2023,04/02/2023,00:00,Cucuta,Real Cartagena,3,1,H,1.90,3.30,4.20,2.00,3.45,4.40,1.85,3.25,4.10
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_football_extra_processor.py
"""The football-data.co.uk "extra" contract — football/extra_processor.py.

These files (new/COL.csv and friends) are a different shape from the main
league CSVs: Home/Away/HG/AG instead of HomeTeam/FTHG, several leagues and
seasons stacked in one file, and OPENING ODDS ONLY. This module maps them onto
the same tidy frame everything else consumes, and it must never let those
opening prices masquerade as a closing baseline.
"""

import os

import pytest

from football.extra_processor import (
    available_leagues,
    load_extra,
    preprocess_extra,
)
from football.processor import CLOSING_SOURCES, MatchFormatError

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "new_COL_sample.csv")


def test_maps_extra_columns_onto_the_tidy_shape():
    matches = load_extra(FIXTURE, league="Colombia Primera A")
    assert list(matches.columns[:6]) == ["ds", "home_team", "away_team",
                                         "home_goals", "away_goals", "outcome"]
    assert (matches["home_team"].iloc[0], matches["away_team"].iloc[0]) == ("Millonarios", "Nacional")
    assert matches["outcome"].tolist() == ["H", "D", "A"]
    assert str(matches["ds"].iloc[0].date()) == "2023-02-04"


def test_odds_source_is_opening_and_never_closing():
    matches = load_extra(FIXTURE, league="Colombia Primera A")
    assert matches.attrs["odds_source"].endswith("_opening")
    assert matches.attrs["odds_source"] not in CLOSING_SOURCES
    assert matches.attrs["odds_are_closing"] is False


def test_league_filter_is_required_when_the_file_has_several():
    with pytest.raises(MatchFormatError):
        load_extra(FIXTURE, league=None)


def test_available_leagues_lists_what_is_in_the_file():
    assert set(available_leagues(FIXTURE)) == {"Colombia Primera A", "Colombia Primera B"}


def test_missing_core_columns_raise():
    import pandas as pd
    with pytest.raises(MatchFormatError):
        preprocess_extra(pd.DataFrame({"Home": ["A"], "Away": ["B"]}))
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_football_extra_processor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'football.extra_processor'`

- [ ] **Step 4: Write `football/extra_processor.py`**

```python
# football/extra_processor.py
"""The football-data.co.uk "extra" file contract: new/COL.csv and its siblings.

Their main league files carry HomeTeam/AwayTeam/FTHG/FTAG, one league per file,
and — from 2019/20 — closing odds. The "extra" files for the rest of the world
are a different animal: Home/Away/HG/AG, many leagues and seasons stacked in
one download, and OPENING ODDS ONLY (AvgH/D/A market average, PH/D/A Pinnacle,
sometimes B365H/D/A). `football/processor.py` refuses them; this module reads
them onto the same tidy frame, with one rule enforced hard: the odds source is
always an opening one, so `odds_are_closing` is always False and no evaluation
built on these files can claim a corrected edge.

One league per load. The file physically contains several, and stacking (say)
Colombia's Primera A and Primera B is the same mistake as merging opening and
closing odds — a `league` argument is required whenever the file has more than
one.
"""

import os
import warnings

import numpy as np
import pandas as pd

from football.common import ODDS_COLUMNS, outcome_from_goals
from football.processor import MatchFormatError, _parse_dates, check_match_format

REQUIRED_COLUMNS = ("Date", "Home", "Away", "HG", "AG")

# All opening. Best first: market average, then Pinnacle, then Bet365.
EXTRA_ODDS_SOURCES = (
    ("extra_market_average_opening", ("AvgH", "AvgD", "AvgA")),
    ("extra_pinnacle_opening", ("PH", "PD", "PA")),
    ("extra_bet365_opening", ("B365H", "B365D", "B365A")),
)


def available_leagues(path):
    """The distinct `League` values in an extra file, for a UI selector."""
    frame = pd.read_csv(path, usecols=["League"])
    return sorted(frame["League"].dropna().unique().tolist())


def _resolve_extra_source(columns):
    available = set(columns)
    for name, triple in EXTRA_ODDS_SOURCES:
        if available.issuperset(triple):
            return name, triple
    return None, None


def preprocess_extra(df, league=None, validate=True):
    """Turn a raw extra-file frame into the tidy match shape, opening odds only."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MatchFormatError(
            f"Missing required columns for an extra file: {missing}. Expected "
            f"{list(REQUIRED_COLUMNS)} — this is not a football-data.co.uk new/ file."
        )

    if "League" in df.columns:
        leagues = df["League"].dropna().unique().tolist()
        if len(leagues) > 1 and league is None:
            raise MatchFormatError(
                f"This file carries {len(leagues)} leagues ({leagues}). Pass league= to pick "
                "one — stacking two would put two competitions in one frame, the same mistake "
                "as mixing two odds sources."
            )
        if league is not None:
            df = df[df["League"] == league]
            if df.empty:
                raise MatchFormatError(f"No rows for league {league!r}. Available: {leagues}.")

    out = pd.DataFrame({
        "ds": _parse_dates(df["Date"]),
        "home_team": df["Home"].astype("string").str.strip(),
        "away_team": df["Away"].astype("string").str.strip(),
        "home_goals": pd.to_numeric(df["HG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["AG"], errors="coerce"),
    })

    unplayed = out["home_goals"].isna() | out["away_goals"].isna()
    if unplayed.any():
        out = out[~unplayed]
        df = df[~unplayed]

    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    out["outcome"] = [outcome_from_goals(h, a)
                      for h, a in zip(out["home_goals"], out["away_goals"])]

    name, triple = _resolve_extra_source(df.columns)
    if name is None:
        for column in ODDS_COLUMNS:
            out[column] = np.nan
    else:
        for column, raw in zip(ODDS_COLUMNS, triple):
            out[column] = pd.to_numeric(df[raw], errors="coerce")
        prices = out[list(ODDS_COLUMNS)]
        unusable = prices.isna().any(axis=1) | (prices <= 1.0).any(axis=1)
        out.loc[unusable, list(ODDS_COLUMNS)] = np.nan

    out = out.sort_values("ds").reset_index(drop=True)
    out.attrs["odds_source"] = name
    out.attrs["odds_are_closing"] = False  # extra files never carry closing odds

    if validate:
        report = check_match_format(out)
        if report:
            warnings.warn(report["message"], stacklevel=2)
    return out


def load_extra(path, league=None, validate=True):
    """Read one extra CSV into the tidy match shape (opening odds only)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    return preprocess_extra(pd.read_csv(path), league=league, validate=validate)
```

- [ ] **Step 5: Run the parser tests to verify they pass**

Run: `python -m pytest tests/test_football_extra_processor.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Add `--extra` to `football/downloader.py`**

Read `football/downloader.py` first. Add near `LEAGUES`:

```python
EXTRA_URL = "https://www.football-data.co.uk/new/{code}.csv"

# The "extra" files (new/COL.csv, ...). Different contract, opening odds only,
# read by football/extra_processor.py rather than football/processor.py.
EXTRA_LEAGUES = {
    "COL": "Colombia — Primera A / Primera B",
    "ARG": "Argentina — Liga Profesional",
    "BRA": "Brazil — Serie A",
    "MEX": "Mexico — Liga MX",
    "USA": "USA — MLS",
}
```

Add a `download_extra` function mirroring `download_seasons` (HTML-sniff, no
shrink without `--force`, validate through `preprocess_extra` with
`league=None, validate=True` so the multi-league frame is only *checked*, not
filtered, at download time; write the file verbatim to
`season_path`-style `os.path.join(out_dir, f"{code}.csv")`):

```python
def download_extra(codes, out_dir=DEFAULT_DATA_DIR, dry_run=False, force=False):
    """Fetch football-data.co.uk new/{code}.csv files. Opening odds only."""
    from football.extra_processor import preprocess_extra  # lazy, like fetch_csv

    os.makedirs(out_dir, exist_ok=True)
    session = _make_session()
    summary = []
    for code in codes:
        if code not in EXTRA_LEAGUES:
            raise ValueError(
                f"Unknown extra code {code!r}. Known: {', '.join(sorted(EXTRA_LEAGUES))}."
            )
        url = EXTRA_URL.format(code=code)
        text = _fetch_text(url, session=session)          # reuse the HTML sniff in fetch_csv
        frame = pd.read_csv(io.StringIO(text))
        preprocess_extra(frame, league=None, validate=True)  # raises on a broken contract
        path = os.path.join(out_dir, f"{code}.csv")
        note = _write_unless_shorter(path, text, force=force, dry_run=dry_run)
        summary.append({"code": code, "name": EXTRA_LEAGUES[code], "path": path,
                        "rows": len(frame), "note": note, "odds": "opening only"})
    return summary
```

If `fetch_csv` is not already factored into a reusable text fetch + an HTML
sniff, extract that part into `_fetch_text(url, session)` and have `fetch_csv`
call it — do not duplicate the sniff. Same for the "don't replace a longer file
with a shorter one" logic (`_write_unless_shorter`).

In `parse_leagues`, when the new `--extra` flag is set, validate against
`EXTRA_LEAGUES` instead of `LEAGUES`. Wire the flag in `main`:

```python
parser.add_argument("--extra", action="store_true",
                    help="fetch the new/ 'extra' files (COL, ARG, ...) instead of league files; "
                         "opening odds only, one league per competition stacked per file")
```

and branch `main` to `download_extra` when `args.extra` is set (ignore
`--seasons` in that path — extra files are not per-season; print a note if it
was given).

- [ ] **Step 7: Add a downloader test**

Append to `tests/test_football_downloader.py`:

```python
def test_extra_code_rejected_without_the_flag():
    from football.downloader import parse_leagues
    with pytest.raises(ValueError):
        parse_leagues("COL")  # extra flag not set


def test_extra_codes_parse_with_the_flag():
    from football.downloader import parse_leagues
    assert parse_leagues("COL,ARG", extra=True) == ["COL", "ARG"]
```

(Adjust `parse_leagues` to take an `extra=False` kwarg.)

- [ ] **Step 8: Run the downloader tests**

Run: `python -m pytest tests/test_football_downloader.py -q`
Expected: PASS

- [ ] **Step 9: Update `docs/data-pipeline.md`**

Add a subsection under the football data notes: the extra-file contract
(`Home/Away/HG/AG`, `League`/`Season` columns, opening odds `AvgH/PH/B365H`),
that `football/extra_processor.py` owns it, that `--extra` fetches it, one
league per load, `odds_are_closing` always False. State the limit: opening
odds mean the market baseline is soft and no corrected edge claim is possible
on Colombian data.

- [ ] **Step 10: Commit**

```bash
git add football/extra_processor.py football/downloader.py \
        tests/test_football_extra_processor.py tests/test_football_downloader.py \
        tests/fixtures/new_COL_sample.csv docs/data-pipeline.md
git commit -m "Add the football-data.co.uk 'extra' contract and downloader --extra

Colombia (new/COL.csv) and its siblings: Home/Away/HG/AG, many leagues per
file, opening odds only. football/extra_processor.py maps them onto the tidy
match frame with odds_are_closing pinned False.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT"
```

---

## Task 5: `football/evaluation.py` — the paired market comparison

**Files:**
- Create: `football/evaluation.py`
- Test: `tests/test_football_evaluation.py`

**Interfaces:**
- Consumes: `core.significance` (`z_test_against_null`, `bonferroni_threshold`, `verdicts`); `football.scoring.per_match_scores`.
- Produces:
  - `beats_market_test(model_probs, market_probs, outcomes, metric="rps", alpha=0.05, n_comparisons=1) -> dict`
    with keys: `metric, n_observations, model_score, market_score, skill_score,
    z, p_value, p_value_greater, effect, ci_low, ci_high, relative_effect,
    observed_mean, null_mean, beats_market, beats_market_corrected,
    bonferroni_threshold`.
  - The verdict keys are renamed from `core.significance.verdicts`'
    `beats_chance` / `beats_chance_corrected` to `beats_market` /
    `beats_market_corrected` (the domain rename, like the lottery's
    `null_mean` → `chance_mean`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_football_evaluation.py
"""Does a football model beat the market? — football/evaluation.py.

The football analogue of lottery/models/baseline.py:beats_chance_test. It forms
the per-match difference between the market's proper score and the model's,
and tests whether that mean improvement is greater than zero — one-sided,
Bonferroni-aware, effect size attached.
"""

import numpy as np
import pytest

from football.evaluation import beats_market_test


def _outcomes(rng, n):
    return rng.choice(["H", "D", "A"], size=n)


def test_model_equal_to_market_shows_no_edge():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([3, 2, 2], size=300)
    outcomes = [["H", "D", "A"][i] for i in [np.argmax(p) for p in probs]]
    result = beats_market_test(probs, probs.copy(), outcomes)
    assert result["effect"] == pytest.approx(0.0, abs=1e-9)
    assert result["beats_market"] is False
    assert result["beats_market_corrected"] is False


def test_strictly_better_model_beats_the_market():
    rng = np.random.default_rng(1)
    n = 400
    outcomes = _outcomes(rng, n)
    truth = np.zeros((n, 3))
    truth[np.arange(n), [{"H": 0, "D": 1, "A": 2}[o] for o in outcomes]] = 1.0
    # Model: 80% toward the truth. Market: 55% toward the truth. Model must win.
    uniform = np.full((n, 3), 1 / 3)
    model = 0.8 * truth + 0.2 * uniform
    market = 0.55 * truth + 0.45 * uniform
    result = beats_market_test(model, market, outcomes, metric="rps")
    assert result["skill_score"] > 0
    assert result["p_value_greater"] < 0.01
    assert result["beats_market"] is True
    assert result["beats_market_corrected"] is True


def test_both_verdict_keys_always_present_even_on_empty_input():
    result = beats_market_test(np.empty((0, 3)), np.empty((0, 3)), [])
    assert "beats_market" in result and "beats_market_corrected" in result
    assert result["n_observations"] == 0


def test_more_comparisons_tighten_the_corrected_threshold():
    rng = np.random.default_rng(2)
    n = 300
    outcomes = _outcomes(rng, n)
    truth = np.zeros((n, 3))
    truth[np.arange(n), [{"H": 0, "D": 1, "A": 2}[o] for o in outcomes]] = 1.0
    uniform = np.full((n, 3), 1 / 3)
    model = 0.62 * truth + 0.38 * uniform
    market = 0.55 * truth + 0.45 * uniform
    one = beats_market_test(model, market, outcomes, n_comparisons=1)
    many = beats_market_test(model, market, outcomes, n_comparisons=10)
    assert many["bonferroni_threshold"] < one["bonferroni_threshold"]


def test_nan_rows_are_dropped_pairwise():
    model = np.array([[0.7, 0.2, 0.1], [np.nan, np.nan, np.nan], [0.3, 0.3, 0.4]])
    market = np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3], [np.nan, np.nan, np.nan]])
    result = beats_market_test(model, market, ["H", "D", "A"])
    assert result["n_observations"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_football_evaluation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'football.evaluation'`

- [ ] **Step 3: Write the implementation**

```python
# football/evaluation.py
"""Whether a football model beats the market — the domain half of the core contract.

`core/significance.py` supplies the arithmetic: a one-sided z-test of a sum of
independent per-observation scores against a null, with the effect size and its
interval, plus the naive and Bonferroni-corrected verdicts. What this module
supplies is the null.

In the lottery the null is the exact hypergeometric hit rate. In football it is
the market: for each held-out match, score the model's probability vector and
the market's with the same proper scoring rule, take the per-match difference
(market score minus model score, so positive means the model did better), and
test whether its mean is greater than zero. That is a paired comparison —
every match is scored by both — which is why the null mean is exactly 0 and the
null variance is the sample variance of the differences.

Only the one-sided p-value may back a "beats the market" claim: a model
significantly *worse* than the market also gets a small two-sided p-value.
"""

import numpy as np

from core.significance import bonferroni_threshold, verdicts, z_test_against_null
from football.scoring import per_match_scores

_EMPTY = {
    "n_observations": 0, "model_score": float("nan"), "market_score": float("nan"),
    "skill_score": float("nan"), "z": float("nan"), "p_value": float("nan"),
    "p_value_greater": float("nan"), "effect": float("nan"), "ci_low": float("nan"),
    "ci_high": float("nan"), "relative_effect": float("nan"),
    "observed_mean": float("nan"), "null_mean": 0.0,
}


def beats_market_test(model_probs, market_probs, outcomes,
                      metric="rps", alpha=0.05, n_comparisons=1):
    """Paired proper-score test of a model against the market.

    `model_probs` and `market_probs` are `(n, 3)` in OUTCOMES order; `outcomes`
    is the actual results. Rows where either side is NaN are dropped from both.
    """
    model_probs = np.asarray(model_probs, dtype=float).reshape(-1, 3)
    market_probs = np.asarray(market_probs, dtype=float).reshape(-1, 3)
    outcomes = np.asarray(list(outcomes))
    threshold = bonferroni_threshold(alpha, n_comparisons)

    keep = ~(np.isnan(model_probs).any(axis=1) | np.isnan(market_probs).any(axis=1))
    model_probs, market_probs, outcomes = model_probs[keep], market_probs[keep], outcomes[keep]

    if len(outcomes) == 0:
        return {"metric": metric, "bonferroni_threshold": threshold,
                "beats_market": False, "beats_market_corrected": False, **_EMPTY}

    model_s = per_match_scores(model_probs, outcomes, metric)
    market_s = per_match_scores(market_probs, outcomes, metric)
    diff = market_s - model_s  # > 0 => model scored lower (better) on that match

    variance = float(np.var(diff, ddof=1)) if len(diff) > 1 else 0.0
    result = z_test_against_null(diff, null_means=0.0, null_variances=variance)

    v = verdicts(result["p_value_greater"], alpha, threshold)
    mean_model, mean_market = float(model_s.mean()), float(market_s.mean())
    return {
        "metric": metric,
        "model_score": mean_model,
        "market_score": mean_market,
        "skill_score": (1.0 - mean_model / mean_market) if mean_market else float("nan"),
        "bonferroni_threshold": threshold,
        "beats_market": v["beats_chance"],
        "beats_market_corrected": v["beats_chance_corrected"],
        **{k: val for k, val in result.items() if k != "null_mean"},
        "null_mean": 0.0,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_football_evaluation.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add football/evaluation.py tests/test_football_evaluation.py
git commit -m "Add football/evaluation.py: paired proper-score test vs the market

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT"
```

---

## Task 6: `football/backtest.py` — walk-forward evaluation

**Files:**
- Create: `football/backtest.py`
- Test: `tests/test_football_backtest.py` (`@pytest.mark.slow`)

**Interfaces:**
- Consumes: `core.windows` (`window_bounds`, `cutoff_bounds`); `football.dixon_coles.DixonColes` / `UnknownTeamError`; `football.market.market_probabilities`; `football.evaluation.beats_market_test`; `football.common` (`OUTCOMES`, `ODDS_COLUMNS`).
- Produces:
  - `run_all(matches, n_windows=30, min_train=100, half_life=None, method="multiplicative", metric="rps") -> dict`
  - `run_holdout(matches, cutoff, mode="expanding", half_life=None, method="multiplicative", metric="rps") -> dict`
  - Both return the `beats_market_test` dict plus `n_windows_scored, n_windows_skipped, mode, method, half_life`.
  - `main()` — CLI: `python -m football.backtest --seasons a.csv,b.csv [--n-windows 30] [--cutoff 2024-03-01 --mode frozen] [--half-life 180] [--method power] [--data-dir ...] [--extra --league "Colombia Primera A"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_football_backtest.py
"""Walk-forward: does Dixon-Coles beat the market? — football/backtest.py.

Mirrors lottery/backtest.py. The two checks that matter are a positive and a
negative control: on a SOFT simulated market (market_noise high) a real model
should beat it; on a SHARP one (market_noise = 0, the book prices the truth)
it should not. A backtest that fires on the sharp market is measuring its own
wiring, not an edge.
"""

import pytest

from football.backtest import run_all, run_holdout
from football.sample_data import generate_matches
from football.processor import preprocess_matches

pytestmark = pytest.mark.slow


def _matches(market_noise, seed=0):
    raw = generate_matches(n_teams=12, seed=seed, market_noise=market_noise).drop(
        columns=["TrueH", "TrueD", "TrueA"])
    return preprocess_matches(raw, validate=False)


def test_beats_a_soft_market():
    result = run_all(_matches(market_noise=0.6), n_windows=40, min_train=120)
    assert result["skill_score"] > 0
    assert result["beats_market"] is True


def test_does_not_beat_a_sharp_market():
    result = run_all(_matches(market_noise=0.0), n_windows=40, min_train=120)
    assert result["beats_market_corrected"] is False


def test_frozen_and_expanding_have_the_same_keys():
    matches = _matches(market_noise=0.3)
    cutoff = matches["ds"].quantile(0.7)
    frozen = run_holdout(matches, cutoff=cutoff, mode="frozen")
    expanding = run_holdout(matches, cutoff=cutoff, mode="expanding")
    assert set(frozen) == set(expanding)
    assert frozen["mode"] == "frozen" and expanding["mode"] == "expanding"


def test_effect_size_and_interval_are_reported():
    result = run_all(_matches(market_noise=0.3), n_windows=30, min_train=120)
    assert "effect" in result and "ci_low" in result and "ci_high" in result
    assert result["n_windows_scored"] > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_football_backtest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'football.backtest'`

- [ ] **Step 3: Write the implementation**

```python
# football/backtest.py
"""Walk-forward evaluation of a football model against the closing line.

The football counterpart of lottery/backtest.py, and it shares the shape on
purpose. `run_all` holds out the last N matches and refits Dixon-Coles before
each one (expanding window, one step ahead). `run_holdout` holds out everything
after a date, either refitting per match (`expanding`) or fitting once at the
cutoff (`frozen`) — the frozen mode is the fast, concrete "train in March,
predict the rest of the season" run.

Both paths end in the same place: `football.evaluation.beats_market_test` over
every (model probs, market probs, outcome) triple collected, so the summaries
are directly comparable. A window whose held-out fixture involves a team the
training slice never saw is skipped, never scored — exactly as the lottery
skips a window a model could not predict.

Fitting Dixon-Coles per window is the slow part; `--n-windows` defaults low and
`--cutoff ... --mode frozen` avoids the refit loop entirely.
"""

import argparse

import numpy as np
import pandas as pd

from core.windows import cutoff_bounds, window_bounds
from football.common import ODDS_COLUMNS, PROBABILITY_COLUMNS
from football.dixon_coles import DixonColes, UnknownTeamError
from football.evaluation import beats_market_test
from football.market import market_probabilities

MIN_TRAIN = 100


def _market_row_probs(match_row, method):
    frame = pd.DataFrame([match_row])
    if frame[list(ODDS_COLUMNS)].isna().any(axis=None):
        return np.array([np.nan, np.nan, np.nan])
    out = market_probabilities(frame, method=method)
    return out[list(PROBABILITY_COLUMNS)].to_numpy()[0]


def _score(model_probs, market_probs, outcomes, metric, n_comparisons=1):
    result = beats_market_test(np.array(model_probs), np.array(market_probs),
                               outcomes, metric=metric, n_comparisons=n_comparisons)
    return result


def run_all(matches, n_windows=30, min_train=MIN_TRAIN, half_life=None,
            method="multiplicative", metric="rps"):
    """Hold out the last `n_windows` matches, refitting before each."""
    matches = matches.sort_values("ds").reset_index(drop=True)
    start, total = window_bounds(len(matches), n_windows, min_train)
    model_probs, market_probs, outcomes = [], [], []
    skipped = 0
    for t in range(start, total):
        train = matches.iloc[:t]
        test = matches.iloc[t]
        try:
            model = DixonColes.fit(train, half_life=half_life)
            p_model = model.predict_outcome(test["home_team"], test["away_team"])
        except UnknownTeamError:
            skipped += 1
            continue
        model_probs.append(p_model)
        market_probs.append(_market_row_probs(test, method))
        outcomes.append(test["outcome"])

    result = _score(model_probs, market_probs, outcomes, metric)
    result.update({"n_windows_scored": len(outcomes), "n_windows_skipped": skipped,
                   "mode": "expanding_last_n", "method": method, "half_life": half_life})
    return result


def run_holdout(matches, cutoff, mode="expanding", half_life=None,
                method="multiplicative", metric="rps"):
    """Hold out every match after `cutoff`. mode: 'expanding' (refit each) or 'frozen' (fit once)."""
    matches = matches.sort_values("ds").reset_index(drop=True)
    n_train, n_holdout = cutoff_bounds(matches["ds"], cutoff)
    model_probs, market_probs, outcomes = [], [], []
    skipped = 0

    frozen_model = None
    if mode == "frozen":
        frozen_model = DixonColes.fit(matches.iloc[:n_train], half_life=half_life)

    for offset in range(n_holdout):
        t = n_train + offset
        test = matches.iloc[t]
        try:
            model = frozen_model or DixonColes.fit(matches.iloc[:t], half_life=half_life)
            p_model = model.predict_outcome(test["home_team"], test["away_team"])
        except UnknownTeamError:
            skipped += 1
            continue
        model_probs.append(p_model)
        market_probs.append(_market_row_probs(test, method))
        outcomes.append(test["outcome"])

    result = _score(model_probs, market_probs, outcomes, metric)
    result.update({"n_windows_scored": len(outcomes), "n_windows_skipped": skipped,
                   "mode": mode, "method": method, "half_life": half_life})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", required=True,
                        help="comma-separated CSV filenames inside --data-dir")
    parser.add_argument("--data-dir", default="exported_data/football")
    parser.add_argument("--n-windows", type=int, default=30)
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN)
    parser.add_argument("--cutoff", default=None, help="ISO date; hold out everything after it")
    parser.add_argument("--mode", choices=("expanding", "frozen"), default="expanding")
    parser.add_argument("--half-life", type=float, default=None, help="days; time-decay weighting")
    parser.add_argument("--method", choices=("multiplicative", "additive", "power"),
                        default="multiplicative")
    parser.add_argument("--metric", choices=("brier", "rps", "log_loss"), default="rps")
    parser.add_argument("--extra", action="store_true", help="load via the extra-file contract")
    parser.add_argument("--league", default=None, help="league to pick from an --extra file")
    args = parser.parse_args()

    import os
    paths = [os.path.join(args.data_dir, name) for name in args.seasons.split(",")]
    if args.extra:
        from football.extra_processor import load_extra
        frames = [load_extra(p, league=args.league) for p in paths]
        matches = pd.concat(frames, ignore_index=True).sort_values("ds").reset_index(drop=True)
    else:
        from football.processor import load_seasons
        matches = load_seasons(paths)

    if args.cutoff:
        result = run_holdout(matches, cutoff=args.cutoff, mode=args.mode,
                             half_life=args.half_life, method=args.method, metric=args.metric)
    else:
        result = run_all(matches, n_windows=args.n_windows, min_train=args.min_train,
                         half_life=args.half_life, method=args.method, metric=args.metric)

    for key, value in result.items():
        print(f"{key:>24}: {value}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_football_backtest.py -q`
Expected: PASS (4 passed). These are slow (real fits); allow a minute or two.
If `test_does_not_beat_a_sharp_market` is flaky, raise `min_train` / lower
`n_windows` so each fit has more data — do not weaken the assertion.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add football/backtest.py tests/test_football_backtest.py
git commit -m "Add football/backtest.py: walk-forward Dixon-Coles vs the market

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT"
```

---

## Task 7: Dashboard — source toggle and the Pronóstico tab

**Files:**
- Modify: `dashboard/football_page.py`
- Modify: `dashboard/ui.py` (add `fb_*` HELP keys)
- Test: extend `tests/test_dashboard_help.py` if it enforces key coverage

**Interfaces:**
- Consumes: `football.h2h` (`team_form`, `head_to_head`), `football.dixon_coles` (`DixonColes`, `UnknownTeamError`), `football.extra_processor` (`load_extra`, `available_leagues`), `football.market` (`implied_probabilities`), `football.common` (`OUTCOMES`, `PROBABILITY_COLUMNS`).
- Produces: a new `render_forecast_tab(matches, method)` helper in `football_page.py`; the sidebar source toggle; four tabs instead of three.

- [ ] **Step 1: Read the current file and `dashboard/ui.py`**

Read `dashboard/football_page.py` and `dashboard/ui.py` in full. Note
`HELP`, `section`, `chart`, and how `load_matches` is cached.

- [ ] **Step 2: Add the HELP keys to `dashboard/ui.py`**

Add to the `HELP` dict (Spanish, each stating what the number does *not* mean):

```python
    "fb_source_toggle": (
        "Europa usa los archivos de liga de football-data.co.uk (cuotas de cierre desde "
        "2019/20). Colombia usa el archivo «extra» new/COL.csv: mismo deporte, pero solo "
        "cuotas de apertura — la línea base es blanda y ninguna ventaja medida ahí está probada."
    ),
    "fb_forecast_tab": (
        "Un pronóstico para un partido concreto. Las probabilidades del modelo se muestran "
        "siempre junto a las del mercado; el modelo por sí solo no dice si acierta — eso lo "
        "responde la pestaña Resultados, sobre muchos partidos y con corrección."
    ),
    "fb_h2h": (
        "Historial y forma reciente. Es descriptivo: no es una predicción, y una racha corta "
        "es en su mayor parte ruido."
    ),
    "fb_form": "Últimos partidos de cada equipo antes de esta fecha. W/E/D desde la óptica del equipo.",
    "fb_model_1x2": (
        "Probabilidad de local / empate / visitante según Dixon-Coles ajustado a las temporadas "
        "cargadas. No incorpora lesiones, alineaciones ni el mercado."
    ),
    "fb_scoreline_grid": (
        "Probabilidad de cada marcador exacto. El más probable rara vez pasa del 10-12%: sirve "
        "para ver la forma de la distribución, no para apostar a un resultado exacto."
    ),
    "fb_most_likely_scores": "Los marcadores con más probabilidad. Suman una fracción pequeña del total.",
    "fb_over_under": "Probabilidad de más/menos de 2.5 goles, derivada de la misma matriz de marcadores.",
    "fb_btts": "Probabilidad de que ambos equipos marquen.",
    "fb_your_odds": (
        "Cuotas decimales actuales de una casa de apuestas para este partido. Si las pones, se "
        "les quita el margen y se comparan con el modelo. Es un partido y una comparación sin "
        "corregir: no es un veredicto."
    ),
    "fb_model_vs_market": (
        "Modelo contra mercado para este partido. Una diferencia a favor del modelo en un "
        "partido no significa nada — hace falta la evaluación de la pestaña Resultados."
    ),
    "fb_half_life": (
        "Vida media en días del peso temporal: un partido de hace tantos días pesa la mitad. "
        "Más bajo = más peso a la forma reciente. 0 = todos los partidos pesan igual."
    ),
    "fb_eval_tab": (
        "Puntuación fuera de muestra del modelo contra la cuota de cierre, con el veredicto "
        "naive y el corregido. Es el único número de esta página que dice si el modelo vale algo."
    ),
    "fb_skill_score": (
        "1 − score(modelo)/score(mercado). Positivo = el modelo puntuó mejor. En una sola "
        "temporada un valor positivo pequeño está dentro del ruido."
    ),
    "fb_beats_market": (
        "Veredicto de una prueba pareada de una cola. Mira siempre la columna corregida: con "
        "varios métodos de de-margen probados a la vez, la naive se supera por azar."
    ),
    "fb_model_calibration": (
        "Cuando el modelo dice 60% de victoria local, ¿gana el local ~60% de las veces? "
        "Con una temporada cada punto tiene pocos partidos, así que la dispersión es ruido."
    ),
```

- [ ] **Step 3: Add the sidebar source toggle and Colombia loading**

In `render()`, before the existing uploader block, add:

```python
    with st.sidebar:
        st.header("Datos")
        source_kind = st.radio(
            "Fuente", ["Europa (football-data)", "Colombia (archivo extra)"],
            help=HELP["fb_source_toggle"],
        )
```

Branch the load: for "Colombia", show a text input for the `new/COL.csv` path
(default `os.path.join(DEFAULT_DATA_DIR, "COL.csv")`), a league `st.selectbox`
populated from `available_leagues(path)` (guarded with a try/except that falls
to the demo data with a message if the file is absent), and call
`load_extra(path, league=chosen_league)` inside a cached wrapper
`load_extra_cached(path, league)`. Keep the European path exactly as it is.

The existing opening-odds warning banner (`source not in CLOSING_SOURCES`)
already covers Colombia — no new copy needed there.

- [ ] **Step 4: Add the Pronóstico tab**

Change `tabs = st.tabs([...])` to
`["Datos", "Mercado", "Pronóstico", "Resultados"]` and insert:

```python
    with tabs[2]:
        render_forecast_tab(matches, method)
```

Implement `render_forecast_tab` in the same module:

```python
def render_forecast_tab(matches, method):
    section("Pronóstico de un partido", "fb_forecast_tab")
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    if len(teams) < 2:
        st.warning("Hacen falta al menos dos equipos en los datos cargados.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    home_team = c1.selectbox("Local", teams, index=0)
    away_team = c2.selectbox("Visitante", teams, index=1)
    half_life = c3.number_input("Vida media (días)", min_value=0, value=180, step=30,
                                help=HELP["fb_half_life"])
    if home_team == away_team:
        st.warning("Elige dos equipos distintos.")
        return

    with st.form("odds_form"):
        st.markdown("**Cuotas actuales (opcional)**", help=HELP["fb_your_odds"])
        o1, o2, o3 = st.columns(3)
        odd_home = o1.number_input("Local", min_value=0.0, value=0.0, step=0.05)
        odd_draw = o2.number_input("Empate", min_value=0.0, value=0.0, step=0.05)
        odd_away = o3.number_input("Visitante", min_value=0.0, value=0.0, step=0.05)
        submitted = st.form_submit_button("Calcular pronóstico")

    if not submitted:
        st.info("Elige los equipos y pulsa **Calcular pronóstico**.")
        return

    # --- head to head ---
    section("Cómo llegan", "fb_h2h")
    h2h = head_to_head(matches, home_team, away_team)
    fc, ac = st.columns(2)
    for col, team in ((fc, home_team), (ac, away_team)):
        form = team_form(matches, team, last_n=5)
        col.metric(team, "".join({"W": "V", "D": "E", "L": "D"}[r] for r in form["results"]) or "—")
        col.caption(f"{form['wins']}V {form['draws']}E {form['losses']}D · "
                    f"{form['goals_for']}-{form['goals_against']} goles · {form['points']} pts")
    st.caption(
        f"{h2h['meetings']} enfrentamientos: {h2h['home_wins']} {home_team}, "
        f"{h2h['draws']} empates, {h2h['away_wins']} {away_team}. "
        f"Media de goles {h2h['avg_goals']:.2f}." if h2h["meetings"] else "Sin enfrentamientos previos."
    )

    # --- model ---
    try:
        model = _fit_dixon_coles(matches, int(half_life) or None)
        p_model = model.predict_outcome(home_team, away_team)
    except UnknownTeamError as exc:
        st.error(f"El modelo no conoce a ese equipo en las temporadas cargadas. Detalle: {exc}")
        return

    section("El modelo", "fb_model_1x2")
    m1, m2, m3 = st.columns(3)
    for col, label, value in zip((m1, m2, m3), ("Local", "Empate", "Visitante"), p_model):
        col.metric(label, f"{value:.1%}")

    grid = model.scoreline_matrix(home_team, away_team, max_goals=6)
    figure = go.Figure(go.Heatmap(z=grid, x=list(range(7)), y=list(range(7)), colorscale="Blues"))
    figure.update_layout(xaxis_title=f"Goles {away_team}", yaxis_title=f"Goles {home_team}", height=380)
    chart(figure, "Probabilidad de cada marcador", "fb_scoreline_grid")

    scores = model.most_likely_scores(home_team, away_team, n=5)
    st.dataframe(pd.DataFrame(
        [{"Marcador": f"{h}-{a}", "Probabilidad": f"{p:.1%}"} for (h, a), p in scores]),
        use_container_width=True, hide_index=True)

    p_over, p_under = model.over_under(home_team, away_team, 2.5)
    ou1, ou2, ou3 = st.columns(3)
    ou1.metric("Más de 2.5", f"{p_over:.1%}", help=HELP["fb_over_under"])
    ou2.metric("Menos de 2.5", f"{p_under:.1%}")
    ou3.metric("Ambos marcan", f"{model.both_teams_to_score(home_team, away_team):.1%}",
               help=HELP["fb_btts"])

    # --- market ---
    odds = (odd_home, odd_draw, odd_away)
    if all(o > 1.0 for o in odds):
        section("Modelo contra mercado", "fb_model_vs_market")
        p_market = implied_probabilities(np.array(odds), method=method)
        figure = go.Figure()
        figure.add_trace(go.Bar(x=["Local", "Empate", "Visitante"], y=p_model, name="Modelo"))
        figure.add_trace(go.Bar(x=["Local", "Empate", "Visitante"], y=p_market, name="Mercado"))
        figure.update_layout(barmode="group", yaxis_tickformat=".0%", height=340)
        chart(figure, "Probabilidades: modelo y mercado", "fb_model_vs_market")
        st.dataframe(pd.DataFrame({
            "Resultado": ["Local", "Empate", "Visitante"],
            "Modelo": [f"{p:.1%}" for p in p_model],
            "Mercado": [f"{p:.1%}" for p in p_market],
            "Modelo − Mercado": [f"{m - k:+.1%}" for m, k in zip(p_model, p_market)],
        }), use_container_width=True, hide_index=True)
        st.caption(
            "Un partido, comparación **sin corregir**. Si el modelo se desvía mucho del mercado "
            "aquí, lo interesante es por qué, no que tenga razón. El veredicto medido está en "
            "**Resultados**."
        )
    else:
        st.info(
            "Sin cuotas para este partido no hay línea base: el pronóstico de arriba es solo el "
            "modelo. Pega las tres cuotas decimales para compararlo con el mercado."
        )


@st.cache_resource(show_spinner="Ajustando el modelo…")
def _fit_dixon_coles(matches, half_life):
    from football.dixon_coles import DixonColes
    return DixonColes.fit(matches, half_life=half_life)
```

Add the imports at the top of `football_page.py`:
`from football.h2h import head_to_head, team_form`,
`from football.dixon_coles import UnknownTeamError`,
`from football.market import implied_probabilities`,
`from football.extra_processor import available_leagues, load_extra`,
and `import numpy as np` if not present.

`_fit_dixon_coles` is cached on `matches` — Streamlit hashes the frame; if it
complains about unhashable, wrap the call site to pass a tuple of the loaded
`paths` + `half_life` and load inside. Prefer the paths key.

- [ ] **Step 5: Byte-compile and smoke-check the page**

Run: `python -m py_compile dashboard/football_page.py dashboard/ui.py dashboard/app.py`
Expected: no output (success)

Run: `python -m pytest tests/test_dashboard_help.py -q`
Expected: PASS (if the test checks every `HELP` key is a non-empty string, the new keys pass; if it cross-checks keys used in pages, this proves the wiring)

- [ ] **Step 6: Manual Streamlit + Playwright check**

Follow `CLAUDE.md`'s dashboard verification. Headless Streamlit, load the page,
pick **⚽ Fútbol**, open the **Pronóstico** tab, select two teams, submit the
form with and without odds, and confirm no `Traceback` / "This app has
encountered an error" in the tab panel. Scope locators to
`get_by_role("tabpanel", name="Pronóstico")`.

- [ ] **Step 7: Commit**

```bash
git add dashboard/football_page.py dashboard/ui.py tests/test_dashboard_help.py
git commit -m "Dashboard: Pronóstico tab and the Europa/Colombia source toggle

Two-team forecast: head-to-head + Dixon-Coles 1X2 and scoreline grid, with
optional current odds shown de-margined beside the model.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT"
```

---

## Task 8: Dashboard — the Resultados evaluation section

**Files:**
- Modify: `dashboard/football_page.py` (extend `tabs[3]` / Resultados)

**Interfaces:**
- Consumes: `football.backtest` (`run_all`, `run_holdout`), `football.dixon_coles.DixonColes`, `core.windows.window_bounds`.
- Produces: an evaluation section inside the Resultados tab, gated behind a button, plus a model-calibration curve beside the existing market one.

- [ ] **Step 1: Add the evaluation section**

Inside `with tabs[3]:` after the existing observed-vs-market content, add:

```python
        section("¿Le gana este modelo al mercado?", "fb_eval_tab")
        if source is None:
            st.warning("Sin cuotas no hay contra qué medir el modelo.")
        elif len(matches) < 150:
            st.info("Hacen falta ~150 partidos para un backtest con sentido; carga más temporadas.")
        else:
            if source not in CLOSING_SOURCES:
                st.warning(
                    "La línea base de estos datos es de **apertura**. Cualquier ventaja que "
                    "aparezca aquí es contra un mercado blando y no es prueba de una ventaja real."
                )
            n_windows = st.slider("Partidos a evaluar (walk-forward)", 20, 200, 40, step=10)
            half_life = st.number_input("Vida media (días), 0 = sin decaimiento",
                                        min_value=0, value=180, step=30, help=HELP["fb_half_life"])
            if st.button("Correr backtest"):
                with st.spinner("Reajustando el modelo por ventana…"):
                    result = _run_backtest(tuple(paths) if not uploaded else ("__upload__",),
                                           matches, n_windows, int(half_life) or None, method)
                v1, v2, v3 = st.columns(3)
                v1.metric("Skill score (RPS)", f"{result['skill_score']:+.3f}",
                          help=HELP["fb_skill_score"])
                v2.metric("Efecto", f"{result['effect']:+.4f}",
                          help=f"IC 95%: [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]")
                v3.metric("Partidos", result["n_windows_scored"])
                b1, b2 = st.columns(2)
                b1.metric("Supera al mercado (naive)", "Sí" if result["beats_market"] else "No",
                          help=HELP["fb_beats_market"])
                b2.metric("Supera al mercado (corregido)",
                          "Sí" if result["beats_market_corrected"] else "No")
                st.dataframe(pd.DataFrame({
                    "Métrica": ["RPS", "Brier", "log-loss"],
                    "Modelo": [f"{result.get('model_score', float('nan')):.4f}", "—", "—"],
                    "Mercado": [f"{result.get('market_score', float('nan')):.4f}", "—", "—"],
                }), use_container_width=True, hide_index=True)
                st.caption(
                    "Mira la columna **corregida**. Un skill score positivo con IC que cruza 0 "
                    "no es una ventaja: es ruido con el signo favorable."
                )
```

```python
@st.cache_data(show_spinner=False)
def _run_backtest(cache_key, matches, n_windows, half_life, method):
    from football.backtest import run_all
    return run_all(matches, n_windows=n_windows, min_train=max(100, len(matches) - n_windows - 1),
                   half_life=half_life, method=method)
```

(`cache_key` makes the cache vary with the loaded files; `matches` is passed
for the actual run. If Streamlit cannot hash `matches`, prefix the arg with a
leading underscore — `_matches` — which tells Streamlit to skip hashing it, and
rely on `cache_key` + `n_windows` + `half_life` + `method` for identity.)

- [ ] **Step 2: Add the model calibration curve**

In the Mercado tab, right after the existing market calibration chart, fit the
model on the loaded frame (reuse `_fit_dixon_coles`), compute `p_home` per
match with `predict_matches`, and draw the same calibration curve for the model
beside the market's. Title: "Calibración del modelo (victoria local)", key
`fb_model_calibration`. Guard it for `len(matches) >= 150` and wrap the fit in
try/except `UnknownTeamError` is not a risk here (all teams are in-sample) but
a non-convergence warning may print — that is fine.

- [ ] **Step 3: Byte-compile and smoke-check**

Run: `python -m py_compile dashboard/football_page.py`
Expected: success

Run the Streamlit + Playwright check again, this time opening **Resultados**,
clicking **Correr backtest**, and confirming the verdict metrics render with no
error in the tab panel. This is slow (real fits) — allow time.

- [ ] **Step 4: Commit**

```bash
git add dashboard/football_page.py
git commit -m "Dashboard: measured model-vs-market verdict in the Resultados tab

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT"
```

---

## Task 9: Documentation sweep, `CLAUDE.md`, final verification

**Files:**
- Modify: `docs/football.md`, `docs/models.md`, `docs/evaluation.md`, `docs/dashboard.md`, `CLAUDE.md`
- (`docs/data-pipeline.md` was done in Task 4)

- [ ] **Step 1: `docs/football.md`**

- §5: extra files are now reachable with `python -m football.downloader --extra
  --leagues COL`; opening odds only; one league per load; `odds_are_closing`
  always False; the corrected-edge claim is off the table for Colombian data.
- §7 "What is not built yet": scoring rules, Dixon-Coles, evaluation and the
  forecast tab are now built. Elo is still deferred — say why (it only gives
  1X2, and the scoreline forecast needed a goals model regardless).
- New section "8. The model and the two-team view": Dixon-Coles parameters,
  the low-score correction, `half_life` time decay, how the Pronóstico tab
  shows model beside market, how the Resultados tab scores it walk-forward.

- [ ] **Step 2: `docs/models.md`**

Add a football section: Dixon-Coles — parameter set, `tau` correction,
`DixonColes.fit(matches, half_life=)`, the prediction API, and "how to add
another football model": produce `(n, 3)` probabilities in `OUTCOMES` order,
score it with `football/scoring.py`, evaluate it with
`football/evaluation.py:beats_market_test` (bump `n_comparisons` when scoring
several models against the same matches).

- [ ] **Step 3: `docs/evaluation.md`**

Add `football/backtest.py` beside the lottery backtest: the paired-difference
framing (market score minus model score, tested against 0), `run_all` vs
`run_holdout(mode=)`, and that the verdict is one-sided and Bonferroni-aware
just like the lottery's.

- [ ] **Step 4: `docs/dashboard.md` §3**

Document the fourth tab (Pronóstico), the Europa/Colombia source toggle, and
the evaluation section in Resultados. Note that Colombia mode shows the
opening-odds warning.

- [ ] **Step 5: `CLAUDE.md`**

- **Module layout:** add entries for `football/scoring.py`,
  `football/dixon_coles.py`, `football/h2h.py`, `football/extra_processor.py`,
  `football/evaluation.py`, `football/backtest.py`.
- **"What this project is":** replace "Football is currently a data layer only:
  contract, market baseline, downloader, synthetic data, and a dashboard page
  that scores nothing. No models yet." with a sentence saying Dixon-Coles, its
  scoring rules and a walk-forward market evaluation now exist, Elo is still
  skipped, and the dashboard's Pronóstico tab is the two-team view.
- **`dashboard/` bullet:** update the football line in the "three domains are
  not equally built" note — football now has a model, a scoring rule and a
  backtest; cycling is still data-only.
- **Cross-cutting invariants:** add
  - *Football scoring is proper and ordered.* `football/scoring.py` uses RPS by
    default because the outcome H–D–A is ordered; a symmetric metric (Brier)
    is a diagnostic, RPS is the verdict. Same shape as the lottery's
    "pooled test is the verdict".
  - *A football forecast is never shown without the market.* Every dashboard
    surface with model probabilities shows the market beside them and links the
    measured verdict; `football/evaluation.py` is a paired one-sided test with
    `null_mean = 0`.
  - *Extra files are opening-odds-only.* `football/extra_processor.py` pins
    `odds_are_closing = False`; no corrected edge claim is possible on
    Colombian data.
- **Setup and commands:** add
  `python -m football.backtest --seasons E0_2324.csv,E0_2223.csv --n-windows 30`
  and the `--extra` download example.
- **Documentation map table:** no new row needed (football.md covers it), but
  check the descriptions still read true.

- [ ] **Step 6: Run the doc link checker**

Run: `python -m lottery.utils.check_docs`
Expected: all links resolve. Fix any broken anchor (headings with `·`/`—`
double-hyphen as the checker docstring warns).

- [ ] **Step 7: Full verification**

```bash
python -m pytest -q
python -m pytest -m "not slow" -q
python -m py_compile dashboard/app.py dashboard/football_page.py dashboard/ui.py
python -m lottery.utils.check_docs
```

All must pass. Then the Streamlit + Playwright pass over all four Fútbol tabs
one more time (Datos, Mercado, Pronóstico, Resultados), checking each tab panel
for `Traceback` / "This app has encountered an error".

- [ ] **Step 8: Commit**

```bash
git add docs/ CLAUDE.md
git commit -m "Docs: Dixon-Coles, football evaluation, the Pronóstico tab, the extra contract

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT"
```

- [ ] **Step 9: Open the PR**

```bash
git push -u origin claude/football-dixon-coles-forecast
gh pr create --title "Football: Dixon-Coles forecast, scoring, evaluation, two-team dashboard view" --body "$(cat <<'EOF'
## Summary

Turns the football half from a data layer into a forecasting tool.

- `football/scoring.py` — Brier / RPS / log-loss + skill score (RPS default; the outcome is ordered).
- `football/dixon_coles.py` — MLE fit (attack/defence/home/rho), optional time decay, scoreline grid → 1X2 / correct score / O-U / BTTS.
- `football/h2h.py` — recent form and head-to-head, descriptive only.
- `football/extra_processor.py` + `downloader --extra` — the `new/COL.csv` contract for Colombia; opening odds only, `odds_are_closing` pinned False.
- `football/evaluation.py` — paired one-sided proper-score test vs the market, naive + Bonferroni verdict, through `core/significance.py`.
- `football/backtest.py` — walk-forward and date-cutoff evaluation, `expanding` / `frozen` modes.
- Dashboard — fourth **Pronóstico** tab (two teams, optional current odds shown de-margined beside the model), a measured verdict section in **Resultados**, and a Europa/Colombia source toggle.

`core/` is untouched. Every surface shows the model beside the market.

## Testing

- `pytest` green (new suites: scoring, h2h, dixon_coles [slow], extra_processor, evaluation, backtest [slow]).
- `pytest -m "not slow"` green.
- `python -m lottery.utils.check_docs` green.
- Streamlit + Playwright over all four Fútbol tabs, no errors.

Spec: `docs/superpowers/specs/2026-09-06-football-dixon-coles-forecast-design.md`
Plan: `docs/superpowers/plans/2026-09-06-football-dixon-coles-forecast.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01H9QfED2Adzb3HLHHx8KDaT
EOF
)"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
| --- | --- |
| §3.1 `scoring.py` | Task 1 |
| §3.2 `dixon_coles.py` (params, tau, decay, prediction API, UnknownTeamError) | Task 3 |
| §3.3 `h2h.py` | Task 2 |
| §3.4 `extra_processor.py` | Task 4 |
| §3.5 `downloader --extra` | Task 4 |
| §3.6 `evaluation.py` | Task 5 |
| §3.7 `backtest.py` | Task 6 |
| §4.1 Pronóstico tab | Task 7 |
| §4.2 Resultados evaluation + model calibration | Task 8 |
| §4.3 `ui.py` HELP keys | Task 7 (defined), Task 8 (consumed) |
| §5 tests | one per task |
| §6 docs | Task 9 (data-pipeline in Task 4) |
| §7 build order | tasks are in that order; Task 4 flagged separable |
| §8 verification | Task 9 Step 7 |
| §9 risks (fit speed, coverage, promoted teams) | slow marks + modest defaults (Task 6); `UnknownTeamError` (Task 3, 7) |

No gaps.

**Placeholder scan:** every code step carries real code. The dashboard tasks
(7, 8) reference reading the current file first because the exact insertion
points depend on line numbers that will have shifted, but the code to insert is
given in full. No "add error handling" / "similar to Task N" / "TODO".

**Type consistency:** `DixonColes.fit` / `.predict_outcome` / `.scoreline_matrix`
/ `.most_likely_scores` / `.over_under` / `.both_teams_to_score` /
`.predict_matches` used identically in Tasks 3, 6, 7, 8. `beats_market_test`
signature and return keys identical in Tasks 5, 6, 8. `per_match_scores(probs,
outcomes, metric)` identical in Tasks 1, 5. `team_form` / `head_to_head` return
keys identical in Tasks 2, 7. `load_extra(path, league=, validate=)` /
`available_leagues(path)` identical in Tasks 4, 7. `preprocess_extra` sets
`attrs["odds_are_closing"] = False` (Task 4) which Task 8 relies on for the
opening-odds banner. Verdict rename `beats_chance` → `beats_market` is applied
in one place (Task 5) and consumed by name everywhere else.
