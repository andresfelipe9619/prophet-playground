"""What a single strength number leaves out: terrain, specialisation, team, fatigue.

`plackett_luce.py` gives every rider one worth. That is the right first model
and it is wrong about the thing cycling fans know best — a sprinter and a
climber are not two points on one scale, they are good at different days. This
module builds the covariates that say so, under the constraint that decides
their shape:

**Terrain is known before a race; a result's terrain is not.** The parcours of
tomorrow's stage is published months in advance, so conditioning on it is not
foresight. But nothing in this project's data contract carries it — a scraped
result has a date, a rank and a time, and no roadbook. So terrain here is
*inferred from the finish*, which is legitimate for a stage that has already
happened and is **leakage for the stage being predicted**. `infer_terrain` is
therefore documented and named as a history-only tool, and everything that
conditions on terrain takes the target race's terrain as an argument the caller
supplies from outside. `terrain_of` exists to make that supply easy for a *past*
race and raises rather than quietly labelling a race at or after `as_of`.

**What the inference actually reads.** The share of finishers credited with the
winner's exact time. A flat stage ends in a bunch sprint and 80% of the field
shares one time; a mountain stage strings the race out and almost nobody does.
Measured on the synthetic Grand Tour the two classes sit at 0.80 and 0.01 with
nothing in between, and on real results the same marker is what the `,,` on a
results page means. It reads the published time structure rather than the time
*spread*, because a spread in seconds is not comparable between a 4-hour stage
and a 45-minute time trial.

**Every feature is built strictly before `as_of`**, the same refusal
`baseline.form_worths` makes, and for the same reason: a feature that has seen
the race it describes is an answer key with extra steps. The test that would
notice a `<` becoming a `<=` is in `tests/test_cycling_features.py`.

**Team strength excludes the rider it is computed for.** A team's form that
includes the rider's own results is that rider's form wearing a team jersey, and
it would enter any model twice — once as their worth and once as their team's.
Leaving them out is what makes it a statement about the eight riders around them.

**Fatigue is race days, not calendar days.** A rider who has raced eighteen of
the last twenty-one days and one who flew in for this race are in different
states, and the date alone does not distinguish them. This counts only the days
a rider actually appears in a result, so a rider who abandoned two weeks ago
correctly stops accumulating.

None of these are wired into a model by this module. `plackett_luce.fit_by_terrain`
consumes the terrain labels, and whether that helps is a measurement rather than
an assumption — `cycling/evaluation.py` makes it, and the answer on the
synthetic races is in `docs/cycling.md`.
"""

import numpy as np
import pandas as pd

from cycling.common import FINISHED
from cycling.processor import GROUP_KEYS

CLIMB = "climb"
SPRINT = "sprint"
UNKNOWN = "unknown"
TERRAINS = (CLIMB, SPRINT, UNKNOWN)

# Above this share of finishers on the winner's exact time, the race ended in a
# bunch. The synthetic classes sit at 0.80 and 0.01, so the threshold is not
# delicate; it is written as a constant rather than tuned, because a number
# fitted to make a downstream model look better is the anti-pattern this
# repository is built around.
BUNCH_SHARE_THRESHOLD = 0.5

# Below this many finishers a bunch share is not a measurement — a five-rider
# result can read as 0.8 by accident.
MIN_FINISHERS = 10

DEFAULT_FATIGUE_WINDOW_DAYS = 21


def bunch_share(group):
    """Share of finishers credited with the winner's exact time.

    NaN when the times are missing or the field is too small to read, which is a
    frequent and ordinary state: a result page without times is still a result.
    """
    finishers = group[(group["status"] == FINISHED) & group["time_seconds"].notna()]
    if len(finishers) < MIN_FINISHERS:
        return float("nan")
    times = finishers["time_seconds"].to_numpy(dtype=float)
    return float((times == times.min()).mean())


def infer_terrain(results, threshold=BUNCH_SHARE_THRESHOLD):
    """Label each past race `climb` / `sprint` / `unknown` from how it finished.

    **History only.** This reads the result, so using it on the race being
    predicted is leakage — and the leak would be invisible, because the label
    looks exactly the same either way. For the target race the terrain comes
    from the roadbook, supplied by the caller.

    Returns one row per (race, kind, stage) with the share it was read from, so
    a label near the threshold is visible rather than just categorical.
    """
    rows = []
    for key, group in results.groupby(GROUP_KEYS, dropna=False):
        share = bunch_share(group)
        rows.append({
            "race": key[0], "kind": key[1], "stage": key[2],
            "ds": group["ds"].min(),
            "bunch_share": share,
            "terrain": UNKNOWN if not np.isfinite(share)
            else (SPRINT if share >= threshold else CLIMB),
        })
    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


def terrain_of(results, race, kind, stage=None, as_of=None, threshold=BUNCH_SHARE_THRESHOLD):
    """The inferred terrain of one **past** race, refused for one at or after `as_of`.

    The refusal is the point. Reading a label off a race that has not happened
    yet is the one mistake this module can make invisibly, so it is made
    impossible to make quietly rather than warned about.
    """
    labels = infer_terrain(results, threshold=threshold)
    match = labels[(labels["race"] == race) & (labels["kind"] == kind)]
    match = match[match["stage"].isna()] if stage is None else match[match["stage"] == stage]
    if match.empty:
        return UNKNOWN

    row = match.iloc[0]
    if as_of is not None and row["ds"] >= pd.Timestamp(as_of):
        raise ValueError(
            f"{race} ({kind}, stage {stage}) is on {row['ds'].date()}, at or after as_of "
            f"{pd.Timestamp(as_of).date()}. Its terrain would be read off its own result. "
            "Supply the target race's terrain from the roadbook instead."
        )
    return row["terrain"]


def _history(results, as_of):
    """Strictly before, always. The one line every feature here goes through."""
    return results if as_of is None else results[results["ds"] < pd.Timestamp(as_of)]


def terrain_history(results, terrain, as_of=None, threshold=BUNCH_SHARE_THRESHOLD):
    """The subset of earlier results run on `terrain`, for fitting on one kind of day.

    `terrain=None` returns the whole history, which is what makes a
    terrain-conditional caller collapse to the unconditional one rather than
    needing a second code path.
    """
    history = _history(results, as_of)
    if terrain is None or history.empty:
        return history

    labels = infer_terrain(history, threshold=threshold)
    wanted = labels[labels["terrain"] == terrain][GROUP_KEYS]
    if wanted.empty:
        return history.iloc[0:0]
    keyed = history.merge(wanted, on=GROUP_KEYS, how="inner")
    keyed.attrs.update(history.attrs)
    return keyed


def placing_percentile(results):
    """Each finisher's placing as a share of the field beaten, 1 being a win.

    Comparable across races in a way a rank is not: 20th of 180 and 20th of 25
    are the same number and nothing alike. Non-finishers score 0 — they beat
    nobody, which is the same refusal the scoring rule makes in arithmetic.
    """
    out = results.copy()
    sizes = out.groupby(GROUP_KEYS, dropna=False)["rider"].transform("size")
    beaten = sizes - out["rank"]
    out["placing_percentile"] = np.where(
        (out["status"] == FINISHED) & out["rank"].notna(),
        beaten / np.maximum(sizes - 1, 1), 0.0)
    return out


def terrain_form(results, riders, as_of=None, terrain=None, threshold=BUNCH_SHARE_THRESHOLD):
    """Mean placing percentile on one kind of day, per rider, from earlier races only.

    A descriptive feature, not a forecast: it is the same quantity for a rider
    with twenty mountain stages behind them and one with a single lucky
    breakaway, so `n_races` comes back beside it and is the column that says
    which of the two is being read.
    """
    history = terrain_history(results, terrain, as_of=as_of, threshold=threshold)
    scored = placing_percentile(history)
    grouped = scored.groupby("rider")["placing_percentile"]
    mean, count = grouped.mean(), grouped.count()
    return pd.DataFrame({
        "rider": list(riders),
        "terrain": terrain if terrain is not None else "all",
        "form": [float(mean.get(r, np.nan)) for r in riders],
        "n_races": [int(count.get(r, 0)) for r in riders],
    })


def specialisation(results, riders, as_of=None, threshold=BUNCH_SHARE_THRESHOLD):
    """Climbing form minus sprinting form — positive is a climber.

    The difference rather than the two levels, because the levels are dominated
    by how good the rider is overall and the interesting part is the *tilt*. A
    rider with no races on one of the two terrains gets NaN rather than a
    difference against nothing.
    """
    climb = terrain_form(results, riders, as_of=as_of, terrain=CLIMB, threshold=threshold)
    sprint = terrain_form(results, riders, as_of=as_of, terrain=SPRINT, threshold=threshold)
    return pd.DataFrame({
        "rider": list(riders),
        "climb_form": climb["form"].to_numpy(),
        "sprint_form": sprint["form"].to_numpy(),
        "specialisation": climb["form"].to_numpy() - sprint["form"].to_numpy(),
        "n_climb": climb["n_races"].to_numpy(),
        "n_sprint": sprint["n_races"].to_numpy(),
    })


def team_strength(results, riders, as_of=None, teams=None):
    """How the rest of a rider's team has been going, excluding the rider.

    `teams` maps rider to team for the race being predicted; without it the most
    recent team seen in the history is used, which is right for a season and
    wrong across a winter transfer — so a caller with a start list should pass
    it.

    The exclusion is what makes this a feature rather than a restatement: a
    team mean that includes the rider is their own form counted twice.
    """
    history = placing_percentile(_history(results, as_of))
    if teams is None:
        teams = (history.sort_values("ds").groupby("rider")["team"].last().to_dict()
                 if len(history) else {})

    totals = history.groupby("team")["placing_percentile"].agg(["sum", "count"])
    own = history.groupby("rider")["placing_percentile"].agg(["sum", "count"])

    rows = []
    for rider in riders:
        team = teams.get(rider)
        total, count = (totals.loc[team] if team in totals.index else (0.0, 0))
        mine, mine_count = (own.loc[rider] if rider in own.index else (0.0, 0))
        others = count - mine_count
        rows.append({
            "rider": rider, "team": team,
            "team_strength": float((total - mine) / others) if others > 0 else float("nan"),
            "n_team_races": int(others),
        })
    return pd.DataFrame(rows)


def race_days(results, riders, as_of=None, window_days=DEFAULT_FATIGUE_WINDOW_DAYS):
    """Days actually raced in the trailing window, and the days since the last one.

    Race days rather than calendar days: a rider who has raced eighteen of the
    last twenty-one and one who flew in are in different states and the date
    alone cannot tell them apart. A rider who abandoned a fortnight ago stops
    accumulating, which is the behaviour that makes this readable on a Grand
    Tour where a sixth of the field is no longer in the race.
    """
    history = _history(results, as_of)
    if as_of is not None and len(history):
        window = history[history["ds"] >= pd.Timestamp(as_of) - pd.Timedelta(days=window_days)]
    else:
        window = history

    counted = window.groupby("rider")["ds"].nunique()
    last = history.groupby("ds").size().index.max() if len(history) else None
    latest = history.groupby("rider")["ds"].max()

    rows = []
    for rider in riders:
        seen = latest.get(rider, pd.NaT)
        reference = pd.Timestamp(as_of) if as_of is not None else last
        rows.append({
            "rider": rider,
            "race_days": int(counted.get(rider, 0)),
            "days_since_last": (float((reference - seen).days)
                                if pd.notna(seen) and reference is not None else float("nan")),
        })
    return pd.DataFrame(rows)


def feature_frame(results, riders, as_of=None, terrain=None, teams=None,
                  window_days=DEFAULT_FATIGUE_WINDOW_DAYS, threshold=BUNCH_SHARE_THRESHOLD):
    """Every feature above for one start list, from results strictly before `as_of`.

    `terrain` is the **target race's** terrain, supplied from the roadbook. It is
    never inferred here, for the reason the module docstring gives: the inference
    reads a result, and the target's result is the thing being predicted.
    """
    frames = [
        terrain_form(results, riders, as_of=as_of, terrain=terrain, threshold=threshold)
        .rename(columns={"form": "terrain_form", "n_races": "n_terrain_races"}),
        specialisation(results, riders, as_of=as_of, threshold=threshold).drop(columns=["rider"]),
        team_strength(results, riders, as_of=as_of, teams=teams).drop(columns=["rider"]),
        race_days(results, riders, as_of=as_of, window_days=window_days).drop(columns=["rider"]),
    ]
    out = pd.concat(frames, axis=1)
    out.attrs["as_of"] = None if as_of is None else pd.Timestamp(as_of)
    out.attrs["terrain"] = terrain
    return out


def roadbook(dates, terrains):
    """A `{date: terrain}` mapping for the races being predicted.

    This is the exogenous half. A published roadbook says which stages are
    mountain stages long before the race, so conditioning on it is not
    foresight — but it has to come from *somewhere other than the result*, and
    this project's data contract carries no such column. Building one is the
    caller's job: from a roadbook, from a hand-written list, or, in tests, from
    the generator's own stage plan, which is known before the race is simulated
    and is therefore honestly exogenous.
    """
    return {pd.Timestamp(d): t for d, t in zip(list(dates), list(terrains), strict=True)}


def terrain_forecaster(roadbook_by_date, min_races=None, **fit_kwargs):
    """A `walk_forward` forecaster fitting per terrain and predicting with the target's.

    The terrain of the race being predicted comes from `roadbook_by_date`, never
    from its result. A date the roadbook does not cover falls back to the
    unconditional fit rather than guessing, which is also what happens when a
    terrain has too few races to fit on its own — and the fallback is the honest
    answer in both cases, not a degradation.
    """
    from cycling.plackett_luce import MIN_TERRAIN_RACES, TerrainPlackettLuce

    min_races = MIN_TERRAIN_RACES if min_races is None else min_races

    def forecast(history, riders, as_of):
        model = TerrainPlackettLuce.fit(history, min_races=min_races, **fit_kwargs)
        return model.worths_for(riders, roadbook_by_date.get(pd.Timestamp(as_of)))

    return forecast
