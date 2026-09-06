"""Seeded synthetic stage races, so the cycling package is runnable and testable
without scraping anything — and, unlike real results, checkable against the answer.

The pattern is `football/sample_data.py`'s: generate from a real model and hand
the model's own truth back alongside the data. Here the truth is per rider
rather than per row — `ability_true`, the latent strength that decides the
finishing order — because that is what a cycling model is actually estimating.
On real results nobody knows it, so a model can only be compared to another
model; here it is known, which is what lets a test check that a forecast which
*is* the truth beats the ranking baseline.

Three things are modelled because leaving them out would make every downstream
test easier than reality and less informative:

- **Bunch finishes.** On a flat stage most of the field is given the winner's
  exact time. So a synthetic stage race must produce large groups of identical
  times, and any code that assumes times are distinct meets that here rather
  than on the first real Tour.
- **Abandons.** Riders leave the race and do not come back. The stage they
  leave on carries a DNF row with no rank and no time, and later stages simply
  do not list them — which is what a scraped file looks like and what
  `cycling/processor.py` refuses to drop silently.
- **Time gaps that grow.** Ability shows up as seconds on a mountain stage and
  as nothing at all in a sprint, so `climbing_stages` scales the spread rather
  than applying one spread to every day.

**What this cannot do.** The generator has no course, no weather, no teams
riding for each other and no crashes correlated across riders. Abandons are
independent draws against a per-rider hazard, which is wrong in the way that
matters most — real abandons cluster on the same day, in the same crash. It is
a fair test bed for a rider-strength model and not a substitute for real
results when the question is about dependence between riders.

`ability_true` is an answer key. Nothing outside tests may read it.
"""

import numpy as np
import pandas as pd

from cycling.common import (
    CSV_COLUMNS,
    DNF,
    FINISHED,
    GC,
    STAGE,
)
from cycling.processor import preprocess_results

# Calibrated to look like a Grand Tour rather than picked for convenience: 176
# starters, three weeks, roughly one rider in six not reaching the end, and
# ability worth minutes over a mountain stage and nothing in a sprint.
DEFAULT_RIDERS = 176
DEFAULT_STAGES = 21
DEFAULT_TEAM_SIZE = 8
BASE_STAGE_SECONDS = 4 * 3600.0     # a four-hour stage
ABILITY_SPREAD = 1.0                # sd of the latent strength, in arbitrary units
CLIMBING_SECONDS_PER_ABILITY = 180.0  # what one unit of ability is worth on a hard day
SPRINT_SECONDS_PER_ABILITY = 4.0     # ... and on a flat one
NOISE_SECONDS = 45.0                # day-to-day form and race circumstance
ABANDON_HAZARD = 0.008              # per rider per stage; ~15% over three weeks
BUNCH_WINDOW_SECONDS = 25.0         # on a climbing day, finishing this close shares a time
SPRINT_BUNCH_SHARE = 0.8            # on a flat day, this share of the field shares the winner's


def rider_abilities(n_riders=DEFAULT_RIDERS, seed=0, spread=ABILITY_SPREAD,
                    team_size=DEFAULT_TEAM_SIZE):
    """The start list, with the latent strength that generates the results.

    Ability is centred so the field average is exactly neutral, which keeps a
    stage's winning time at `BASE_STAGE_SECONDS` whatever the spread is set to.
    """
    rng = np.random.default_rng(seed)
    ability = rng.normal(0.0, spread, n_riders)
    return pd.DataFrame({
        "rider": [f"Rider {i + 1:03d}" for i in range(n_riders)],
        "team": [f"Team {i // team_size + 1:02d}" for i in range(n_riders)],
        "ability_true": ability - ability.mean(),
    })


def generate_stage_race(n_stages=DEFAULT_STAGES, n_riders=DEFAULT_RIDERS, seed=0,
                        race="synthetic-grand-tour", start_date="2024-06-29",
                        climbing_stages=None, abandon_hazard=ABANDON_HAZARD,
                        bunch_finishes=True):
    """A stage race's results in the on-disk contract's own column shape.

    Output goes through the real parser in `cycling/processor.py`, so every test
    that uses this data exercises the contract rather than an in-memory
    shortcut. Returns a frame with `CSV_COLUMNS`.

    `climbing_stages` is a set of 1-based stage numbers where ability is worth
    minutes; the rest are sprints where it is worth seconds. It defaults to
    every third stage, which puts a realistic third of the race in the mountains.
    """
    rng = np.random.default_rng(seed)
    riders = rider_abilities(n_riders, seed=seed)
    if climbing_stages is None:
        climbing_stages = set(range(3, n_stages + 1, 3))

    start = pd.to_datetime(start_date)
    active = list(riders.index)
    rows = []

    for stage in range(1, n_stages + 1):
        date = (start + pd.Timedelta(days=stage - 1)).strftime("%d/%m/%Y")
        seconds_per_ability = (CLIMBING_SECONDS_PER_ABILITY if stage in climbing_stages
                               else SPRINT_SECONDS_PER_ABILITY)

        # Who leaves the race today. They get a DNF row on this stage and do not
        # appear on any later one, which is exactly how a scraped file reads.
        abandoning = [i for i in active if rng.random() < abandon_hazard]
        finishing = [i for i in active if i not in abandoning]

        times = {
            i: BASE_STAGE_SECONDS
            - riders.at[i, "ability_true"] * seconds_per_ability
            + rng.normal(0.0, NOISE_SECONDS)
            for i in finishing
        }
        order = sorted(finishing, key=lambda i: times[i])
        winning_time = times[order[0]] if order else None

        # A bunch finish credits a group of riders with the winner's exact time
        # while keeping their placings — what the ',,' marker on a results page
        # means. The rule differs by terrain because reality does: on a flat
        # stage the peloton arrives together and most of the field shares one
        # time, while on a mountain stage only the riders who were still with
        # the leader at the line do.
        bunch_size = (0 if not bunch_finishes
                      else max(1, int(SPRINT_BUNCH_SHARE * len(order)))
                      if stage not in climbing_stages else 0)

        for rank, index in enumerate(order, start=1):
            in_bunch = (rank <= bunch_size
                        or (bunch_finishes and stage in climbing_stages
                            and times[index] - winning_time <= BUNCH_WINDOW_SECONDS))
            recorded = winning_time if in_bunch else times[index]
            rows.append(_row(date, race, STAGE, stage, rank, riders.loc[index],
                             FINISHED, recorded))

        for index in abandoning:
            rows.append(_row(date, race, STAGE, stage, None, riders.loc[index], DNF, None))

        active = finishing

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def _row(date, race, kind, stage, rank, rider, status, seconds):
    return {
        "Date": date,
        "Race": race,
        "Kind": kind,
        "Stage": "" if stage is None else stage,
        "Rank": "" if rank is None else rank,
        "Rider": rider["rider"],
        "Team": rider["team"],
        "Status": status,
        "TimeSeconds": "" if seconds is None else round(float(seconds), 3),
    }


def general_classification(stage_rows, race=None):
    """The GC that follows from a set of stage results: cumulative time, re-ranked.

    Only riders who finished **every** stage are classified, which is the rule a
    real GC follows and the reason this is a separate frame rather than a column
    on the stage results: its `rank` is a different quantity, over a different
    period, and `cycling/processor.py` will not load the two together.
    """
    stages = preprocess_results(stage_rows, validate=False)
    finished = stages[stages["status"] == FINISHED]
    n_stages = int(stages["stage"].max())

    per_rider = finished.groupby("rider").agg(
        stages_finished=("stage", "count"),
        total=("time_seconds", "sum"),
        team=("team", "last"),
    )
    classified = (per_rider[per_rider["stages_finished"] == n_stages]
                  .sort_values("total"))

    date = stages["ds"].max().strftime("%d/%m/%Y")
    race = race or stages["race"].iloc[0]
    rows = [
        {
            "Date": date, "Race": race, "Kind": GC, "Stage": n_stages,
            "Rank": rank, "Rider": rider, "Team": row["team"],
            "Status": FINISHED, "TimeSeconds": round(float(row["total"]), 3),
        }
        for rank, (rider, row) in enumerate(classified.iterrows(), start=1)
    ]
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def load_sample_and_preprocess(validate=False, **kwargs):
    """Tidy stage results plus the generative truth, joined on the rider.

    Returns what `cycling.processor.load_and_preprocess` returns, with an
    `ability_true` column added. That column does not exist on real data and
    nothing outside tests may depend on it — it is the answer key, and a model
    reading its own answer key is not being measured.
    """
    raw = generate_stage_race(**kwargs)
    results = preprocess_results(raw, validate=validate)

    truth = rider_abilities(
        n_riders=kwargs.get("n_riders", DEFAULT_RIDERS),
        seed=kwargs.get("seed", 0),
    ).set_index("rider")["ability_true"]
    kind = results.attrs.get("result_kind")
    results["ability_true"] = results["rider"].map(truth).astype(float)
    results.attrs["result_kind"] = kind
    return results


def load_sample_gc(validate=False, **kwargs):
    """The general classification of the same synthetic race, as its own frame."""
    raw = generate_stage_race(**kwargs)
    return preprocess_results(general_classification(raw), validate=validate)
