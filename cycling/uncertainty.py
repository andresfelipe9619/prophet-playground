"""How much of a rider's worth is the rider, and how much is the races they got.

`football/uncertainty.py`'s question, asked of an ordering. A Plackett-Luce
worth fitted from three stages and one fitted from three weeks are the same
number on screen and completely different claims, and `plackett_luce.py`'s
Gamma prior already makes the thin-data case *visible* — a rider with little
evidence is pulled toward the field average — without making it **quantitative**.
This does that.

**The resampling unit is the race, not the rider.** A finishing order is one
observation: 180 riders placed relative to each other in a single event, and
resampling riders within a race would invent orderings that never happened and
break the very structure the model is fitted on. So whole races are drawn with
replacement, which is also what makes the band mean "had the season gone
differently" rather than "had this race been scored differently".

**The prior is not a nuisance to be removed for this.** Without it a rider
nobody finished behind has an unbounded worth, and a bootstrap over such fits
produces a band whose upper end is an artefact of which resample happened to
contain that rider's best day. The shrinkage is what makes the band readable,
and it is why a thin-evidence rider's interval is wide *and* centred near the
field rather than wide and centred on a fantasy.

**A wide band is information, not a verdict.** A neo-pro with two results has a
wide interval because two results is what is known about them — which is the
honest state of affairs, and exactly what a bare point estimate hides.

Two measured behaviours to know before reading a band, both of which contradict
the obvious expectation. Over 20 riders on synthetic stage races, three seeds,
4 races against 16:

    seed    absolute width    width / worth
       3     0.624 → 0.409    0.754 → 0.747
       7     0.553 → 0.491    0.757 → 1.027
      11     0.900 → 0.386    1.108 → 0.923

**The absolute width narrows every time. The relative width goes either way.**
As races accumulate riders differentiate, so the worths spread out and the
denominator grows alongside the numerator — by an amount that depends on how
the field happened to separate, which is why the ratio moves in no consistent
direction. `relative_band_width` is for comparing riders **within one fit**,
where the ratio scale makes an absolute width meaningless across strengths. It
is **not** a measure of how much evidence there is, and using it as one reads
noise as a trend in whichever direction the season took.

**The band is a percentile interval and need not contain the point**, and does
so less often as data accumulates. That is not a defect to correct: the
bootstrap distribution of a shrunk, mean-normalised, ratio-scale estimator is
not centred on the full-sample fit, and a rider whose full-calendar worth is
more extreme than a typical resample produces is a rider whose estimate leans
on which races happened. `contains_point` flags exactly those, and it is the
most useful column in the table — it is the model saying which of its own
numbers it would not reproduce.
"""

import numpy as np
import pandas as pd

from cycling.plackett_luce import PlackettLuce
from cycling.processor import GROUP_KEYS

DEFAULT_RESAMPLES = 200
DEFAULT_CONFIDENCE = 0.90


def _race_keys(results):
    """One key per race, which is the unit a bootstrap here resamples."""
    return list(results.groupby(list(GROUP_KEYS), dropna=False).groups.items())


def bootstrap_worths(results, riders, n_resamples=DEFAULT_RESAMPLES,
                     confidence=DEFAULT_CONFIDENCE, seed=0, **fit_kwargs):
    """Refit on resampled race calendars and return each rider's band.

    Races are drawn with replacement; a rider absent from a resample simply has
    one fewer draw, and `n_usable` reports how many they got. That is the
    honest treatment: a rider who appears in few races *should* end up with a
    band built from few refits, and forcing every rider to the same count would
    mean discarding races to match the thinnest one.

    Returns one row per rider with the full-sample worth, the band, and the
    count behind it.
    """
    riders = list(riders)
    groups = _race_keys(results)
    if not groups:
        raise ValueError("No races in this frame — nothing to resample.")

    rng = np.random.default_rng(seed)
    point = PlackettLuce.fit(results, **fit_kwargs)
    point_worths = dict(zip(riders, point.worths_for(riders), strict=True))

    draws = {rider: [] for rider in riders}
    for _ in range(int(n_resamples)):
        chosen = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([results.loc[groups[i][1]] for i in chosen])
        try:
            refit = PlackettLuce.fit(sample, **fit_kwargs)
        except Exception:  # noqa: BLE001 — a degenerate calendar is data, not a defect
            continue

        present = set(sample["rider"])
        for rider in riders:
            # A rider absent from this resample contributes nothing rather than
            # the prior's field average: recording the shrinkage target as if it
            # were an estimate would pull every thin rider's band toward the
            # middle and make it look tighter than the evidence is.
            if rider not in present:
                continue
            worth = float(refit.worth(rider))
            if np.isfinite(worth) and worth > 0:
                draws[rider].append(worth)

    tail = (1.0 - confidence) / 2.0
    rows = []
    for rider in riders:
        column = np.array(draws[rider], dtype=float)
        rows.append({
            "rider": rider,
            "worth": float(point_worths[rider]),
            "low": float(np.quantile(column, tail)) if column.size else float("nan"),
            "high": float(np.quantile(column, 1.0 - tail)) if column.size else float("nan"),
            "n_usable": int(column.size),
        })
    out = pd.DataFrame(rows)
    # Flagged rather than corrected: a point outside its own percentile band is
    # the fit saying it would not reproduce that number on a redrawn calendar.
    out["contains_point"] = (out["low"] <= out["worth"]) & (out["worth"] <= out["high"])
    out.attrs["n_resamples"] = int(n_resamples)
    out.attrs["n_races"] = len(groups)
    return out


def relative_band_width(bands):
    """Band width as a multiple of the worth — for comparing riders, not seasons.

    Worths are on a ratio scale with no natural unit, so an absolute width says
    nothing *across riders*: a band of 0.4 around a worth of 8 is tight and the
    same 0.4 around 0.3 is not. Within one fit this is the comparable form.

    **It does not track evidence**, measured across seeds: as races accumulate it
    rises, falls or stays flat depending on how the field separated, while the
    absolute width narrows every time. Use the absolute width for "how much is
    known"; this one only ranks riders against each other within one fit.
    """
    out = bands.copy()
    out["relative_width"] = (out["high"] - out["low"]) / out["worth"].replace(0.0, np.nan)
    return out


def appearances(results, riders):
    """How many races each rider actually appears in — the band's explanation.

    A wide interval is not a failure of the model; it is the count in this
    column being small. Reported beside the band so the two are read together.
    """
    counts = (results.groupby("rider")[list(GROUP_KEYS)[0]].count()
              if len(results) else pd.Series(dtype=int))
    return pd.DataFrame({"rider": list(riders),
                         "n_races": [int(counts.get(rider, 0)) for rider in riders]})
