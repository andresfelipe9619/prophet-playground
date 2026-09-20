"""How much of a forecast is the model, and how much is the sample it saw.

`DixonColes` and `Elo` return point estimates. Six matches into a season those
points are confidently wrong, and nothing in a probability vector says so — a
0.62 fitted from eighty matches and a 0.62 fitted from six look identical on
screen and mean completely different things.

This module attaches the missing half: a **bootstrap band** on the predicted
probabilities. Resample the training matches with replacement, refit, predict,
and take the quantiles across refits. The spread of those predictions is how
much the forecast would have moved had the season gone slightly differently,
which is the question a standard error is trying to answer anyway.

**Why bootstrap rather than a Hessian.** The obvious alternative is the inverse
observed information at the optimum, which is cheaper and, here, wrong in a way
that is hard to notice. Dixon-Coles' `rho` is **bounded** at ±0.4, and a
quadratic approximation around a *constrained* optimum understates the
uncertainty exactly where it matters.

That is not a theoretical worry — `attrs["n_at_bound"]` counts it, and on this
project's synthetic seasons it dominates the small-sample case:

    training matches   band width   refits with a parameter at its bound
                  20        0.726                           76 of 80
                  45        0.461                           48 of 80
                  90        0.353                           21 of 80

At twenty matches nineteen refits in twenty end up against the bound, so a
Hessian there would be describing the curvature of a wall. Resampling makes no
such assumption, costs a few hundred refits, and is the honest answer at the
sample sizes this project works at. Note the band narrows with data and the
bound-hit rate falls with it: both are the same fact seen twice.

**What the band is not.** It is sampling uncertainty — how much the fit moves
when the data moves — and nothing else. A model that is wrong about football
will produce a tight band around a wrong number, and a narrow band is therefore
not evidence of anything. It says how much of the forecast came from the sample,
not whether the model was the right shape to fit.

**Where it is load-bearing:** `value.py`. A bet is a bet only if the model's
probability clears the raw price, and a *point* clearing it by a hair while its
band straddles it is noise with a favourable sign. `downgrade_uncertain_value`
demotes those to `disagreement_only` — the state this package already keeps for
"the model disagrees with the price and cannot pay for the spread", now
extended to "the model may not disagree with the price at all".
"""

import warnings

import numpy as np
import pandas as pd

from football.common import OUTCOMES, PROBABILITY_COLUMNS
from football.value import DISAGREEMENT_ONLY, VALUE, break_even_probability

DEFAULT_RESAMPLES = 200
DEFAULT_CONFIDENCE = 0.90


def bootstrap_predictions(matches, fixtures, fit, n_resamples=DEFAULT_RESAMPLES,
                          confidence=DEFAULT_CONFIDENCE, seed=0, half_life=None):
    """Refit on resampled seasons and return the spread of each fixture's forecast.

    `fit(matches, half_life)` returns a fitted model exposing
    `predict_outcome(home, away)`; `fixtures` is an iterable of `(home, away)`.
    Both `DixonColes.fit` and `Elo.fit` match that shape, which is the only
    thing this module requires of a model.

    A resample that cannot predict a fixture — a team that happened not to be
    drawn, or a fit that came back non-finite — is **skipped for that fixture
    only**, and `n_usable` reports how many survived. Dropping the whole
    resample would bias the band toward seasons in which every team appears,
    which is exactly the seasons where the fit is most confident.

    Non-finite predictions are filtered rather than propagated. A single NaN
    turns a quantile into NaN, so one degenerate resample silently erases the
    band for **every** fixture — which is how the first version of this function
    behaved, reporting 57 usable draws and a band of NaN in the same row. That
    reads as "no uncertainty computed" rather than "one bad refit", so
    `n_usable` now counts draws that survived rather than draws attempted.

    Returns a frame with one row per fixture and outcome: the point estimate
    from the full-sample fit, the band, the count behind it, and
    `attrs["n_at_bound"]` — how many refits pushed a parameter to its limit,
    which is the evidence for preferring a bootstrap over a Hessian here.
    """
    fixtures = [tuple(f) for f in fixtures]
    rng = np.random.default_rng(seed)

    point = fit(matches, half_life)
    point_probs = {f: np.asarray(point.predict_outcome(*f), dtype=float) for f in fixtures}

    draws = {f: [] for f in fixtures}
    at_bound = 0
    for _ in range(int(n_resamples)):
        sample = matches.iloc[rng.integers(0, len(matches), len(matches))]
        # The bound warning is the model's own voice and worth counting, but
        # 200 copies of it is noise. Caught, tallied, and reported once.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                refit = fit(sample, half_life)
            except Exception:  # noqa: BLE001 — a degenerate resample is data, not a defect
                continue
        at_bound += any("bound" in str(w.message) for w in caught)

        for fixture in fixtures:
            try:
                predicted = np.asarray(refit.predict_outcome(*fixture), dtype=float)
            except Exception:  # noqa: BLE001 — team absent from this resample
                continue
            if np.isfinite(predicted).all():
                draws[fixture].append(predicted)

    tail = (1.0 - confidence) / 2.0
    rows = []
    for fixture in fixtures:
        stacked = np.array(draws[fixture]) if draws[fixture] else np.empty((0, 3))
        for i, outcome in enumerate(OUTCOMES):
            column = stacked[:, i] if len(stacked) else np.array([])
            rows.append({
                "home_team": fixture[0], "away_team": fixture[1], "outcome": outcome,
                "probability": float(point_probs[fixture][i]),
                "low": float(np.quantile(column, tail)) if column.size else float("nan"),
                "high": float(np.quantile(column, 1.0 - tail)) if column.size else float("nan"),
                "n_usable": int(column.size),
            })
    out = pd.DataFrame(rows)
    out.attrs["n_at_bound"] = int(at_bound)
    out.attrs["n_resamples"] = int(n_resamples)
    return out


def band_width(bands):
    """Mean width of the interval, per fixture — one number for "how sure is this".

    Worth reporting beside a forecast rather than instead of it: a wide band on
    a promoted side is information about the data, not a reason to distrust the
    model more than the sample warrants.
    """
    out = bands.copy()
    out["width"] = out["high"] - out["low"]
    return (out.groupby(["home_team", "away_team"], as_index=False)["width"].mean()
            .rename(columns={"width": "mean_band_width"}))


def downgrade_uncertain_value(states, bands_low, odds):
    """Demote a `value` call whose band crosses the price it has to beat.

    The point estimate clearing `1/odds` while the lower end of its band does
    not is a bet placed on the sampling noise having landed favourably. It is
    the same failure `value.py` already names for a disagreement too small to
    pay the margin, so it gets the same state rather than a new one: the reader
    has one thing to learn, and both mean "not a bet".

    Nothing is ever upgraded. A band's upper end clearing the price while the
    point does not is still a model that does not, on its own estimate, have an
    edge — and promoting that would turn an interval into a second opinion.
    """
    states = np.asarray(states, dtype=object)
    low = np.asarray(bands_low, dtype=float)
    threshold = np.asarray(break_even_probability(odds), dtype=float)

    uncertain = (states == VALUE) & ~(low > threshold)
    return np.where(uncertain, DISAGREEMENT_ONLY, states)


def with_bands(value_rows, bands):
    """Join a `value_table` to its bands and apply the downgrade.

    `value_table` names its column `verdict`, so that is what this reads and
    writes — the helper follows the existing API rather than inventing a second
    name for the same thing. The pre-band call is kept as `verdict_point`, so a
    row that changed is visible rather than silently rewritten; those rows are
    the most interesting ones in the table.
    """
    merged = value_rows.merge(bands[["outcome", "low", "high"]], on="outcome", how="left")
    merged["verdict_point"] = merged["verdict"]
    merged["verdict"] = downgrade_uncertain_value(
        merged["verdict"].to_numpy(), merged["low"].to_numpy(), merged["odds"].to_numpy())
    return merged


def probability_columns(bands, fixture):
    """The three-vector and its band for one fixture, in `PROBABILITY_COLUMNS` order."""
    rows = bands[(bands["home_team"] == fixture[0]) & (bands["away_team"] == fixture[1])]
    ordered = rows.set_index("outcome").loc[list(OUTCOMES)]
    return {
        "probabilities": dict(zip(PROBABILITY_COLUMNS, ordered["probability"], strict=True)),
        "low": dict(zip(PROBABILITY_COLUMNS, ordered["low"], strict=True)),
        "high": dict(zip(PROBABILITY_COLUMNS, ordered["high"], strict=True)),
    }
