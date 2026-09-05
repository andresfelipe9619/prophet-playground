"""Can these tests detect an edge that is actually there?

The rest of the project checks one direction: run the tests on i.i.d. uniform
draws and confirm they answer "looks random". That is a **specificity** check —
it proves the tests do not cry wolf. It says nothing about whether they can
hear a wolf at all. A test that always returns "looks random" would pass every
check in `utils/sample_data.py` perfectly.

So this module does the other direction. It generates draws with a **known,
deliberately planted bias**, runs the real detectors over them, and reports how
often each one fires. Without this, every null result in the project rests on
an untested assumption: that a real signal would have shown up.

Three detectors run together, and it is the *pattern* across them that carries
the information:

| Detector | On uniform draws | On biased draws |
| --- | --- | --- |
| `pooled` — `pooled_uniformity_test` | ~alpha | should fire |
| `hot` — `evaluate_strategy("hot")` | ~alpha | should fire |
| `random` — `evaluate_strategy("random")` | ~alpha | **still ~alpha** |

`random` staying at the floor on biased data is not a failure, it is the
control working. A ticket drawn uniformly has expected matches 5x5/43 no matter
how the balls are weighted — E[matches] sums P(drawn) over five numbers picked
without regard to the bias, and that sum is unchanged. Only a strategy that
*learns* which numbers are favoured can convert the bias into hits, which is
exactly what separates `hot` from `random` here.

`pooled` tests the data directly; `hot` tests the entire chain — generation,
ticket scoring, the hypergeometric baseline, the z-test. If `pooled` fires and
`hot` does not, the bug is downstream of the statistics, in the ticket path.

The strength at which detection crosses ~80% is this pipeline's sensitivity
threshold, and it is the number that gives a null result its meaning.
"""

import numpy as np
import pandas as pd

from analysis.randomness import pooled_uniformity_test
from analysis.tickets import evaluate_strategy
from models.common import (
    MAIN_BALLS_DRAWN,
    MAIN_BALL_RANGE,
    MAIN_POOL,
    SUPER_BALL_RANGE,
    main_positions,
    next_draw_dates,
)
from utils.processor import preprocess_draws

DEFAULT_ALPHA = 0.05

# 0.0 is not optional. It is the control: with no bias planted, every detection
# rate below must land near alpha, and a rate far above it means the detector is
# broken rather than sensitive — which would invalidate every positive row.
DEFAULT_STRENGTHS = (0.0, 0.25, 0.5, 1.0, 2.0)

# Three numbers, spread across the pool so the bias cannot be confused with an
# edge or ordering artifact. Which numbers is arbitrary; how many is not — a
# bias concentrated in fewer numbers is easier to detect at the same strength.
DEFAULT_FAVORED = (7, 21, 38)


def biased_draws(n_draws=500, favored=DEFAULT_FAVORED, strength=0.5, seed=0,
                 start_date="2018-01-03"):
    """Baloto-shaped draws where `favored` numbers are over-represented by a known amount.

    Each favoured number carries weight `1 + strength` against 1 for every
    other, sampled without replacement. `strength=0` reproduces uniform draws
    exactly, which is what makes it usable as the control arm.

    Output goes through the same CSV contract as real data, so the detectors
    downstream cannot tell this apart from a scraped file except by its
    statistics — the point being to test them, not to give them a shortcut.
    """
    if strength < 0:
        raise ValueError(f"strength must be non-negative, got {strength}")

    rng = np.random.default_rng(seed)
    pool = np.arange(MAIN_BALL_RANGE[0], MAIN_BALL_RANGE[1] + 1)
    weights = np.ones(len(pool))
    weights[np.isin(pool, favored)] = 1.0 + strength
    probabilities = weights / weights.sum()

    day_before = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    rows = []
    for date in next_draw_dates(day_before, n_draws):
        main = rng.choice(pool, size=MAIN_BALLS_DRAWN, replace=False, p=probabilities)
        super_ball = rng.integers(SUPER_BALL_RANGE[0], SUPER_BALL_RANGE[1] + 1)
        rows.append({
            "Date": date.strftime("%d/%m/%Y"),
            "Ball": "-".join(str(n) for n in [*main, super_ball]),
        })
    return pd.DataFrame(rows)


def load_biased_and_preprocess(**kwargs):
    """Same return shape as utils.processor.load_and_preprocess, for drop-in use."""
    return preprocess_draws(biased_draws(**kwargs))


def measured_favored_share(balls_expanded, favored=DEFAULT_FAVORED):
    """What share of drawn main balls the favoured numbers actually took.

    Reported alongside `strength` because the weight is an input knob with no
    intuitive meaning, while "these 3 numbers took 9.4% of all balls instead of
    the 7.0% uniform share" says how big the planted effect really is. Measured
    rather than derived: weighted sampling without replacement has no tidy
    closed form for the marginal share.
    """
    n_columns = balls_expanded.shape[1]
    pooled = balls_expanded.iloc[:, list(main_positions(n_columns))].to_numpy().ravel()
    return {
        "favored_share": float(np.isin(pooled, favored).mean()),
        "uniform_share": len(favored) / MAIN_POOL,
    }


def _detect_pooled(df, balls_expanded, alpha, **kwargs):
    n_columns = balls_expanded.shape[1]
    result = pooled_uniformity_test(balls_expanded, main_positions(n_columns))
    return result["p_value"], bool(result["p_value"] < alpha)


def _strategy_detector(strategy):
    def detect(df, balls_expanded, alpha, seed=0, **kwargs):
        result = evaluate_strategy(strategy, df, balls_expanded, seed=seed, **kwargs)
        p = result["p_value_better_than_chance"]
        return p, bool(pd.notna(p) and p < alpha)
    return detect


DETECTORS = {
    "pooled": _detect_pooled,
    "hot": _strategy_detector("hot"),
    "random": _strategy_detector("random"),
}


def detection_rate(strength, detector="pooled", n_draws=500, n_seeds=10, alpha=DEFAULT_ALPHA,
                   favored=DEFAULT_FAVORED, **strategy_kwargs):
    """Fraction of independent runs in which this detector fires at the given bias.

    Each seed regenerates the data as well as the tickets, so the repetitions
    are independent experiments rather than re-rolls of one dataset.
    """
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector {detector!r}. Available: {sorted(DETECTORS)}")

    detect = DETECTORS[detector]
    flags, p_values, shares = 0, [], []
    for seed in range(n_seeds):
        df, balls_expanded = load_biased_and_preprocess(
            n_draws=n_draws, favored=favored, strength=strength, seed=seed)
        shares.append(measured_favored_share(balls_expanded, favored)["favored_share"])
        p, fired = detect(df, balls_expanded, alpha, seed=seed, **strategy_kwargs)
        p_values.append(p)
        flags += fired

    return {
        "detector": detector,
        "strength": strength,
        "favored_share": float(np.mean(shares)),
        "uniform_share": len(favored) / MAIN_POOL,
        "n_seeds": n_seeds,
        "times_detected": flags,
        "detection_rate": flags / n_seeds,
        "median_p_value": float(np.nanmedian(p_values)),
    }


def sensitivity_report(strengths=DEFAULT_STRENGTHS, detectors=("pooled", "hot", "random"),
                       n_draws=500, n_seeds=10, alpha=DEFAULT_ALPHA, favored=DEFAULT_FAVORED,
                       **strategy_kwargs):
    """The whole grid: one row per (bias strength, detector).

    Read the `strength = 0` rows first. If any detector fires much more often
    than alpha there, it is broken, and every other row in the table is
    meaningless. Only once those look right does a rising detection rate down
    the column mean the pipeline can hear a real signal.
    """
    rows = [
        detection_rate(strength, detector=detector, n_draws=n_draws, n_seeds=n_seeds,
                       alpha=alpha, favored=favored, **strategy_kwargs)
        for strength in strengths
        for detector in detectors
    ]
    return pd.DataFrame(rows)


def sensitivity_threshold(report, detector, target_rate=0.80):
    """Smallest tested bias at which `detector` reaches `target_rate`, or None.

    None is a real answer, not a missing one: it means nothing in the tested
    range was reliably detectable, so any null result from this detector only
    rules out edges larger than the strongest bias tried.
    """
    rows = report[(report["detector"] == detector) & (report["detection_rate"] >= target_rate)]
    if rows.empty:
        return None
    best = rows.loc[rows["strength"].idxmin()]
    return {
        "strength": float(best["strength"]),
        "favored_share": float(best["favored_share"]),
        "uniform_share": float(best["uniform_share"]),
        "detection_rate": float(best["detection_rate"]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-draws", type=int, default=500)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--detectors", default="pooled,hot,random",
                        help=f"comma-separated, from {sorted(DETECTORS)}")
    parser.add_argument("--draws-back", type=int, default=200,
                        help="draws the ticket strategies are evaluated over")
    parser.add_argument("--tickets-per-draw", type=int, default=5)
    args = parser.parse_args()

    report = sensitivity_report(
        detectors=tuple(d.strip() for d in args.detectors.split(",")),
        n_draws=args.n_draws, n_seeds=args.n_seeds, alpha=args.alpha,
        n_draws_back=args.draws_back, tickets_per_draw=args.tickets_per_draw,
    )

    pd.set_option("display.width", 140)
    print(f"\n=== Detection rates over {args.n_seeds} seeds, {args.n_draws} draws each ===")
    print(report.to_string(index=False, formatters={
        "favored_share": "{:.2%}".format, "uniform_share": "{:.2%}".format,
        "detection_rate": "{:.0%}".format, "median_p_value": "{:.4f}".format,
    }))

    print(f"\nControl (strength = 0) should sit near alpha = {args.alpha:g} for every detector.")
    for detector in report["detector"].unique():
        threshold = sensitivity_threshold(report, detector)
        if threshold is None:
            print(f"  {detector}: never reached 80% detection in the tested range — a null result "
                  "from it rules out nothing smaller than the strongest bias tried.")
        else:
            print(f"  {detector}: reaches 80% detection at strength {threshold['strength']:g} "
                  f"({threshold['favored_share']:.2%} of balls vs {threshold['uniform_share']:.2%} "
                  "uniform).")
    report.to_csv("sensitivity_report.csv", index=False)
