"""Plant an edge of known size, and measure how often the market test finds it.

`power.py` says what the test *should* be able to see, from arithmetic.
This says what it *does* see, by running it. The two together are what turn "no
model beat the closing line" from an assertion into a finding: the first gives
the resolution, the second proves the instrument works at all.

It is `lottery/analysis/sensitivity.py` pointed at football, and the shape is
the same — a strength dial, a detection rate per strength, and a control at
strength 0 that must sit near alpha. What differs is what "planting an edge"
means. The lottery plants a bias in the *draws* and asks whether a detector
notices the data is not uniform. Here the data is fine; what gets planted is
**information in the forecast**.

`football/sample_data.py` computes each match's generative truth exactly and
carries it as `p_true_*`. A forecast blended toward that truth knows something
the market does not, by an amount the blend weight names. Strength 0 is the
market's own vector — a forecast with no information beyond the price, which
must fire at alpha and no more. Strength 1 is the truth itself, which must fire
essentially always. Everything in between measures the slope.

**The answer key is the point and also the trap.** `p_true_*` may be read by
tests and by this module only; anything that reads it outside a measurement has
stopped measuring and started cheating. The same rule `cycling/sample_data.py`
puts on `ability_true`.

**Independent seeds, for the reason the lottery module documents at length.**
The season generator and the blend's own randomness take their streams from
`SeedSequence.spawn`, never from one shared integer. Getting that wrong on the
lottery side manufactured a 17.5% false-positive rate on bias-free data and
cost a full round of investigation aimed at the wrong module. There is no
reason to believe football would be kinder.

Two things this module measured that are worth knowing before reading a report.

**Against a market that prices the truth exactly, the dial is inert — by
construction, not by failure.** At `market_noise = 0` the de-margined price
*is* the truth, so a forecast blended toward the truth is the price again and
every strength detects at the false-positive rate. That is not a broken
harness; it is the domain's own statement that nothing can beat a perfect
market. The default is therefore a book that is good but beatable, because a
sensitivity curve needs an edge to exist before it can measure whether the test
finds it.

**Detection falls as the planted edge grows, past a point.** Measured over 240
matches at `market_noise = 0.3`:

    strength   mean effect        sd      z
        0.05       0.00081   0.00258   4.83
        0.10       0.00159   0.00515   4.77
        0.25       0.00381   0.01281   4.59
        0.50       0.00704   0.02542   4.27
        1.00       0.01177   0.05012   3.62

The effect grows fifteenfold and the spread grows nineteenfold, so the
statistic *shrinks*. A forecast that departs further from the price disagrees
with it on more matches and by more, and the paired difference gets noisier
faster than it gets bigger. **A model that knows a little and hugs the price is
easier to prove right than one that knows more and says so loudly** — which is
`ensemble.py`'s argument for pooling with the market, arriving here from a
completely different direction.
"""

import numpy as np
import pandas as pd

from football.common import PROBABILITY_COLUMNS
from football.evaluation import beats_market_test
from football.market import market_probabilities
from football.processor import preprocess_matches
from football.sample_data import generate_matches

DEFAULT_ALPHA = 0.05

# How blurred the simulated book is. **Not 0**, which would price the truth
# exactly and leave nothing for a forecast to know -- see the module docstring.
# 0.3 is a book that is good but beatable, which is the only setting where
# "can this test find an edge" is a question with an answer.
DEFAULT_MARKET_NOISE = 0.3

# The blend weight on the generative truth. 0 is the market itself -- the
# control -- and 1 is omniscience. The interesting rows are the small ones: a
# real model's information advantage over a closing line, if it has one at all,
# lives down near 0.05.
DEFAULT_STRENGTHS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)

TRUTH_COLUMNS = ("TrueH", "TrueD", "TrueA")


def independent_seeds(seed, n=2):
    """Split one seed into `n` streams that share no random numbers.

    Lifted verbatim in spirit from `lottery/analysis/sensitivity.py`, where the
    docstring explains at length what reusing a single seed across two
    generators did to that module's false-positive rate. The short version: two
    things drawn from `default_rng(seed)` are not independent, and a test whose
    whole job is to detect dependence will find that one.
    """
    return [int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(n)]


def planted_forecast(truth, market, strength, rng=None, noise=0.0):
    """A forecast that knows the truth by `strength` and the price by the rest.

    Linear, deliberately: at 0 it **is** the market vector and must score
    identically, which is the same load-bearing endpoint `ensemble.py` pins for
    its blend. Any detection at strength 0 is the false-positive rate and
    nothing else.

    `noise` blurs the result afterwards, for the case where a caller wants a
    model that is informed *and* imprecise. It defaults to off, because the
    control has to be exactly the market and not approximately it.
    """
    truth = np.asarray(truth, dtype=float).reshape(-1, 3)
    market = np.asarray(market, dtype=float).reshape(-1, 3)
    blended = (1.0 - strength) * market + strength * truth

    if noise:
        rng = rng if rng is not None else np.random.default_rng()
        blended = np.exp(np.log(np.clip(blended, 1e-12, 1.0))
                         + rng.normal(0.0, noise, blended.shape))
    return blended / blended.sum(axis=1, keepdims=True)


def _season(seed, n_teams, market_noise):
    """A played season with the truth and the de-margined market side by side.

    The truth is joined onto the processed frame **on the match keys**, not
    zipped by row order. `preprocess_matches` is entitled to reorder and reindex
    -- that is its business, not this module's -- and taking the raw frame's row
    order to line up with the processed one is a positional assumption that
    happens to hold for the first few rows and then drifts.

    It did drift, on the first version of this module: the "omniscient"
    forecast at strength 1 scored *worse* than the market, because it was the
    truth about a different match. A planted-signal harness that plants the
    signal in the wrong row measures nothing and looks like a broken test,
    which is the failure this project's lottery module already paid for once.
    """
    raw = generate_matches(n_teams=n_teams, seed=seed, market_noise=market_noise)
    matches = market_probabilities(
        preprocess_matches(raw.drop(columns=list(TRUTH_COLUMNS)), validate=False))

    answer_key = pd.DataFrame({
        "ds": pd.to_datetime(raw["Date"], dayfirst=True),
        "home_team": raw["HomeTeam"], "away_team": raw["AwayTeam"],
    })
    for column in TRUTH_COLUMNS:
        answer_key[column] = raw[column].to_numpy(dtype=float)

    joined = matches.merge(answer_key, on=["ds", "home_team", "away_team"], how="inner")
    if len(joined) != len(matches):
        raise RuntimeError(
            f"The answer key joined {len(joined)} of {len(matches)} matches. Every generated "
            "fixture has a truth by construction, so a short join means the keys disagree "
            "and the planted signal would land on the wrong rows."
        )

    truth = joined[list(TRUTH_COLUMNS)].to_numpy(dtype=float)
    truth = truth / truth.sum(axis=1, keepdims=True)
    market = joined[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)

    keep = np.isfinite(market).all(axis=1)
    return truth[keep], market[keep], list(joined.loc[keep, "outcome"])


def detection_rate(strength, n_seasons=20, n_teams=16,
                   market_noise=DEFAULT_MARKET_NOISE, seed=0,
                   alpha=DEFAULT_ALPHA, n_comparisons=1, noise=0.0):
    """How often `beats_market_corrected` fires on a forecast with this much truth in it.

    `market_noise` is how blurred the simulated book is, and it must be above
    zero for this to measure anything: a book that prices the truth exactly
    leaves a forecast nothing to know, so every strength collapses onto the
    control. See the module docstring.

    Returns the rate under both verdicts. The corrected one is the column to
    read, as everywhere else here.
    """
    season_seeds, blend_seeds = independent_seeds(seed, 2)
    season_rng = np.random.default_rng(season_seeds)
    blend_rng = np.random.default_rng(blend_seeds)

    naive = corrected = 0
    effects = []
    for _ in range(n_seasons):
        truth, market, outcomes = _season(
            int(season_rng.integers(0, 2**31 - 1)), n_teams, market_noise)
        forecast = planted_forecast(truth, market, strength, rng=blend_rng, noise=noise)

        result = beats_market_test(forecast, market, outcomes,
                                   alpha=alpha, n_comparisons=n_comparisons)
        naive += bool(result["beats_market"])
        corrected += bool(result["beats_market_corrected"])
        effects.append(result["effect"])

    return {
        "strength": strength,
        "n_seasons": int(n_seasons),
        "detection_rate": naive / n_seasons,
        "detection_rate_corrected": corrected / n_seasons,
        "mean_effect": float(np.nanmean(effects)),
        "alpha": alpha,
        "market_noise": market_noise,
    }


def sensitivity_report(strengths=DEFAULT_STRENGTHS, n_seasons=20, n_teams=16,
                       market_noise=DEFAULT_MARKET_NOISE, seed=0, alpha=DEFAULT_ALPHA,
                       noise=0.0):
    """One row per planted strength. Read the `strength = 0` row first.

    That row is the control: a forecast that is exactly the market, which
    carries no information the price does not already have. Its detection rate
    is the measured false-positive floor, and it must sit near alpha. A control
    firing well above alpha means something is broken — and, on the evidence of
    the lottery module's history, the first place to look is the harness rather
    than the test.
    """
    return pd.DataFrame([
        detection_rate(strength, n_seasons=n_seasons, n_teams=n_teams,
                       market_noise=market_noise, seed=seed + i, alpha=alpha, noise=noise)
        for i, strength in enumerate(strengths)
    ])


def sensitivity_threshold(report, target_rate=0.80, corrected=True):
    """The smallest planted strength detected at least `target_rate` of the time.

    NaN when no row clears it, which is a real answer: it means this much
    football cannot reliably find any of the edges that were tried.
    """
    column = "detection_rate_corrected" if corrected else "detection_rate"
    clearing = report[report[column] >= target_rate]
    return float(clearing["strength"].min()) if len(clearing) else float("nan")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seasons", type=int, default=20)
    parser.add_argument("--n-teams", type=int, default=16)
    parser.add_argument("--market-noise", type=float, default=DEFAULT_MARKET_NOISE,
                        help="how blurred the book is; 0 prices the truth exactly and "
                             "leaves nothing to plant, so the dial goes inert")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = sensitivity_report(n_seasons=args.n_seasons, n_teams=args.n_teams,
                                market_noise=args.market_noise, seed=args.seed)
    print(report.to_string(index=False, formatters={
        "detection_rate": "{:.0%}".format,
        "detection_rate_corrected": "{:.0%}".format,
        "mean_effect": "{:+.4f}".format,
    }))
    control = report[report["strength"] == 0.0]
    if len(control):
        print(f"\nControl (strength 0, a forecast that IS the market): "
              f"{control['detection_rate_corrected'].iloc[0]:.0%} — should be near "
              f"{DEFAULT_ALPHA:.0%}. Well above it means the harness is wrong, not the test.")
    threshold = sensitivity_threshold(report)
    print(f"Smallest strength detected 80% of the time: "
          f"{threshold if np.isfinite(threshold) else 'none of those tried'}")


if __name__ == "__main__":
    main()
