"""Whether a cycling forecast beats the pre-race ranking — the domain half of the contract.

Cycling's `football/evaluation.py`, down to the arithmetic being borrowed from
`core/significance.py` and only the null being supplied here.

In the lottery the null is the exact hypergeometric hit rate. In football it is
the closing price. In cycling it is **the pre-race ranking**: for each race,
score the forecast and the ranking baseline with the same rule, take the
per-race difference (baseline minus model, so positive means the model did
better), and test whether its mean is greater than zero. Paired, one-sided, and
for the same reason as everywhere else — a model significantly *worse* than the
baseline also earns a small two-sided p-value.

**There is no separate backtest module.** In football one existed because the
expensive part is refitting a goals model inside the window loop. Here the walk
is five lines: race by race in date order, with every forecaster handed only the
results that came before. `walk_forward` is that loop, and keeping it beside the
test is what makes it obvious that the two share an `as_of`.

**The hard part is not the statistics, it is the sample size.** A Grand Tour is
21 scored races and a season of one-day classics is a few dozen. Twenty-odd
paired observations resolve only a large difference, so a null result here is
even more a statement about the sample than it is on the lottery side — which is
why `n_races` travels with every verdict.
"""

import numpy as np
import pandas as pd

from core.significance import bonferroni_threshold, verdicts, z_test_against_null
from cycling.processor import GROUP_KEYS
from cycling.scoring import DEFAULT_METRIC, race_score

_EMPTY = {
    "n_observations": 0, "model_score": float("nan"), "baseline_score": float("nan"),
    "skill_score": float("nan"), "z": float("nan"), "p_value": float("nan"),
    "p_value_greater": float("nan"), "effect": float("nan"), "ci_low": float("nan"),
    "ci_high": float("nan"), "relative_effect": float("nan"),
    "observed_mean": float("nan"), "null_mean": 0.0,
}


def beats_baseline_test(model_scores, baseline_scores, metric=DEFAULT_METRIC,
                        alpha=0.05, n_comparisons=1):
    """Paired one-sided test of a forecast against the ranking baseline.

    Both arguments are one score per race, already aligned. Races where either
    side is NaN — a stage nobody finished, a forecast that could not be built —
    are dropped from both, so the comparison stays paired.
    """
    model = np.asarray(model_scores, dtype=float)
    baseline = np.asarray(baseline_scores, dtype=float)
    threshold = bonferroni_threshold(alpha, n_comparisons)

    keep = np.isfinite(model) & np.isfinite(baseline)
    model, baseline = model[keep], baseline[keep]

    if len(model) == 0:
        return {"metric": metric, "bonferroni_threshold": threshold, "n_races": 0,
                "beats_baseline": False, "beats_baseline_corrected": False, **_EMPTY}

    difference = baseline - model  # > 0 => the model scored lower, which is better
    variance = float(np.var(difference, ddof=1)) if len(difference) > 1 else 0.0
    result = z_test_against_null(difference, null_means=0.0, null_variances=variance)
    if np.isnan(result["effect"]):
        result["effect"] = float(difference.mean())

    verdict = verdicts(result["p_value_greater"], alpha, threshold)
    mean_model, mean_baseline = float(model.mean()), float(baseline.mean())
    return {
        "metric": metric,
        "n_races": int(len(model)),
        "model_score": mean_model,
        "baseline_score": mean_baseline,
        "skill_score": (1.0 - mean_model / mean_baseline) if mean_baseline else float("nan"),
        "bonferroni_threshold": threshold,
        "beats_baseline": verdict["beats_chance"],
        "beats_baseline_corrected": verdict["beats_chance_corrected"],
        **{k: v for k, v in result.items() if k != "null_mean"},
        "null_mean": 0.0,
    }


def race_groups(results):
    """Every scored unit — one stage, one classic, one classification — in date order.

    Grouping on the same keys the processor uses is deliberate: a stage and the
    general classification of the same race are different scored units, and a
    grouping that merged them would score a day's placing against three weeks of
    accumulated time.
    """
    ordered = results.sort_values("ds")
    groups = [(key, group) for key, group in ordered.groupby(GROUP_KEYS, dropna=False)]
    return sorted(groups, key=lambda item: item[1]["ds"].min())


def walk_forward(results, forecasters, metric=DEFAULT_METRIC, min_history=1):
    """Score each forecaster on each race, using only what came before it.

    `forecasters` maps a name to `f(history, riders, as_of) -> worths`, where
    `history` is every result strictly before the race and `riders` is its start
    list. A forecaster that raises or returns None for a race is scored NaN
    there and, because the test drops a race where either side is NaN, drops out
    of the comparison for that race rather than distorting it.

    Returns a frame with one row per (race, forecaster).
    """
    rows = []
    groups = race_groups(results)
    for position, (key, group) in enumerate(groups):
        if position < min_history:
            continue  # nothing to build a forecast from yet
        as_of = group["ds"].min()
        history = results[results["ds"] < as_of]
        riders = list(group["rider"])

        for name, forecaster in forecasters.items():
            try:
                worths = forecaster(history, riders, as_of)
                score = float("nan") if worths is None else race_score(
                    worths, riders, group, metric=metric)
            except ValueError:
                score = float("nan")
            rows.append({"race": key[0], "kind": key[1], "stage": key[2], "ds": as_of,
                         "forecaster": name, "score": score, "n_riders": len(riders)})

    return pd.DataFrame(rows)


def compare_forecasters(results, forecasters, baseline, metric=DEFAULT_METRIC,
                        alpha=0.05, min_history=1):
    """Walk forward, then test every forecaster against `baseline`, corrected together.

    `n_comparisons` is the number of challengers, so testing three forecasters
    against one baseline faces a threshold of 0.05/3. Read the corrected column,
    for the reason the rest of this project keeps repeating: k challengers give
    k chances for one to clear an uncorrected 5% on luck alone.
    """
    if baseline not in forecasters:
        raise ValueError(
            f"Baseline {baseline!r} is not among the forecasters {sorted(forecasters)}.")

    scores = walk_forward(results, forecasters, metric=metric, min_history=min_history)
    wide = scores.pivot_table(index=["race", "kind", "stage", "ds"], columns="forecaster",
                              values="score", dropna=False)
    challengers = [name for name in forecasters if name != baseline]

    rows = []
    for name in challengers:
        result = beats_baseline_test(wide[name], wide[baseline], metric=metric, alpha=alpha,
                                     n_comparisons=max(len(challengers), 1))
        rows.append({"forecaster": name, "baseline": baseline, **result})
    return pd.DataFrame(rows), scores
