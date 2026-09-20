"""Start-list forecasts recorded before the race, scored against the ranking.

Cycling's half of `core/registry.py`, built the same way football's is: the
refusals and the storage are `core/`'s, and this supplies what a prediction is
here and how to score one.

**A cycling prediction is an ordering, so it is stored as worths and scored by
the Plackett-Luce log score** — the only rule proper over a whole finishing
order, and `cycling/scoring.py`'s verdict. A registry that stored "who I think
wins" would be recording the easy, nearly uninformative part; the worths carry
what the forecast actually claimed about everybody.

**And it is scored against the pre-race ranking, not against nothing.** Each
row records its own baseline score at scoring time, so what accumulates is the
difference — the same quantity `evaluation.py:beats_baseline_test` accumulates
in a walk-forward, in the same units. A uniform draw over the start list is not
a baseline and is not accepted as one here either; the ranking baseline comes
from `cycling/baseline.py:form_worths`, built strictly from results before the
race, exactly as the walk-forward builds it.

**The start list is part of the prediction.** A forecast made over 180 riders
and scored over the 140 who started is a different forecast, so the riders and
their worths are stored together and the scorer refuses a race whose finishers
are not the field that was predicted. That refusal is the registry's version of
the rule already enforced in scoring: non-finishers stay in the denominator,
and quietly shrinking the field turns "predict the finishing order" into the
strictly easier "predict the order among those who finished".
"""

import json

import numpy as np
import pandas as pd

from core import registry as core_registry
from core.registry import RegistryError, RegistrySchema
from cycling.scoring import DEFAULT_METRIC, race_score

DEFAULT_REGISTRY_PATH = "cycling_predictions.csv"

SCHEMA = RegistrySchema(
    event_column="race_date",
    # `riders` and `worths` are JSON lists rather than separate rows, because a
    # prediction over a start list is one prediction: splitting it across 180
    # rows would let half of it be scored and the other half not, which is the
    # subset scoring core/registry.py refuses.
    prediction_columns=("race", "kind", "stage", "riders", "worths"),
    result_columns=("metric", "model_score", "baseline_score", "score_difference"),
)

COLUMNS = list(SCHEMA.columns)

__all__ = ["COLUMNS", "DEFAULT_REGISTRY_PATH", "SCHEMA", "RegistryError",
           "load", "pending", "record", "score_pending", "status", "summary"]


def load(path=DEFAULT_REGISTRY_PATH):
    """Read the registry, or an empty frame with the right columns if it does not exist yet."""
    return core_registry.load(SCHEMA, path)


def record(riders, worths, race_date, race, label, kind="stage", stage=None,
           note="", path=DEFAULT_REGISTRY_PATH, now=None):
    """Register one start-list forecast for a race that has not been run.

    `riders` and `worths` are parallel and must stay so — a forecast whose
    worths do not line up with its start list is not recoverable afterwards,
    since nothing in the stored row would reveal the misalignment.
    """
    riders = list(riders)
    worths = np.asarray(worths, dtype=float).reshape(-1)
    if len(riders) != worths.size:
        raise RegistryError(
            f"{len(riders)} riders against {worths.size} worths. A forecast whose worths do "
            "not line up with its start list cannot be recovered later — nothing in the stored "
            "row would reveal which way they slipped."
        )
    if not riders:
        raise RegistryError("An empty start list is not a forecast.")
    if len(set(riders)) != len(riders):
        raise RegistryError("The same rider appears twice in the start list.")
    if not np.isfinite(worths).all() or (worths <= 0).any():
        raise RegistryError(
            "Worths must be finite and strictly positive. A worth of zero takes a logarithm "
            "to minus infinity in the Plackett-Luce score, which is why plackett_luce.py "
            "shrinks toward the field rather than allowing one."
        )

    prediction = {
        "race": race,
        "kind": kind,
        "stage": stage if stage is not None else pd.NA,
        "riders": json.dumps(riders),
        "worths": json.dumps([float(w) for w in worths]),
    }
    return core_registry.record(SCHEMA, prediction, race_date, label,
                                path=path, note=note, now=now)


def score_pending(results, baseline, path=DEFAULT_REGISTRY_PATH, metric=DEFAULT_METRIC):
    """Score every registered race that has since been run.

    `results` is a result frame in `cycling/processor.py`'s contract, with
    non-finishers still in it. `baseline(history, riders, as_of)` produces the
    worths the forecast is measured against — pass
    `cycling/baseline.py:form_worths`, or a market-derived baseline once one
    exists; it is called with results **strictly before** the race, the same
    `as_of` discipline the walk-forward uses.

    A race missing from the frame is left pending. A race present but whose
    field does not match what was predicted is **refused**, not silently
    rescored over the riders who turned up: that is the one way a registry row
    could quietly become a different, easier claim than the one recorded.
    """
    def resolve(row):
        group = results[(results["race"] == row["race"])
                        & (results["kind"] == row["kind"])]
        if pd.notna(row["stage"]):
            group = group[group["stage"].astype(float) == float(row["stage"])]
        if group.empty:
            return None  # not run yet, or not in this frame

        riders = json.loads(row["riders"])
        worths = np.asarray(json.loads(row["worths"]), dtype=float)
        if set(group["rider"]) != set(riders):
            raise RegistryError(
                f"{row['race']} was registered over {len(riders)} riders and the result carries "
                f"{group['rider'].nunique()}. Scoring the forecast over a different field would "
                "make it a different, easier claim than the one recorded — fix the result frame "
                "or leave the row unscored."
            )

        as_of = group["ds"].min()
        history = results[results["ds"] < as_of]
        model = float(race_score(worths, riders, group, metric=metric))
        reference = np.asarray(baseline(history, riders, as_of), dtype=float)
        floor = float(race_score(reference, riders, group, metric=metric))
        return {
            "metric": metric,
            "model_score": model,
            "baseline_score": floor,
            # Lower is better for every metric here, so the model beating the
            # ranking is a positive difference — the same sign convention
            # evaluation.py:beats_baseline_test uses.
            "score_difference": floor - model,
        }

    return core_registry.score_pending(SCHEMA, path, resolve)


def summary(path=DEFAULT_REGISTRY_PATH, registry=None, by_label=False, alpha=0.05):
    """Did the registered forecasts beat the pre-race ranking?

    The same paired one-sided test as `evaluation.py:beats_baseline_test`, on
    per-race scores that were written down before the race. `n_comparisons` is
    the number of labels, so the threshold tightens as more are registered
    against the same races — the correct direction.
    """
    from core.significance import bonferroni_threshold, verdicts, z_test_against_null

    registry = load(path) if registry is None else registry
    scored = registry[registry["score_difference"].notna()]

    columns = ["label", "n_scored", "model_score", "baseline_score", "effect",
               "ci_low", "ci_high", "p_value_greater", "bonferroni_threshold",
               "beats_baseline", "beats_baseline_corrected"]
    if scored.empty:
        return pd.DataFrame(columns=columns)

    groups = list(scored.groupby("label")) if by_label else [("all", scored)]
    threshold = bonferroni_threshold(alpha, max(len(groups), 1))
    rows = []
    for label, group in groups:
        difference = group["score_difference"].astype(float).to_numpy()
        row = {"label": label, "n_scored": len(group),
               "model_score": float(group["model_score"].astype(float).mean()),
               "baseline_score": float(group["baseline_score"].astype(float).mean()),
               "bonferroni_threshold": threshold}
        if difference.size < 2:
            row.update({"effect": float("nan"), "ci_low": float("nan"),
                        "ci_high": float("nan"), "p_value_greater": float("nan"),
                        "beats_baseline": False, "beats_baseline_corrected": False})
        else:
            result = z_test_against_null(difference, null_means=0.0,
                                         null_variances=float(np.var(difference, ddof=1)))
            verdict = verdicts(result["p_value_greater"], alpha, threshold)
            row.update({
                "effect": result["effect"], "ci_low": result["ci_low"],
                "ci_high": result["ci_high"], "p_value_greater": result["p_value_greater"],
                "beats_baseline": verdict["beats_chance"],
                "beats_baseline_corrected": verdict["beats_chance_corrected"],
            })
        rows.append(row)
    return pd.DataFrame(rows)[columns].sort_values("effect", ascending=False).reset_index(drop=True)


def pending(path=DEFAULT_REGISTRY_PATH, registry=None):
    """Forecasts whose race has not been run, or has not been scored yet."""
    return core_registry.pending(SCHEMA, path=path, registry=registry)


def status(path=DEFAULT_REGISTRY_PATH):
    """Counts and dates. The verdict lives in `summary`, which needs races to exist."""
    state = core_registry.status(SCHEMA, path)
    next_race = state.pop("next_event")
    return {**state, "next_race": next_race}
