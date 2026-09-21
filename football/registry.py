"""Forecasts recorded before kick-off, scored against the closing line.

The lottery has had a pre-registration log for a long time, in the one domain
where everybody already knows the answer is no. Football is where a forward
record would actually mean something, and it had none. This is that record; the
refusals and the storage are `core/registry.py`'s, and what this module adds is
the football half.

**A registered football prediction is a forward market test, not a forward
accuracy number.** Scoring a 1X2 vector by how often it was "right" says
nothing — a forecast that backs the favourite every week is right about half
the time and has no edge whatsoever. So a row is scored the way
`evaluation.py` scores a backtest: the RPS of the forecast and the RPS of the
**de-margined closing price on the same match**, and the difference between
them. That difference is the thing worth accumulating.

The closing price is supplied at scoring time and not at recording time, which
is the point. A forecast is registered against a fixture before kick-off, when
the closing line does not exist yet; the bar it will be judged against is
therefore also unfalsifiable at the moment of writing.

`summary` reports the mean difference through the same one-sided test the
backtest uses, so a registry and a backtest of the same model are answering the
same question in the same units — and a registry that disagrees with its
backtest is the most interesting result this project could produce.

One caveat travels with every number here: a season of registered forecasts is
a few hundred matches, and `beats_market_test` needs thousands to separate a
real 2% edge from nothing. That is why the summary carries `n_scored` beside
the verdict, and why `clv.py` exists as the measurement that converges first.
"""

import numpy as np
import pandas as pd

from core import registry as core_registry
from core.registry import RegistryError, RegistrySchema
from football.common import OUTCOMES, PROBABILITY_COLUMNS
from football.power import minimum_detectable_edge
from football.scoring import METRICS, per_match_scores

DEFAULT_REGISTRY_PATH = "football_predictions.csv"

SCHEMA = RegistrySchema(
    event_column="match_date",
    # The fixture, then the forecast. Stored as three columns rather than one
    # packed string because the one thing anyone will want to do with an old
    # row is score it again under a different metric.
    prediction_columns=("home_team", "away_team", *PROBABILITY_COLUMNS),
    result_columns=("outcome", "model_score", "market_score", "score_difference"),
)

COLUMNS = list(SCHEMA.columns)

__all__ = ["COLUMNS", "DEFAULT_REGISTRY_PATH", "SCHEMA", "RegistryError",
           "load", "pending", "record", "score_pending", "summary"]


def load(path=DEFAULT_REGISTRY_PATH):
    """Read the registry, or an empty frame with the right columns if it does not exist yet."""
    return core_registry.load(SCHEMA, path)


def record(probabilities, match_date, home_team, away_team, label,
           note="", path=DEFAULT_REGISTRY_PATH, now=None):
    """Register one 1X2 forecast for a fixture that has not kicked off.

    `probabilities` is `(p_home, p_draw, p_away)` in `OUTCOMES` order — the one
    ordering this package never re-derives, because transposing two of them is
    a bug no range check can catch.

    `now` is injectable so the refusal can be tested; leave it alone in normal
    use, where letting the caller choose "now" would defeat the point.
    """
    vector = np.asarray(probabilities, dtype=float).reshape(-1)
    if vector.size != len(OUTCOMES):
        raise RegistryError(
            f"Expected {len(OUTCOMES)} probabilities in {OUTCOMES} order, got {vector.size}.")
    if not np.isfinite(vector).all() or (vector < 0).any():
        raise RegistryError("A forecast must be finite and non-negative.")
    total = vector.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        raise RegistryError(
            f"Probabilities sum to {total:.4f}, not 1. An unnormalised forecast scores "
            "differently from the one that was meant, and the registry cannot tell which "
            "was intended after the fact."
        )

    prediction = {"home_team": home_team, "away_team": away_team}
    prediction.update(dict(zip(PROBABILITY_COLUMNS, vector, strict=True)))
    return core_registry.record(SCHEMA, prediction, match_date, label,
                                path=path, note=note, now=now)


def score_pending(results, path=DEFAULT_REGISTRY_PATH, metric="rps"):
    """Score every registered fixture that has since been played.

    `results` is a match frame carrying the outcome **and** the de-margined
    market probabilities (`football/market.py:market_probabilities`). A row
    whose fixture is not in the frame, or whose market vector is missing, is
    left pending rather than scored against nothing: a forecast with no bar
    beside it is the football version of a bare model number, and the whole
    point of this file is that the bar was fixed in advance.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Expected one of {list(METRICS)}.")

    missing = [c for c in ("ds", "home_team", "away_team", "outcome") if c not in results.columns]
    if missing:
        raise ValueError(f"The results frame is missing {missing}.")
    if not set(PROBABILITY_COLUMNS) <= set(results.columns):
        raise ValueError(
            "The results frame carries no market probabilities. Pass it through "
            "football/market.py:market_probabilities first — a registered forecast is scored "
            "against the closing price, not on its own."
        )

    played = {
        (pd.Timestamp(row.ds).normalize(), row.home_team, row.away_team): row
        for row in results.itertuples()
    }

    def resolve(row):
        match = played.get((row["match_date"], row["home_team"], row["away_team"]))
        if match is None:
            return None  # not played yet, or not in this frame

        market = np.array([getattr(match, column) for column in PROBABILITY_COLUMNS], dtype=float)
        if not np.isfinite(market).all():
            return None  # played, but with no usable price: no bar, so no score

        forecast = np.array([row[column] for column in PROBABILITY_COLUMNS], dtype=float)
        outcomes = [match.outcome]
        model = float(per_match_scores(forecast, outcomes, metric)[0])
        baseline = float(per_match_scores(market, outcomes, metric)[0])
        return {
            "outcome": match.outcome,
            "model_score": model,
            "market_score": baseline,
            # Positive means the model scored lower (better) than the price on
            # this match — the same sign convention evaluation.py uses.
            "score_difference": baseline - model,
        }

    return core_registry.score_pending(SCHEMA, path, resolve)


def summary(path=DEFAULT_REGISTRY_PATH, registry=None, by_label=False, alpha=0.05):
    """Did the registered forecasts beat the closing line?

    The same one-sided paired test `evaluation.py` runs on a backtest, on rows
    that were written down before anyone knew. `n_comparisons` is the number of
    labels scored together, because scoring three models against one set of
    fixtures gives three chances at an uncorrected 5%.

    Every row carries `min_detectable_edge`, computed from the **spread this
    registry's own rows actually show** rather than from a reference. A null
    result here without it is unreadable: a season of registered forecasts is a
    few hundred matches, and at that size the smallest RPS improvement the test
    can find is several times what a good model takes out of a closing line. So
    "did not beat the market" is usually a statement about how much football has
    happened, and the column is what says which.
    """
    registry = load(path) if registry is None else registry
    scored = registry[registry["score_difference"].notna()]

    columns = ["label", "n_scored", "model_score", "market_score", "effect",
               "ci_low", "ci_high", "p_value_greater", "bonferroni_threshold",
               "beats_market", "beats_market_corrected", "min_detectable_edge"]
    if scored.empty:
        return pd.DataFrame(columns=columns)

    groups = list(scored.groupby("label")) if by_label else [("all", scored)]
    rows = []
    for label, group in groups:
        # The per-match scores are what the file holds, so the paired test runs
        # on them directly. Re-scoring the stored probability vectors here would
        # let a change of metric silently disagree with what was written down,
        # which is the one thing a pre-registration file may never do.
        model_scores = group["model_score"].astype(float).to_numpy()
        market_scores = group["market_score"].astype(float).to_numpy()

        result = _paired(model_scores, market_scores, alpha=alpha,
                         n_comparisons=max(len(groups), 1))
        # The spread these rows actually show, not a reference: the resolution
        # of a verdict is a property of the run that produced it.
        spread = float(np.std(market_scores - model_scores, ddof=1)) if len(group) > 1 else 0.0
        result["min_detectable_edge"] = (
            minimum_detectable_edge(len(group), score_sd=spread)["absolute"]
            if spread > 0 else float("nan"))
        rows.append({"label": label, "n_scored": len(group),
                     "model_score": float(model_scores.mean()),
                     "market_score": float(market_scores.mean()),
                     **{key: result[key] for key in columns[4:]}})
    return pd.DataFrame(rows).sort_values("effect", ascending=False).reset_index(drop=True)


def _paired(model_scores, market_scores, alpha, n_comparisons):
    """The paired one-sided test, on scores that are already computed.

    `evaluation.py:beats_market_test` takes probability vectors because that is
    what a backtest has in hand; it then forms exactly this difference and hands
    it to `core/significance.py`. This takes the same last step, so a registry
    verdict and a backtest verdict are the same test in the same units and can
    be put side by side — which is the whole reason for keeping the per-match
    scores in the file rather than only the vectors.
    """
    from core.significance import bonferroni_threshold, verdicts, z_test_against_null

    difference = np.asarray(market_scores, dtype=float) - np.asarray(model_scores, dtype=float)
    threshold = bonferroni_threshold(alpha, n_comparisons)
    if difference.size < 2:
        return {"effect": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value_greater": float("nan"), "bonferroni_threshold": threshold,
                "beats_market": False, "beats_market_corrected": False}

    result = z_test_against_null(difference, null_means=0.0,
                                 null_variances=float(np.var(difference, ddof=1)))
    verdict = verdicts(result["p_value_greater"], alpha, threshold)
    return {
        "effect": result["effect"], "ci_low": result["ci_low"], "ci_high": result["ci_high"],
        "p_value_greater": result["p_value_greater"], "bonferroni_threshold": threshold,
        "beats_market": verdict["beats_chance"],
        "beats_market_corrected": verdict["beats_chance_corrected"],
    }


def pending(path=DEFAULT_REGISTRY_PATH, registry=None):
    """Forecasts whose fixture has not been played, or has not been scored yet."""
    return core_registry.pending(SCHEMA, path=path, registry=registry)


def status(path=DEFAULT_REGISTRY_PATH):
    """Counts and dates, with the caveat this domain has to attach.

    A season of registered forecasts is a few hundred matches and
    `beats_market_test` needs thousands, so `n_scored` is the number to read
    first — a null result below it is a statement about the sample size.
    """
    state = core_registry.status(SCHEMA, path)
    next_match = state.pop("next_event")
    return {**state, "next_match": next_match}
