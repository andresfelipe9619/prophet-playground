"""Walk-forward evaluation of the football models against the closing line.

The football counterpart of lottery/backtest.py, and it shares the shape on
purpose. `run_all` holds out the last N matches and refits before each one
(expanding window, one step ahead). `run_holdout` holds out everything after a
date, either refitting per match (`expanding`) or fitting once at the cutoff
(`frozen`) — the frozen mode is the fast, concrete "train in March, predict the
rest of the season" run. `compare_models` runs several models over the *same*
held-out matches and returns one row each.

Every path ends in the same place: `football.evaluation.beats_market_test` over
the (model probs, market probs, outcome) triples collected, so the summaries are
directly comparable. A window whose held-out fixture involves a team *any*
requested model never saw is skipped for all of them, never scored — that is
what keeps a multi-model comparison paired, since two models scored on different
matches are not comparable at all. It mirrors the lottery skipping a window a
model could not predict.

**Multiple models means the corrected verdict is the one that counts.**
`compare_models` passes `n_comparisons = len(models)` into every test it runs,
so three models scored against the same matches face a threshold of 0.05/3. A
table that reported one uncorrected verdict per model would reintroduce exactly
the bug the lottery side is built around.

Fitting Dixon-Coles per window is the slow part — Elo is a single pass and the
blend is arithmetic on probabilities already computed, so adding them to a run
costs almost nothing. `--cutoff ... --mode frozen` avoids the refit loop.
"""

import argparse
import os

import numpy as np
import pandas as pd

from core.manifest import data_fingerprint, run_manifest
from core.windows import cutoff_bounds, window_bounds
from football.calibration import CALIBRATORS, prequential_calibrate
from football.common import ODDS_COLUMNS, PROBABILITY_COLUMNS, UnknownTeamError
from football.dixon_coles import DixonColes
from football.elo import Elo
from football.ensemble import POOLS, blend
from football.evaluation import beats_market_test
from football.market import market_probabilities

MIN_TRAIN = 100

# The models this module can score. "blend" is not a third fit: it is
# Dixon-Coles pooled with the market, which is why it is derived after
# collection rather than fitted inside the window loop.
BASE_MODELS = ("dixon_coles", "elo")
MODEL_NAMES = ("dixon_coles", "elo", "blend")
DEFAULT_BLEND_WEIGHT = 0.5

_FITTERS = {
    "dixon_coles": lambda train, half_life: DixonColes.fit(train, half_life=half_life),
    "elo": lambda train, half_life: Elo.fit(train),
}


def _market_row_probs(match_row, method):
    """Market probability vector for one match row, NaN if the row has no price."""
    frame = pd.DataFrame([match_row])
    if frame[list(ODDS_COLUMNS)].isna().to_numpy().any():
        return np.array([np.nan, np.nan, np.nan])
    out = market_probabilities(frame, method=method)
    return out[list(PROBABILITY_COLUMNS)].to_numpy()[0]


def _score(model_probs, market_probs, outcomes, metric, n_comparisons=1):
    return beats_market_test(np.array(model_probs), np.array(market_probs), outcomes,
                             metric=metric, n_comparisons=n_comparisons)


def _base_models(models):
    """Which models actually need fitting for this request."""
    needed = [name for name in models if name in BASE_MODELS]
    if "blend" in models and "dixon_coles" not in needed:
        needed.append("dixon_coles")  # the blend's model half
    return needed


def _collect(train_for, matches, indices, half_life, method, models=("dixon_coles",),
             frozen=None):
    """Fit-and-predict over `indices`.

    Returns `(probs_by_model, market_probs, outcomes, odds, skipped)`. A window
    is skipped when *any* fitted model cannot predict it, so every model in the
    call ends up scored on an identical set of matches.

    The **raw** price triple of each scored window is collected alongside the
    de-margined one. It is not used for scoring — a model is judged against the
    de-margined price — but a staking surface needs the price actually paid, and
    reconstructing "the last N priced matches" downstream does not reproduce
    this set: windows with no price are scored here (as NaN) and windows an
    unknown team skipped are not, so the two can be the same length and line up
    row-for-row with the wrong matches.
    """
    needed = _base_models(models)
    probs = {name: [] for name in needed}
    market_probs, outcomes, odds = [], [], []
    skipped = 0

    for t in indices:
        test = matches.iloc[t]
        try:
            fitted = frozen if frozen is not None else {
                name: _FITTERS[name](train_for(t), half_life) for name in needed}
            row = {name: fitted[name].predict_outcome(test["home_team"], test["away_team"])
                   for name in needed}
        except UnknownTeamError:
            skipped += 1
            continue
        for name, vector in row.items():
            probs[name].append(vector)
        market_probs.append(_market_row_probs(test, method))
        outcomes.append(test["outcome"])
        odds.append(test[list(ODDS_COLUMNS)].to_numpy(dtype=float))

    return probs, market_probs, outcomes, odds, skipped


def _with_blend(probs, market_probs, models, weight, pool):
    """Derive the blended series from the already-collected Dixon-Coles one."""
    if "blend" not in models:
        return probs
    if not probs.get("dixon_coles"):
        return {**probs, "blend": []}
    blended = blend(np.array(probs["dixon_coles"]), np.array(market_probs),
                    weight=weight, pool=pool)
    return {**probs, "blend": list(blended)}


DEFAULT_CALIBRATION_MIN_FIT = 50


def _recalibrate(probs, market_probs, outcomes, odds, models, calibrate, min_fit):
    """Apply a prequentially-fitted recalibration, then trim every series to match.

    The correction for each forecast is fitted on the forecasts that came before
    it and on nothing else -- see `football/calibration.py`, where the gate and
    the reason for it live.

    The trim is the part that is easy to get wrong. The leading `min_fit`
    forecasts pass through a recalibrator uncalibrated, so scoring the whole
    series mixes rows where the correction applied with rows where it could not
    and pulls any difference toward zero. Every model is trimmed by the same
    amount, along with the market and the outcomes, so the comparison stays on
    one identical set of matches -- the rule `compare_models` already follows
    for a window a model cannot predict. The raw prices are trimmed with them:
    they are not scored, but a staking surface reads them row-for-row against
    the forecasts, and a trim that missed them would shift every bet onto
    another match's price.
    """
    calibrated, kept = {}, None
    for name in models:
        series = probs[name]
        if not series:
            calibrated[name] = series
            continue
        adjusted, n = prequential_calibrate(np.array(series), outcomes,
                                            calibrator=calibrate, min_fit=min_fit)
        calibrated[name] = list(adjusted)
        kept = n if kept is None else min(kept, n)

    if not kept:
        return calibrated, market_probs, outcomes, odds
    return ({name: series[-kept:] if series else series
             for name, series in calibrated.items()},
            market_probs[-kept:], outcomes[-kept:], odds[-kept:])


def _summarise(probs, market_probs, outcomes, models, metric, skipped, mode, method,
               half_life, weight, pool, calibrate=None, calibrate_min_fit=None):
    """One row per model, all sharing the same Bonferroni threshold.

    `calibrate` is a column rather than only a manifest field: two runs over the
    same matches that disagree are not comparable unless what differed between
    them is on the table someone reads.
    """
    rows = []
    for name in models:
        result = _score(probs[name], market_probs, outcomes, metric,
                        n_comparisons=len(models))
        rows.append({"model": name, "n_windows_scored": len(outcomes),
                     "n_windows_skipped": skipped, "mode": mode, "method": method,
                     "half_life": half_life,
                     "blend_weight": weight if name == "blend" else None,
                     "pool": pool if name == "blend" else None,
                     "calibrate": calibrate,
                     "calibrate_min_fit": calibrate_min_fit if calibrate else None,
                     **result})
    return pd.DataFrame(rows)


def compare_models(matches, n_windows=30, min_train=MIN_TRAIN, half_life=None,
                   method="multiplicative", metric="rps", models=MODEL_NAMES,
                   blend_weight=DEFAULT_BLEND_WEIGHT, pool="linear",
                   calibrate=None, calibrate_min_fit=DEFAULT_CALIBRATION_MIN_FIT):
    """Score several models over the same held-out matches, corrected together.

    Returns one row per model. Read `beats_market_corrected`: the threshold is
    already divided by the number of models in the call, because scoring three
    models against one set of matches gives three chances for luck to clear an
    uncorrected 5%.
    """
    models = tuple(models)
    unknown = [name for name in models if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Expected from {list(MODEL_NAMES)}.")
    if pool not in POOLS:
        raise ValueError(f"Unknown pool {pool!r}. Expected one of {sorted(POOLS)}.")

    matches = matches.sort_values("ds").reset_index(drop=True)
    start, total = window_bounds(len(matches), n_windows, min_train)
    probs, market_probs, outcomes, odds, skipped = _collect(
        lambda t: matches.iloc[:t], matches, range(start, total), half_life, method, models)
    probs = _with_blend(probs, market_probs, models, blend_weight, pool)
    if calibrate:
        # After the blend, not before: the blend pools a model with the price,
        # and recalibrating its inputs separately would change what is being
        # pooled rather than how the pool is stated.
        probs, market_probs, outcomes, odds = _recalibrate(
            probs, market_probs, outcomes, odds, models, calibrate, calibrate_min_fit)

    table = _summarise(probs, market_probs, outcomes, models, metric, skipped,
                       "expanding_last_n", method, half_life, blend_weight, pool,
                       calibrate=calibrate, calibrate_min_fit=calibrate_min_fit)
    # The forecasts themselves, so a calibration surface does not have to refit
    # every model a second time to ask a different question of the same run.
    table.attrs["forecasts"] = {
        "market": np.array(market_probs),
        "outcomes": list(outcomes),
        "models": {name: np.array(probs[name]) for name in models if probs.get(name)},
        # The raw price of each scored window, carried rather than looked up
        # again: a caller reconstructing "the last N priced matches" gets a set
        # of the same length made of different matches.
        "odds": np.array(odds, dtype=float),
    }
    table.attrs["manifest"] = run_manifest({
        "data": data_fingerprint(matches), "n_matches": int(len(matches)),
        "n_windows": n_windows, "min_train": min_train, "half_life": half_life,
        "method": method, "metric": metric, "models": list(models),
        "blend_weight": blend_weight, "pool": pool,
        "calibrate": calibrate, "calibrate_min_fit": calibrate_min_fit if calibrate else None,
        "odds_are_closing": matches.attrs.get("odds_are_closing"),
        "odds_source": matches.attrs.get("odds_source"),
    })
    return table


def run_all(matches, n_windows=30, min_train=MIN_TRAIN, half_life=None,
            method="multiplicative", metric="rps", model="dixon_coles"):
    """Hold out the last `n_windows` matches, refitting `model` before each."""
    matches = matches.sort_values("ds").reset_index(drop=True)
    start, total = window_bounds(len(matches), n_windows, min_train)
    probs, market_probs, outcomes, _odds, skipped = _collect(
        lambda t: matches.iloc[:t], matches, range(start, total), half_life, method, (model,))
    probs = _with_blend(probs, market_probs, (model,), DEFAULT_BLEND_WEIGHT, "linear")

    result = _score(probs[model], market_probs, outcomes, metric)
    result.update({"model": model, "n_windows_scored": len(outcomes),
                   "n_windows_skipped": skipped, "mode": "expanding_last_n",
                   "method": method, "half_life": half_life,
                   "manifest": run_manifest({
                       "data": data_fingerprint(matches), "n_matches": int(len(matches)),
                       "n_windows": n_windows, "min_train": min_train,
                       "half_life": half_life, "method": method, "metric": metric,
                       "model": model,
                       "odds_are_closing": matches.attrs.get("odds_are_closing"),
                       "odds_source": matches.attrs.get("odds_source"),
                   })})
    return result


def run_holdout(matches, cutoff, mode="expanding", half_life=None,
                method="multiplicative", metric="rps", model="dixon_coles"):
    """Hold out every match after `cutoff`.

    `mode='expanding'` refits before each held-out match; `mode='frozen'` fits
    once at the cutoff and forecasts the whole remaining horizon.
    """
    if mode not in ("expanding", "frozen"):
        raise ValueError(f"mode must be 'expanding' or 'frozen', got {mode!r}")

    matches = matches.sort_values("ds").reset_index(drop=True)
    n_train, n_holdout = cutoff_bounds(matches["ds"], cutoff)

    frozen = None
    if mode == "frozen":
        train = matches.iloc[:n_train]
        frozen = {name: _FITTERS[name](train, half_life) for name in _base_models((model,))}

    probs, market_probs, outcomes, _odds, skipped = _collect(
        lambda t: matches.iloc[:t], matches, range(n_train, n_train + n_holdout),
        half_life, method, (model,), frozen=frozen)
    probs = _with_blend(probs, market_probs, (model,), DEFAULT_BLEND_WEIGHT, "linear")

    result = _score(probs[model], market_probs, outcomes, metric)
    result.update({"model": model, "n_windows_scored": len(outcomes),
                   "n_windows_skipped": skipped, "mode": mode, "method": method,
                   "half_life": half_life,
                   "manifest": run_manifest({
                       "data": data_fingerprint(matches), "n_matches": int(len(matches)),
                       "cutoff": f"{pd.Timestamp(cutoff):%Y-%m-%d}", "mode": mode,
                       "n_train": n_train, "n_holdout": n_holdout,
                       "half_life": half_life, "method": method, "metric": metric,
                       "model": model,
                       "odds_are_closing": matches.attrs.get("odds_are_closing"),
                       "odds_source": matches.attrs.get("odds_source"),
                   })})
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
    parser.add_argument("--models", default=None,
                        help=f"comma-separated, from {list(MODEL_NAMES)}; scores them together "
                             "with the Bonferroni threshold divided by how many")
    parser.add_argument("--blend-weight", type=float, default=DEFAULT_BLEND_WEIGHT,
                        help="weight on the model in 'blend'; 0 is the market itself")
    parser.add_argument("--pool", choices=tuple(POOLS), default="linear")
    parser.add_argument("--calibrate", choices=tuple(CALIBRATORS), default=None,
                        help="recalibrate each forecast on the forecasts before it "
                             "(prequential, never in-sample); only with --models")
    parser.add_argument("--calibrate-min-fit", type=int, default=DEFAULT_CALIBRATION_MIN_FIT,
                        help="forecasts to accumulate before the correction starts applying; "
                             "the leading ones are dropped from the comparison")
    parser.add_argument("--extra", action="store_true", help="load via the extra-file contract")
    parser.add_argument("--league", default=None, help="league to pick from an --extra file")
    args = parser.parse_args()

    paths = [os.path.join(args.data_dir, name) for name in args.seasons.split(",")]
    if args.extra:
        from football.extra_processor import load_extra
        frames = [load_extra(p, league=args.league) for p in paths]
        matches = pd.concat(frames, ignore_index=True).sort_values("ds").reset_index(drop=True)
    else:
        from football.processor import load_seasons
        matches = load_seasons(paths)

    if args.models:
        table = compare_models(
            matches, n_windows=args.n_windows, min_train=args.min_train,
            half_life=args.half_life, method=args.method, metric=args.metric,
            models=tuple(name.strip() for name in args.models.split(",")),
            blend_weight=args.blend_weight, pool=args.pool,
            calibrate=args.calibrate, calibrate_min_fit=args.calibrate_min_fit)
        columns = ["model", "n_windows_scored", "model_score", "market_score", "skill_score",
                   "effect", "ci_low", "ci_high", "p_value_greater", "bonferroni_threshold",
                   "beats_market", "beats_market_corrected"]
        print(table[columns].to_string(index=False))
        print("\nRead beats_market_corrected, not beats_market: "
              f"{len(table)} models scored against the same matches means "
              f"{len(table)} chances for one of them to clear an uncorrected 5% by luck.")
        if args.calibrate:
            print(f"Recalibrated ({args.calibrate}, fitted only on earlier forecasts). "
                  "A better score here means the model was stating its case wrongly, "
                  "not that it knows more than the price.")
        return

    if args.cutoff:
        result = run_holdout(matches, cutoff=args.cutoff, mode=args.mode,
                             half_life=args.half_life, method=args.method, metric=args.metric)
    else:
        result = run_all(matches, n_windows=args.n_windows, min_train=args.min_train,
                         half_life=args.half_life, method=args.method, metric=args.metric)

    manifest = result.pop("manifest", None)
    for key, value in result.items():
        print(f"{key:>24}: {value}")
    if manifest:
        git = manifest["git"]
        commit = (git["commit"] or "unknown")[:8]
        dirty = " (dirty tree — not reproducible from any commit)" if git["dirty"] else ""
        print(f"{'run':>24}: {commit}{dirty} · data {manifest['inputs']['data'][:12]} · "
              f"{manifest['generated_at']}")


if __name__ == "__main__":
    main()
