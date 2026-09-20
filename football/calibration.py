"""Is a forecast's 30% a real 30%? — and can it be made into one?

`scoring.py` returns one number per forecast, and one number cannot separate
two very different failures. A model that ranks matches well but states its
case too strongly, and a model that is appropriately humble about matches it
has no idea about, can post the same RPS. The first is fixable with one
parameter; the second needs a better model. Nothing in a results table tells
them apart.

That distinction is not academic here, because of what sits downstream.
`value.py` turns a probability into a stake, and an over-confident probability
sizes up **precisely when the model is most confidently wrong** — which is what
the quarter-Kelly default is already hedging against without being able to say
so. A calibration curve is the thing that can say so.

Three questions, three answers:

**"Does 30% mean 30%?"** — `reliability_curve` and
`expected_calibration_error`. Bin the forecasts, compare each bin's mean
forecast against how often that outcome actually happened. A sparse bin
returns NaN rather than a point, the same rule
`lottery/analysis/structure.py:goodness_of_fit` follows: three matches in a
bin cannot measure a frequency and drawing them as if they could is how a
reliability diagram invents a story.

**"Is it biased overall?"** — `calibration_in_the_large`, the mean forecast
against the base rate, per outcome, with an interval. Note this test is
**two-sided**, which is a deliberate departure from the rest of this package:
elsewhere only `p_value_greater` may back a claim, because a model worse than
its baseline is not a finding. Here it is. Forecasting home wins at 50% when
they happen 40% of the time and forecasting them at 30% are both
mis-calibration, and a one-sided test would wave one of them through.

**"Can it be fixed?"** — `TemperatureScaler` (one parameter, sharpen or soften
everything) and `IsotonicCalibrator` (a free monotone map per outcome).
Temperature is the honest default: it cannot fit noise because it has nowhere
to put it, and if a model's only problem is over-confidence, one parameter is
the whole repair.

**The calibrator is never fitted on the matches it is then scored on.** That is
the leak this package exists around — `ensemble.py` refuses the same temptation
with its blend weight, and for the same reason. `prequential_calibrate` fits
each row's correction on the rows strictly before it, which is what a forecaster
could actually have done at the time, and leaves the opening rows uncalibrated
rather than borrowing from their own future. A calibrator fitted in-sample
improves every score it touches and means nothing.

One thing this module cannot do, and it matters: **a calibrated model is not a
profitable one.** Calibration is about whether the numbers mean what they say,
not whether they beat the price. A perfectly calibrated forecast that is simply
the market's own vector is perfectly calibrated and worth nothing. The verdict
is still `evaluation.py:beats_market_test`.
"""

import numpy as np

from core.significance import bonferroni_threshold, verdicts, z_test_against_null
from football.common import OUTCOMES, outcome_index
from football.scoring import _as_matrix, _onehot

DEFAULT_BINS = 10

# Below this many forecasts a bin's observed frequency is noise, not a
# measurement: at 3 matches the only values it can take are 0, 1/3, 2/3 and 1,
# none of which says anything about a forecast of 0.27.
MIN_BIN_COUNT = 5

_EPS = 1e-12


def _normalise(probs):
    """Rescale rows to sum to 1, leaving all-zero rows alone rather than dividing by 0."""
    totals = probs.sum(axis=1, keepdims=True)
    safe = np.where(totals > _EPS, totals, 1.0)
    return probs / safe


def _pairs(probs, outcomes, outcome=None):
    """(forecast, happened) pairs — pooled across the three outcomes, or one of them.

    Pooling is the default because the question "does 30% mean 30%" is about
    the number, not about which outcome carried it, and three separate curves
    over the same matches are three chances to read a wiggle as a finding.
    """
    p = _as_matrix(probs)
    actual = _onehot(outcomes)
    if outcome is None:
        return p.reshape(-1), actual.reshape(-1)
    column = outcome_index(outcome)
    return p[:, column], actual[:, column]


def reliability_curve(probs, outcomes, bins=DEFAULT_BINS, outcome=None,
                      min_count=MIN_BIN_COUNT):
    """Forecast against realised frequency, in bins.

    Returns a list of dicts with `bin_low`, `bin_high`, `mean_forecast`,
    `observed_frequency` and `count`. A bin holding fewer than `min_count`
    forecasts reports `observed_frequency` as NaN: it has a count, so the
    reader can see it exists, and no frequency, because it does not have one.

    A perfectly calibrated forecast puts every bin on the diagonal. Bins above
    it are outcomes that happened more often than forecast (under-confident
    there); below it, less often.
    """
    forecast, happened = _pairs(probs, outcomes, outcome)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    # `np.digitize` with right=False puts 1.0 in a bin of its own; fold it back
    # into the last real bin, since a forecast of exactly 1 is still a forecast.
    index = np.clip(np.digitize(forecast, edges[1:-1], right=False), 0, int(bins) - 1)

    rows = []
    for b in range(int(bins)):
        in_bin = index == b
        count = int(in_bin.sum())
        rows.append({
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "count": count,
            "mean_forecast": float(forecast[in_bin].mean()) if count else float("nan"),
            "observed_frequency": (float(happened[in_bin].mean())
                                   if count >= min_count else float("nan")),
        })
    return rows


def expected_calibration_error(probs, outcomes, bins=DEFAULT_BINS, outcome=None,
                               min_count=MIN_BIN_COUNT):
    """Mean |forecast - realised| across bins, weighted by how full each bin is.

    Bins too sparse to measure are dropped from the average **and from its
    weights**, so a model whose mass sits in bins nobody can measure gets a
    small ECE over a small share of its forecasts rather than a flattering one
    over all of them. `coverage` reports that share; read it beside the error.
    """
    curve = reliability_curve(probs, outcomes, bins=bins, outcome=outcome,
                              min_count=min_count)
    usable = [row for row in curve if not np.isnan(row["observed_frequency"])]
    total = sum(row["count"] for row in curve)
    measured = sum(row["count"] for row in usable)
    if not measured:
        return {"ece": float("nan"), "coverage": 0.0, "n_bins_used": 0,
                "n_forecasts": int(total)}

    error = sum(row["count"] * abs(row["mean_forecast"] - row["observed_frequency"])
                for row in usable) / measured
    return {"ece": float(error), "coverage": measured / total if total else 0.0,
            "n_bins_used": len(usable), "n_forecasts": int(total)}


def calibration_in_the_large(probs, outcomes, confidence=0.95, alpha=0.05):
    """Mean forecast against base rate, per outcome, with an interval.

    The coarsest calibration check there is and the one that catches a whole
    model listing to one side. Returns one dict per outcome with
    `mean_forecast`, `base_rate`, the difference and its interval, a
    **two-sided** p-value, and both verdicts.

    Two-sided is the deliberate departure noted in the module docstring:
    everywhere else here only `p_value_greater` may back a claim, because a
    model worse than its baseline is not a finding. Mis-calibration in either
    direction is.

    Both verdicts, because this is three tests over one set of matches and
    three chances for one of them to clear an uncorrected 5% -- the same
    arithmetic that makes six per-position chi-squares on Baloto produce a
    spurious "No" about once a run. `miscalibrated_corrected` is the column to
    read; a single uncorrected verdict here would be the repository's own
    documented bug, reintroduced in a new module.
    """
    p = _as_matrix(probs)
    actual = _onehot(outcomes)

    threshold = bonferroni_threshold(alpha, len(OUTCOMES))
    rows = []
    for i, outcome in enumerate(OUTCOMES):
        forecast = p[:, i]
        # Each match is one Bernoulli trial whose success probability is what
        # the model said. Under perfect calibration the realised indicator has
        # that mean and p(1-p) variance, which is exactly the null core/ wants.
        result = z_test_against_null(actual[:, i], null_means=forecast,
                                     null_variances=forecast * (1.0 - forecast),
                                     confidence=confidence)
        rows.append({
            "outcome": outcome,
            "mean_forecast": float(forecast.mean()),
            "base_rate": float(actual[:, i].mean()),
            "difference": result["effect"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "p_value": result["p_value"],
            "n_observations": result["n_observations"],
            **_renamed(verdicts(result["p_value"], alpha, threshold)),
        })
    return rows


def _renamed(verdict):
    """core/ names its keys `beats_chance*`; this domain is asking a different question."""
    return {"miscalibrated": verdict["beats_chance"],
            "bonferroni_threshold": verdict["bonferroni_threshold"],
            "miscalibrated_corrected": verdict["beats_chance_corrected"]}


# ------------------------------------------------------------------ calibrators
#
# Both expose `.fit(probs, outcomes)` and `.transform(probs)`, so
# `prequential_calibrate` does not care which it was handed.


class TemperatureScaler:
    """One parameter: raise every probability to 1/T and renormalise.

    T > 1 softens a forecast toward the uniform, T < 1 sharpens it, T = 1 is
    the identity. That is the entire model, and its smallness is the point --
    it has nowhere to put noise, so a temperature fitted on a few hundred
    matches is a claim about over-confidence and nothing else. If a model's
    only fault is stating its case too strongly, this fixes it completely; if
    the fix is large, the diagnosis was wrong.
    """

    def __init__(self, temperature=1.0):
        self.temperature = float(temperature)

    @classmethod
    def fit(cls, probs, outcomes, bounds=(0.2, 5.0)):
        from scipy.optimize import minimize_scalar

        p = np.clip(_as_matrix(probs), _EPS, 1.0)
        index = np.array([outcome_index(o) for o in outcomes])
        rows = np.arange(index.size)

        def negative_log_likelihood(log_t):
            scaled = _normalise(p ** (1.0 / np.exp(log_t)))
            return -np.log(np.clip(scaled[rows, index], _EPS, 1.0)).mean()

        # Optimised in logs so the temperature cannot cross zero, the same
        # trick elo.py uses to stop its ordered-logit cut points crossing.
        best = minimize_scalar(negative_log_likelihood, method="bounded",
                               bounds=(np.log(bounds[0]), np.log(bounds[1])))
        return cls(float(np.exp(best.x)))

    def transform(self, probs):
        p = np.clip(_as_matrix(probs), _EPS, 1.0)
        return _normalise(p ** (1.0 / self.temperature))


class IsotonicCalibrator:
    """A free monotone map per outcome, then renormalise.

    Strictly more flexible than a temperature and strictly more able to fit
    noise: with a few hundred matches it will happily learn that forecasts
    between 0.31 and 0.34 never happen. Worth running as the upper bound on
    what recalibration could buy -- if isotonic barely beats temperature, the
    model's problem is not its confidence.

    The per-outcome maps do not preserve the sum, so the renormalisation
    afterwards is not cosmetic; it is what keeps the output a probability
    vector at all.
    """

    def __init__(self, models=None):
        self.models = models

    @classmethod
    def fit(cls, probs, outcomes, out_of_bounds="clip"):
        from sklearn.isotonic import IsotonicRegression

        p = _as_matrix(probs)
        actual = _onehot(outcomes)
        models = []
        for i in range(len(OUTCOMES)):
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=out_of_bounds)
            model.fit(p[:, i], actual[:, i])
            models.append(model)
        return cls(models)

    def transform(self, probs):
        p = _as_matrix(probs)
        mapped = np.column_stack([model.predict(p[:, i]) for i, model in enumerate(self.models)])
        # An all-zero row is possible here and means every outcome was mapped
        # to zero; falling back to the input beats returning a vector that is
        # not a distribution.
        totals = mapped.sum(axis=1, keepdims=True)
        return np.where(totals > _EPS, _normalise(mapped), _normalise(p))


CALIBRATORS = {"temperature": TemperatureScaler, "isotonic": IsotonicCalibrator}


def prequential_calibrate(probs, outcomes, calibrator="temperature", min_fit=50,
                          refit_every=1):
    """Recalibrate each forecast using only the forecasts that came before it.

    This is the gate. At row `t` the correction is fitted on rows `[0, t)` --
    forecasts whose outcomes a forecaster would already have known -- and
    applied to row `t` alone. The first `min_fit` rows pass through unchanged,
    because the alternative is fitting a correction on a handful of matches and
    calling the result out-of-sample.

    `refit_every` refits every k rows instead of every row; the fit is cheap for
    a temperature and not for isotonic. It changes runtime, not the guarantee:
    a correction applied at row `t` was still fitted only on rows before it.

    Returns `(calibrated_probs, n_calibrated)`. A caller comparing calibrated
    against raw must compare on the same rows, so it should score the last
    `n_calibrated` of both -- the leading rows are identical by construction
    and averaging them in dilutes the difference toward zero.
    """
    factory = CALIBRATORS[calibrator] if isinstance(calibrator, str) else calibrator
    p = _as_matrix(probs)
    outcomes = list(outcomes)
    if len(outcomes) != len(p):
        raise ValueError(f"{len(p)} forecasts against {len(outcomes)} outcomes.")

    out = p.copy()
    fitted = None
    for t in range(len(p)):
        if t < min_fit:
            continue
        if fitted is None or (t - min_fit) % max(int(refit_every), 1) == 0:
            fitted = factory.fit(p[:t], outcomes[:t])
        out[t] = fitted.transform(p[t])[0]

    return out, max(len(p) - min_fit, 0)
