"""Is an observed score distinguishable from the null — and would it survive
being one of several?

Two things live here, and they are the reason a second domain should reuse
this package rather than start over.

**The test is one-sided.** `z_test_against_null` returns both a two-sided
p-value ("differs from the null") and a one-sided one ("beats the null").
Only the one-sided value may back a "beats the baseline" claim: a predictor
significantly *worse* than the null also gets a small two-sided p-value, and
reading that column as the verdict turns a bad model into a good one.

**The comparison is corrected.** Scoring k models against the same held-out
observations gives k chances at a false positive: at alpha = 0.05 with six
models, there is a ~26% chance at least one signal-free model clears the bar.
Every evaluation surface must report the corrected verdict alongside the
naive one and point the reader at the corrected column.

What the domain supplies is the null itself: a mean and variance per
observation. For a lottery those come from the exact hypergeometric
distribution; for a match-outcome model they would come from the market's
implied probabilities. The arithmetic below does not change.
"""

import numpy as np
from scipy.stats import norm

EMPTY_RESULT = {
    "z": np.nan,
    "p_value": np.nan,
    "p_value_greater": np.nan,
    "observed_mean": np.nan,
    "null_mean": np.nan,
}


def z_test_against_null(observed, null_means, null_variances):
    """z-test of a sum of independent scores against the null it is judged on.

    `observed` is one score per held-out observation. `null_means` and
    `null_variances` are the mean and variance of that score under the null,
    either as one value applying to every observation or as a per-observation
    sequence — the latter matters whenever the null shifts between
    observations (in the lottery, when collisions change how many distinct
    numbers a model actually committed to).

    Each observation is an independent trial, so the sum of scores is
    asymptotically normal under the null.
    """
    observed = np.asarray(observed, dtype=float)
    if observed.size == 0:
        return dict(EMPTY_RESULT)

    means = np.broadcast_to(np.asarray(null_means, dtype=float), observed.shape)
    variances = np.broadcast_to(np.asarray(null_variances, dtype=float), observed.shape)

    observed_mean = float(observed.mean())
    null_mean = float(means.mean())
    se_sum = float(np.sqrt(variances.sum()))
    if se_sum == 0:
        return {**EMPTY_RESULT, "observed_mean": observed_mean, "null_mean": null_mean}

    z = float((observed.sum() - means.sum()) / se_sum)
    return {
        "z": z,
        "p_value": float(2 * (1 - norm.cdf(abs(z)))),
        "p_value_greater": float(1 - norm.cdf(z)),
        "observed_mean": observed_mean,
        "null_mean": null_mean,
    }


def bonferroni_threshold(alpha, n_comparisons):
    """The per-test threshold that keeps the family-wise error rate at alpha."""
    return alpha / max(n_comparisons, 1)


def verdicts(p_value, alpha, threshold):
    """The naive and corrected verdicts for one row of a comparison table.

    Returned together, and named the same way everywhere, so that no
    evaluation surface can report one without the other. A missing p-value is
    not a pass.
    """
    decidable = p_value is not None and not (isinstance(p_value, float) and np.isnan(p_value))
    return {
        "beats_chance": bool(p_value < alpha) if decidable else False,
        "bonferroni_threshold": threshold,
        "beats_chance_corrected": bool(p_value < threshold) if decidable else False,
    }
