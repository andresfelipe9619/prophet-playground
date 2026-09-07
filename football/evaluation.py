"""Whether a football model beats the market — the domain half of the core contract.

`core/significance.py` supplies the arithmetic: a one-sided z-test of a sum of
independent per-observation scores against a null, with the effect size and its
interval, plus the naive and Bonferroni-corrected verdicts. What this module
supplies is the null.

In the lottery the null is the exact hypergeometric hit rate. In football it is
the market: for each held-out match, score the model's probability vector and
the market's with the same proper scoring rule, take the per-match difference
(market score minus model score, so positive means the model did better), and
test whether its mean is greater than zero. That is a paired comparison —
every match is scored by both — which is why the null mean is exactly 0 and the
null variance is the sample variance of the differences.

Only the one-sided p-value may back a "beats the market" claim: a model
significantly *worse* than the market also gets a small two-sided p-value.
"""

import numpy as np

from core.significance import bonferroni_threshold, verdicts, z_test_against_null
from football.scoring import per_match_scores

_EMPTY = {
    "n_observations": 0, "model_score": float("nan"), "market_score": float("nan"),
    "skill_score": float("nan"), "z": float("nan"), "p_value": float("nan"),
    "p_value_greater": float("nan"), "effect": float("nan"), "ci_low": float("nan"),
    "ci_high": float("nan"), "relative_effect": float("nan"),
    "observed_mean": float("nan"), "null_mean": 0.0,
}


def beats_market_test(model_probs, market_probs, outcomes,
                      metric="rps", alpha=0.05, n_comparisons=1):
    """Paired proper-score test of a model against the market.

    `model_probs` and `market_probs` are `(n, 3)` in OUTCOMES order; `outcomes`
    is the actual results. Rows where either side is NaN are dropped from both.
    """
    model_probs = np.asarray(model_probs, dtype=float).reshape(-1, 3)
    market_probs = np.asarray(market_probs, dtype=float).reshape(-1, 3)
    outcomes = np.asarray(list(outcomes))
    threshold = bonferroni_threshold(alpha, n_comparisons)

    keep = ~(np.isnan(model_probs).any(axis=1) | np.isnan(market_probs).any(axis=1))
    model_probs, market_probs, outcomes = model_probs[keep], market_probs[keep], outcomes[keep]

    if len(outcomes) == 0:
        return {"metric": metric, "bonferroni_threshold": threshold,
                "beats_market": False, "beats_market_corrected": False, **_EMPTY}

    model_s = per_match_scores(model_probs, outcomes, metric)
    market_s = per_match_scores(market_probs, outcomes, metric)
    diff = market_s - model_s  # > 0 => model scored lower (better) on that match

    variance = float(np.var(diff, ddof=1)) if len(diff) > 1 else 0.0
    result = z_test_against_null(diff, null_means=0.0, null_variances=variance)

    # Ensure effect is computable even when variance is 0 (tied outcome)
    if np.isnan(result["effect"]):
        result["effect"] = float(diff.mean())

    v = verdicts(result["p_value_greater"], alpha, threshold)
    mean_model, mean_market = float(model_s.mean()), float(market_s.mean())
    return {
        "metric": metric,
        "model_score": mean_model,
        "market_score": mean_market,
        "skill_score": (1.0 - mean_model / mean_market) if mean_market else float("nan"),
        "bonferroni_threshold": threshold,
        "beats_market": v["beats_chance"],
        "beats_market_corrected": v["beats_chance_corrected"],
        **{k: val for k, val in result.items() if k != "null_mean"},
        "null_mean": 0.0,
    }
