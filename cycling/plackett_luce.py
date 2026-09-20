"""A rider-strength model fitted to observed finishing orders.

Cycling's Dixon-Coles: the first thing in this package that estimates something
rather than describing it. One latent strength per rider, fitted by maximum
likelihood over the Plackett-Luce likelihood of every finishing order in the
training data.

**How it differs from the baseline it has to beat.** `baseline.form_worths`
scores a placing heuristically — win a 180-rider stage, take 180 points — and
sums. That is a reasonable proxy for a pre-race ranking and it is what a ranking
*is*. This fits the same kind of number properly: it asks which strengths make
the orderings actually observed most likely, which correctly discounts beating a
weak field and rewards beating a strong one. Whether that difference is worth
anything is not for this docstring to say; `cycling/evaluation.py` measures it.

**Abandons are information, not missing data.** The likelihood places riders one
at a time and keeps everyone unplaced in the denominator, so a rider who started
and did not finish contributes: they were available to win each position and
took none. Strengths are therefore estimated on the field as it started, which
is the only field anyone could have bet on.

**Fitted with MM, not a general optimiser.** The Plackett-Luce likelihood has a
classical minorise-maximise update (Hunter 2004) that is monotone by
construction — every iteration raises the likelihood — and costs one pass over
the data. A generic optimiser over 180 parameters is slower and can wander; this
cannot.

**The scale is arbitrary and the model says so.** Multiplying every worth by a
constant changes no probability, so the fit is normalised to a mean worth of 1
and nothing should ever be read off a single worth's magnitude.
"""

import numpy as np

from cycling.common import FINISHED
from cycling.processor import GROUP_KEYS

DEFAULT_ITERATIONS = 200
CONVERGENCE_TOLERANCE = 1e-8

# Every rider starts here, and it is also what an unseen rider is given: the
# field average, which is the honest prior for someone with no results.
PRIOR_WORTH = 1.0

# Pseudo-observations pulling each rider toward the field average, in the units
# the likelihood counts in: a rider with this many placings' worth of evidence
# is halfway between the prior and what their results alone would say.
#
# It is not a tuning knob added for accuracy. Without it the fit is not merely
# noisy on thin data, it is **undefined**: a rider nobody ever finished behind
# has a maximum-likelihood worth that rises without bound, and one who never
# finished ahead of anyone has one that goes to zero — which then takes a
# logarithm to negative infinity in every downstream score. On the synthetic
# Grand Tour, going from pure MLE to this prior moves the correlation between
# fitted strength and the generator's own truth from undefined to 0.76.
DEFAULT_PRIOR_STRENGTH = 5.0

# A floor under the fitted worths, so nothing downstream ever takes log(0).
# With any positive prior the update cannot reach zero anyway; this is what
# keeps `prior_strength=0` — a legitimate ask for the pure MLE — from producing
# a model that poisons every score computed from it.
_WORTH_FLOOR = 1e-9


class PlackettLuce:
    def __init__(self, worths):
        self.worths = dict(worths)

    @property
    def riders(self):
        return tuple(sorted(self.worths))

    def worth(self, rider):
        """A rider's fitted strength, or the field average for one never seen.

        An unknown rider is given the prior rather than refused, unlike
        football's `UnknownTeamError`. The two cases are genuinely different: a
        football fixture between two teams, one unheard of, cannot be predicted
        at all, while a 180-rider start list with three neo-pros in it is an
        ordinary Tuesday and dropping the race over them would throw away the
        177 riders the model does know.
        """
        return float(self.worths.get(rider, PRIOR_WORTH))

    def worths_for(self, riders):
        """Worths aligned to a start list, ready for `baseline`/`scoring`."""
        return np.array([self.worth(rider) for rider in riders], dtype=float)

    @classmethod
    def fit(cls, results, iterations=DEFAULT_ITERATIONS, half_life=None,
            prior_strength=DEFAULT_PRIOR_STRENGTH):
        """Strengths from every finishing order in `results`, shrunk toward the field.

        `prior_strength` is in placings: a rider with that much evidence sits
        halfway between the field average and what their own results say. Pass
        0 for the pure maximum-likelihood fit, which is unbounded for a rider
        nobody finished behind and should not be used for scoring.
        """
        orders, fields, weights, riders = _races_from_results(results, half_life=half_life)
        if not orders:
            return cls({rider: PRIOR_WORTH for rider in riders})
        worths = _fit_mm(orders, fields, weights, len(riders), iterations, prior_strength)
        return cls(dict(zip(riders, worths, strict=True)))


def _races_from_results(results, half_life=None):
    """Per race: the finishing order, the full start list, and a time weight.

    Both lists come out as indices into the shared rider vocabulary, and both
    come from the same pass so an order can never be paired with another race's
    field. The start list is the whole group — abandons included — which is what
    keeps them in the likelihood's denominator.
    """
    ordered = results.sort_values("ds")
    riders = sorted(set(ordered["rider"].dropna()))
    index = {rider: i for i, rider in enumerate(riders)}
    latest = ordered["ds"].max() if len(ordered) else None

    orders, fields, weights = [], [], []
    for _, group in ordered.groupby(GROUP_KEYS, dropna=False):
        ranked = group[(group["status"] == FINISHED) & group["rank"].notna()]
        if len(ranked) < 2:
            continue  # a single placing constrains nothing
        orders.append([index[rider] for rider in ranked.sort_values("rank")["rider"]])
        fields.append([index[rider] for rider in group["rider"].dropna().unique()])
        if half_life:
            age_days = max((latest - group["ds"].min()).days, 0)
            weights.append(0.5 ** (age_days / float(half_life)))
        else:
            weights.append(1.0)
    return orders, fields, weights, riders


def _fit_mm(orders, fields, weights, n_riders, iterations, prior_strength):
    """Hunter's minorise-maximise iteration for Plackett-Luce worths.

    Each sweep accumulates, per rider, how many placings they took (the
    numerator) and how much "exposure" they had — the reciprocal of the worth
    still unplaced at each step they were available for (the denominator). The
    ratio is the next iterate, and the likelihood cannot go down.

    The prior enters as the same quantity on both sides: `prior_strength`
    pseudo-placings and the matching pseudo-exposure. A rider with no results
    therefore lands exactly on 1, the field average, and one with plenty is
    barely moved — which is what shrinkage should do.
    """
    worths = np.ones(n_riders, dtype=float)
    prior = float(prior_strength)

    for _ in range(int(iterations)):
        wins = np.full(n_riders, prior)
        exposure = np.full(n_riders, prior)

        for order, field, weight in zip(orders, fields, weights, strict=True):
            available = np.zeros(n_riders, dtype=bool)
            available[field] = True
            remaining = float(worths[available].sum())

            for rider in order:
                if remaining <= 0:
                    break
                wins[rider] += weight
                # Everyone still unplaced shared the risk of this position.
                exposure[available] += weight / remaining
                available[rider] = False
                remaining -= float(worths[rider])

        with np.errstate(divide="ignore", invalid="ignore"):
            updated = np.where(exposure > 0, wins / exposure, worths)
        updated = np.where(np.isfinite(updated), updated, worths)
        updated = np.maximum(updated, _WORTH_FLOOR)
        # The scale is arbitrary — every probability is a ratio — so pinning the
        # mean keeps successive iterates comparable and the numbers readable.
        updated = updated / updated.mean()

        shift = float(np.abs(updated - worths).max())
        worths = updated
        if shift < CONVERGENCE_TOLERANCE:
            break

    return worths / worths.mean()
