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
            prior_strength=DEFAULT_PRIOR_STRENGTH, prior_worths=None):
        """Strengths from every finishing order in `results`, shrunk toward the field.

        `prior_strength` is in placings: a rider with that much evidence sits
        halfway between the field average and what their own results say. Pass
        0 for the pure maximum-likelihood fit, which is unbounded for a rider
        nobody finished behind and should not be used for scoring.

        `prior_worths` maps a rider to what to shrink them *toward*, replacing
        the field average. On a full calendar there is no reason to use it; on a
        subset — the seven mountain stages of one Grand Tour — it is what keeps
        the fit from being seven races of noise, because the rider is pulled
        toward what the whole calendar says about them rather than toward 1.
        """
        orders, fields, weights, riders = _races_from_results(results, half_life=half_life)
        target = (None if prior_worths is None
                  else np.array([float(prior_worths.get(r, PRIOR_WORTH)) for r in riders]))
        if not orders:
            return cls({rider: (PRIOR_WORTH if target is None else target[i])
                        for i, rider in enumerate(riders)})
        worths = _fit_mm(orders, fields, weights, len(riders), iterations, prior_strength,
                         prior_worths=target)
        return cls(_with_prior(dict(zip(riders, worths, strict=True)), prior_worths))


def _with_prior(fitted, prior_worths):
    """Keep a rider the fit never saw at whatever the prior said about them.

    Without this a terrain-conditional fit silently resets a rider who has not
    yet ridden a mountain stage to the field average — discarding exactly the
    information the prior was carrying, and doing it invisibly, since the
    resulting worth is a perfectly ordinary number. A rider with no climbing
    results is not an average climber; they are whoever the rest of the
    calendar said they were.

    The fitted worths are normalised to mean 1 over the riders in *this* frame
    and the prior over the riders in its own, so the two scales agree only
    approximately. That is the ordinary Plackett-Luce scale indeterminacy and
    it is the price of keeping the rider at all — the alternative is a number
    that is confidently wrong rather than approximately right.
    """
    if not prior_worths:
        return fitted
    return {**{rider: float(worth) for rider, worth in prior_worths.items()
               if rider not in fitted}, **fitted}


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


def _fit_mm(orders, fields, weights, n_riders, iterations, prior_strength, prior_worths=None):
    """Hunter's minorise-maximise iteration for Plackett-Luce worths.

    Each sweep accumulates, per rider, how many placings they took (the
    numerator) and how much "exposure" they had — the reciprocal of the worth
    still unplaced at each step they were available for (the denominator). The
    ratio is the next iterate, and the likelihood cannot go down.

    The prior enters as the same quantity on both sides: `prior_strength`
    pseudo-placings and the matching pseudo-exposure. A rider with no results
    therefore lands exactly on the prior mean, and one with plenty is barely
    moved — which is what shrinkage should do.

    `prior_worths` moves that mean off the field average. It is what makes a
    terrain-conditional fit possible at all: the mountain stages of one Grand
    Tour are seven races, and a rider is shrunk toward what the **whole**
    calendar says about them rather than toward 1. Left out, every rider is
    shrunk toward the field, which is the unconditional behaviour.
    """
    target = (np.ones(n_riders, dtype=float) if prior_worths is None
              else np.asarray(prior_worths, dtype=float))
    worths = target.copy()
    prior = float(prior_strength)

    for _ in range(int(iterations)):
        # Pseudo-placings in proportion to the prior mean, and the matching
        # exposure at 1: their ratio is the prior worth, so a rider with no
        # results lands exactly on it.
        wins = prior * target
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


# Below this many races of a given terrain, a conditional fit is not a fit — it
# is the prior with a handful of placings on top, and the unconditional model is
# the honest answer. Seven mountain stages of one Grand Tour is already thin.
MIN_TERRAIN_RACES = 4


class TerrainPlackettLuce:
    """One strength per rider **per kind of day**, shrunk toward their overall one.

    The model the sport obviously wants and the one whose value has to be
    measured rather than assumed. A sprinter and a climber are not two points on
    one scale, so fitting the mountain stages separately from the flat ones
    should help — and on a calendar of 21 races it may simply split thin data in
    two. `cycling/evaluation.py` is where that question is settled; this class
    only makes the comparison possible.

    **The terrain of the race being predicted is an argument, never an
    inference.** `features.infer_terrain` reads a result, so labelling the
    target race would be leakage of the invisible kind — the label looks
    identical either way. Training labels come from history; the target's comes
    from the caller, who has the roadbook.

    **A terrain without enough races falls back to the unconditional fit**
    rather than to a fit of four stages, and `available` says which terrains got
    their own. Falling back silently would make the model *look* conditional
    everywhere while being unconditional half the time, which is exactly the
    kind of thing a results table cannot show.
    """

    def __init__(self, overall, by_terrain, n_races=None):
        self.overall = overall
        self.by_terrain = dict(by_terrain)
        self.n_races = dict(n_races or {})

    @property
    def available(self):
        """The terrains that got their own fit, rather than the fallback."""
        return tuple(sorted(self.by_terrain))

    def model_for(self, terrain):
        return self.by_terrain.get(terrain, self.overall)

    def worth(self, rider, terrain=None):
        return self.model_for(terrain).worth(rider)

    def worths_for(self, riders, terrain=None):
        return self.model_for(terrain).worths_for(riders)

    @classmethod
    def fit(cls, results, iterations=DEFAULT_ITERATIONS, half_life=None,
            prior_strength=DEFAULT_PRIOR_STRENGTH, min_races=MIN_TERRAIN_RACES,
            terrains=None):
        """Fit the whole calendar, then each terrain's races on top of it.

        The conditional fits use the unconditional worths as their prior mean,
        so a rider with two mountain stages is shrunk toward their own overall
        strength rather than toward the field — which is the difference between
        a conditional model and a noisier copy of one.
        """
        from cycling.features import TERRAINS, UNKNOWN, infer_terrain

        overall = PlackettLuce.fit(results, iterations=iterations, half_life=half_life,
                                   prior_strength=prior_strength)
        labels = infer_terrain(results)
        wanted = [t for t in (terrains or TERRAINS) if t != UNKNOWN]

        by_terrain, counts = {}, {}
        for terrain in wanted:
            keys = labels[labels["terrain"] == terrain][GROUP_KEYS]
            counts[terrain] = len(keys)
            if len(keys) < min_races:
                continue
            subset = results.merge(keys, on=GROUP_KEYS, how="inner")
            by_terrain[terrain] = PlackettLuce.fit(
                subset, iterations=iterations, half_life=half_life,
                prior_strength=prior_strength, prior_worths=overall.worths)
        return cls(overall, by_terrain, counts)
