"""The bar a cycling model has to clear — the pre-race ranking, as a distribution.

This is cycling's `football/market.py`: the thing that turns "what was known
before the race" into probabilities a model can be scored against. It is the
piece that was missing, and without it nothing else in the domain could mean
anything.

**A uniform draw over the start list is not a baseline.** With ~180 starters it
gives everyone 0.55%, which any model beats by knowing a single name, and a
model that beats it has demonstrated only that cycling has favourites. It is
implemented here anyway, as `uniform_worths`, precisely so the mistake has a name
and a docstring saying what it is — an unnamed mistake gets made quietly.

**The real baseline is the market where a price exists and otherwise the
pre-race ranking**: UCI or PCS points, or start-list quality. `worths_from_points`
takes the first, `form_worths` builds the second out of a rider's earlier
results when no points are to hand, and both feed the same Plackett-Luce
machinery.

**Why Plackett-Luce.** The target is an ordering of ~180 riders, not a three-way
outcome, so a probability vector over winners is not enough on its own — it
says nothing about who comes second. Luce's rule gives each rider a positive
*worth*, makes the winner's probability its share of the total, then removes the
winner and repeats for second place. One number per rider generates a
distribution over whole finishing orders, which is exactly the shape the target
has.
"""

import numpy as np
import pandas as pd

from cycling.common import FINISHED

# Points are ordinal and a rider on zero is not impossible, only unlikely. This
# floor keeps a zero-point rider's worth positive, so Plackett-Luce can still
# place them and a log score of an actual win by one is finite rather than
# infinite. Small enough to sit well below any scoring rider.
POINTS_FLOOR = 1.0

# What one unit of rating is worth as a multiplier on the odds of winning.
# Only meaningful together with the rating's own scale, which is why it is a
# parameter and not a constant anyone should trust.
DEFAULT_RATING_SCALE = 1.0


def _as_worths(values):
    """A positive float array, with the zero-sum case refused rather than divided by."""
    worths = np.asarray(values, dtype=float)
    if worths.ndim != 1:
        raise ValueError(f"Expected one worth per rider, got shape {worths.shape}.")
    if not np.all(np.isfinite(worths)) or np.any(worths <= 0):
        raise ValueError("Worths must all be finite and strictly positive.")
    return worths


def uniform_worths(n_riders):
    """Equal worth for everyone — **not a baseline**, and here so the mistake is named.

    Scoring a model against this measures whether cycling has favourites, which
    it does, rather than whether the model knows anything. Use it only as the
    floor of a sanity check: a model that cannot beat *this* is broken.
    """
    return np.ones(int(n_riders), dtype=float)


def worths_from_points(points, floor=POINTS_FLOOR):
    """UCI/PCS points to Plackett-Luce worths, straight through with a floor.

    Points are already roughly proportional to how often a rider wins, which is
    what a Luce worth is, so no transform is applied — inventing one would make
    the baseline a model.
    """
    values = np.asarray(points, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    return np.maximum(values, float(floor))


def worths_from_rating(ratings, scale=DEFAULT_RATING_SCALE):
    """A rating on an arbitrary scale to worths, via `exp(rating / scale)`.

    For any strength expressed in log-odds-like units rather than in points.
    Subtracting the mean first only rescales every worth by a constant, which
    Plackett-Luce normalises away, but it keeps the numbers out of overflow.
    """
    values = np.asarray(ratings, dtype=float)
    values = np.where(np.isfinite(values), values, np.nanmin(values) if len(values) else 0.0)
    return np.exp((values - values.mean()) / float(scale))


def form_worths(results, riders, as_of=None, half_life=None, floor=POINTS_FLOOR):
    """Worths built from each rider's earlier results, for when no points exist.

    The honest fallback: a rider who has been finishing near the front is more
    likely to win the next one. Each past ranked finish scores
    `n_ranked - rank + 1`, so winning a 180-rider stage is worth 180 and coming
    last is worth 1; a non-finish scores nothing but does not erase what came
    before.

    **`as_of` is the whole point.** Only results strictly before it are used, so
    a baseline built for stage 12 cannot have seen stage 12. Leaving it None
    uses everything, which is right for describing a race and wrong for scoring
    a forecast of one.
    """
    history = results if as_of is None else results[results["ds"] < pd.Timestamp(as_of)]
    ranked = history[(history["status"] == FINISHED) & history["rank"].notna()]

    scores = {rider: 0.0 for rider in riders}
    if len(ranked):
        per_group = ranked.groupby(["race", "kind", "stage"], dropna=False)
        latest = pd.Timestamp(as_of) if as_of is not None else ranked["ds"].max()
        for _, group in per_group:
            size = len(group)
            if half_life:
                age_days = (latest - group["ds"].iloc[0]).days
                weight = 0.5 ** (max(age_days, 0) / float(half_life))
            else:
                weight = 1.0
            for row in group.itertuples():
                if row.rider in scores:
                    scores[row.rider] += weight * (size - float(row.rank) + 1.0)

    return np.maximum(np.array([scores[rider] for rider in riders], dtype=float), float(floor))


def win_probabilities(worths):
    """Each rider's share of the total worth — P(this rider wins), under Luce's rule."""
    worths = _as_worths(worths)
    return worths / worths.sum()


def predicted_order(worths, riders):
    """Riders from most to least likely, which is the baseline's point forecast."""
    worths = _as_worths(worths)
    order = np.argsort(-worths, kind="stable")
    return [riders[i] for i in order]


def sample_orders(worths, n_samples=2000, seed=0):
    """Draw whole finishing orders from the Plackett-Luce distribution.

    Uses the Gumbel-max trick: adding a standard Gumbel to each `log(worth)` and
    sorting descending produces an exact Plackett-Luce sample, in one vectorised
    pass instead of the sequential "pick, remove, renormalise" loop. Returns an
    `(n_samples, n_riders)` array of rider indices, best placed first.
    """
    worths = _as_worths(worths)
    rng = np.random.default_rng(seed)
    gumbel = rng.gumbel(size=(int(n_samples), len(worths)))
    return np.argsort(-(np.log(worths) + gumbel), axis=1, kind="stable")


def top_n_probabilities(worths, n=10, n_samples=2000, seed=0):
    """P(each rider finishes in the top `n`), by sampling.

    There is no cheap exact form for this beyond n = 1: the number of orderings
    to sum over grows factorially, which is why it is estimated. With 2,000
    samples the standard error on a probability near 0.5 is about 1.1 points —
    fine for reading, too coarse to split hairs with, so raise `n_samples`
    before drawing a conclusion from a small difference.
    """
    orders = sample_orders(worths, n_samples=n_samples, seed=seed)
    counts = np.bincount(orders[:, :int(n)].ravel(), minlength=len(worths))
    return counts / float(len(orders))


def baseline_frame(riders, worths, n=10, n_samples=2000, seed=0):
    """One row per rider: worth, win probability, top-N probability, predicted rank."""
    worths = _as_worths(worths)
    table = pd.DataFrame({
        "rider": list(riders),
        "worth": worths,
        "p_win": win_probabilities(worths),
        f"p_top_{int(n)}": top_n_probabilities(worths, n=n, n_samples=n_samples, seed=seed),
    })
    table = table.sort_values("worth", ascending=False).reset_index(drop=True)
    table.insert(0, "predicted_rank", np.arange(1, len(table) + 1))
    return table
