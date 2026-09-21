"""Outright prices into the bar a cycling model actually has to clear.

This is the module `docs/cycling.md` has been apologising for. Every cycling
verdict in this project is against the **pre-race ranking**, which is the soft
bar — the one a model beats by being a slightly better reader of recent form.
The hard bar, the one football has had from the start, is the market. Where a
price exists it should be the baseline, and until now none could be.

`football/market.py` is the model for this, and most of it carries over: raw
prices imply probabilities, the overround has to come off before they mean
anything, and the normalisation used is a modelling choice rather than
arithmetic. Three things do not carry over, and they are why this is a separate
module rather than a call into that one.

**The field is ~180 runners, not three.** Football's overround is 1.02-1.08;
an outright cycling book runs far higher, because a bookmaker pricing 180
mutually exclusive outcomes takes a margin on each. Removing that much excess
is a much larger intervention, and the three normalisations that "disagree on
longshots" in football disagree *enormously* here — where almost every runner
is a longshot. `compare_methods` is not optional reading.

**The favourite-longshot bias is the whole shape of the book.** A 200/1 rider
is not priced at 200/1 because anyone believes 0.5%; they are priced there
because that is the shortest price the book can offer on a runner who will not
win. Multiplicative de-margining scales every price by the same factor and so
leaves that bias fully intact, which is why `power` exists and why it matters
more here than in football. A model measured against a multiplicatively
de-margined outright book is measured against a baseline that is badly wrong
about the tail — and the tail is 170 of the 180 riders.

**Probabilities are not worths.** Plackett-Luce worths are what everything
downstream of `cycling/baseline.py` consumes, and a win probability is not one:
the PL win probability of a rider is `w_i / sum(w)`, so recovering worths from
win probabilities is only exact up to the scale the model does not identify.
`worths_from_market` does that inversion explicitly and says what it assumes.

**No price source is wired.** This module takes prices; nothing in this
repository fetches them, and `docs/cycling.md` says so rather than implying a
pipeline that does not exist. What it does do is make the baseline *expressible*
so that the moment prices are available, the verdict can be against them.
"""

import numpy as np
import pandas as pd

METHODS = ("multiplicative", "additive", "power")
DEFAULT_METHOD = "power"

# A price shorter than this is not a price on a bike race. Cycling books quote
# 1.5 on a dominant favourite at the very shortest; anything under 1.01 is a
# fractional or American column that has not been converted.
MIN_DECIMAL_ODDS = 1.01


class MarketFormatError(ValueError):
    """The prices do not describe one book on one race."""


def overround(odds):
    """Bookmaker margin: how much the raw implied probabilities exceed 1.

    **Same convention as `football/market.py:overround`** — the excess, not the
    total — because two functions of the same name meaning different things in
    two packages of one project is a trap nobody catches by reading either one.
    On an outright market the number is large and is *supposed* to be: pricing
    180 mutually exclusive runners carries a margin on each. A book at 0.40 is
    ordinary here; football's 0.05 would be extraordinary.
    """
    odds = np.asarray(odds, dtype=float)
    return float(np.nansum(1.0 / odds) - 1.0)


def _multiplicative(raw):
    """Scale every price by the same factor.

    Leaves the favourite-longshot bias entirely intact, which on a 180-runner
    book means the tail stays as wrong as the bookmaker made it. Offered for
    comparison, not as a default.
    """
    return raw / raw.sum()


def _additive(raw):
    """Take the same absolute excess off every runner.

    Included to show why it does not work here, not as a usable option. With
    180 runners the per-runner excess is large relative to a longshot's own
    probability, so subtracting it drives most of the field **negative** —
    measured on a realistic book, 106 of 180 runners.

    Clipping at zero is unavoidable (a negative probability is not one) but it
    breaks the sum, so the result is renormalised afterwards. That second step
    is a patch over a method that does not suit this shape of market, and
    `compare_methods` reports `n_zeroed` so the damage is visible rather than
    hidden behind a vector that now adds to 1. A method that says 106 riders
    cannot finish first is not a baseline; it is a warning.
    """
    excess = (raw.sum() - 1.0) / raw.size
    clipped = np.clip(raw - excess, 0.0, None)
    total = clipped.sum()
    return clipped / total if total > 0 else clipped


def _power(raw, tolerance=1e-10, max_iterations=100):
    """Raise every probability to a common exponent until they sum to 1.

    The one of the three that touches longshots differently from favourites,
    which is the whole point on a market whose tail is most of the field. The
    exponent is solved by bisection rather than assumed.
    """
    low, high = 1e-6, 100.0
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        total = np.sum(raw ** mid)
        if abs(total - 1.0) < tolerance:
            break
        if total > 1.0:
            low = mid
        else:
            high = mid
    return raw ** mid


_NORMALISERS = {"multiplicative": _multiplicative, "additive": _additive, "power": _power}


def implied_probabilities(odds, method=DEFAULT_METHOD):
    """Decimal outright odds to win probabilities summing to 1.

    Unlike football's, this takes a whole field at once rather than a triple:
    the normalisation is over the runners in one race, and a partial field
    would be normalised against the wrong total.
    """
    if method not in _NORMALISERS:
        raise MarketFormatError(f"Unknown method {method!r}. Available: {sorted(_NORMALISERS)}")

    odds = np.asarray(odds, dtype=float)
    if odds.ndim != 1:
        raise MarketFormatError(f"Expected one price per runner, got shape {odds.shape}.")

    finite = np.isfinite(odds)
    if not finite.any():
        return np.full(odds.shape, np.nan)
    if np.any(odds[finite] <= MIN_DECIMAL_ODDS):
        raise MarketFormatError(
            f"Decimal odds must exceed {MIN_DECIMAL_ODDS}; the shortest here is "
            f"{np.nanmin(odds):.3f}. A column of fractional or American odds usually "
            "looks like this."
        )

    raw = np.full(odds.shape, np.nan)
    raw[finite] = 1.0 / odds[finite]
    if raw[finite].sum() <= 1.0:
        raise MarketFormatError(
            f"These prices imply a total of {raw[finite].sum():.3f}, at or under 1.0. An "
            "outright book always overrounds; a total below 1 means the field is partial, "
            "and normalising a partial field spreads the missing runners' probability over "
            "the ones that are present."
        )

    out = np.full(odds.shape, np.nan)
    out[finite] = _NORMALISERS[method](raw[finite])
    return out


def worths_from_market(odds, method=DEFAULT_METHOD):
    """De-margined prices as Plackett-Luce worths, normalised to mean 1.

    **What this assumes, stated plainly:** that the market's win probabilities
    *are* the PL win probabilities, so `w_i` is proportional to `p_i`. Under
    Plackett-Luce the probability of winning is exactly `w_i / sum(w)`, so that
    inversion is exact up to the scale the model does not identify — and the
    scale is fixed here the same way `plackett_luce.py` fixes it, at mean 1.

    What it does **not** recover is how the market would order the rest of the
    field. A book prices who wins; Plackett-Luce describes the whole finishing
    order, and the parts of it below first place are not in the prices at all.
    A market baseline built this way is therefore a strong claim about the
    front of the race and an extrapolation about the back.
    """
    probabilities = implied_probabilities(odds, method=method)
    finite = np.isfinite(probabilities)
    if not finite.any():
        return probabilities

    worths = np.full(probabilities.shape, np.nan)
    worths[finite] = probabilities[finite] / probabilities[finite].mean()
    return worths


def market_frame(riders, odds, method=DEFAULT_METHOD):
    """One row per runner: the price, its implied probability and its worth."""
    riders = list(riders)
    odds = np.asarray(odds, dtype=float)
    if len(riders) != odds.size:
        raise MarketFormatError(f"{len(riders)} riders against {odds.size} prices.")
    if len(set(riders)) != len(riders):
        raise MarketFormatError("The same rider is priced twice in one field.")

    return pd.DataFrame({
        "rider": riders,
        "odds": odds,
        "probability": implied_probabilities(odds, method=method),
        "worth": worths_from_market(odds, method=method),
    })


def compare_methods(odds, methods=METHODS):
    """The same field under every normalisation, side by side.

    **Not optional on an outright book.** In football the three agree to within
    a point or two except on longshots; here almost every runner is a longshot
    and the gap between them is the difference between a baseline that is
    roughly right about the tail and one that is not. If a verdict flips
    between methods it is a statement about the margin model, not about the
    model being tested.

    Reports the favourite and the tail separately, because that is where they
    differ and an average over 180 runners hides it.
    """
    odds = np.asarray(odds, dtype=float)
    rows = []
    for method in methods:
        try:
            probabilities = implied_probabilities(odds, method=method)
        except MarketFormatError:
            continue
        finite = probabilities[np.isfinite(probabilities)]
        if not finite.size:
            continue
        ordered = np.sort(finite)[::-1]
        tail = ordered[max(len(ordered) // 2, 1):]
        rows.append({
            "method": method,
            "favourite": float(ordered[0]),
            "top_10_share": float(ordered[:10].sum()),
            "tail_share": float(tail.sum()),
            "n_zeroed": int((finite <= 0).sum()),
        })
    out = pd.DataFrame(rows)
    out.attrs["overround"] = overround(odds)
    out.attrs["n_runners"] = int(np.isfinite(odds).sum())
    return out


# What to do with a rider the book did not quote. A bookmaker prices the 40
# runners anyone will bet on and leaves 140 unquoted, so this is the ordinary
# case rather than an edge case, and it has to be decided rather than defaulted
# into.
UNPRICED_POLICIES = ("longest", "refuse")


def field_worths(riders, priced_riders, odds, method=DEFAULT_METHOD, unpriced="longest"):
    """Worths for a whole start list from a book that quoted part of it.

    **The quoted field is not the field.** De-margining normalises over the
    runners with prices, which implicitly says every unquoted rider has
    probability zero — and a zero takes a logarithm to minus infinity the moment
    one of them wins a stage, which unquoted riders do. So the unquoted riders
    have to be given something, and what they are given is a modelling choice
    this function makes visible rather than buries.

    `unpriced="longest"` gives each unquoted rider the implied probability of
    the **longest-priced runner the book did quote**, then renormalises the
    whole field. That is the bookmaker's own statement of its floor: it quoted
    200/1 and stopped, so nobody left out is shorter than that. It is an
    extrapolation and it is deliberately unflattering to the market — handing
    140 riders a real probability each takes probability away from the
    favourites, which makes the baseline *weaker* than the book actually is.
    A model that beats it has not yet beaten the market.

    `unpriced="refuse"` raises instead, for a caller who would rather drop the
    race than score against an extrapolation. `attrs` is not available on an
    array, so the count comes back beside the worths: `(worths, n_unpriced)`.
    """
    if unpriced not in UNPRICED_POLICIES:
        raise MarketFormatError(f"Unknown policy {unpriced!r}. Available: {list(UNPRICED_POLICIES)}")

    riders = list(riders)
    quoted = dict(zip(list(priced_riders), np.asarray(odds, dtype=float), strict=True))
    probabilities = implied_probabilities([quoted[r] for r in priced_riders], method=method)
    quoted_probability = dict(zip(list(priced_riders), probabilities, strict=True))

    missing = [r for r in riders if r not in quoted_probability]
    if missing and unpriced == "refuse":
        raise MarketFormatError(
            f"{len(missing)} of {len(riders)} starters are unquoted, e.g. {missing[:3]}. With "
            "`unpriced='refuse'` the race is not scored rather than scored against an "
            "extrapolation over the riders the book left out."
        )

    finite = probabilities[np.isfinite(probabilities)]
    floor = float(finite.min()) if finite.size else 0.0
    filled = np.array([quoted_probability.get(r, floor) for r in riders], dtype=float)

    total = filled.sum()
    if total <= 0:
        raise MarketFormatError("No usable prices for this start list.")
    filled = filled / total
    return filled / filled.mean(), len(missing)
