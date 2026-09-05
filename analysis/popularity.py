"""Jackpot splitting: the one lever in this game that is genuinely worth pulling.

Every other module here ends the same way — nothing changes your probability of
winning, because all 15,401,568 combinations are equally likely. That remains
true and this module does not dent it. **Picking unpopular numbers does not make
you win more often.**

What it changes is the *other* half of expected value. A jackpot is split among
everyone holding the winning combination, and people do not choose uniformly.
They pick birthdays, so numbers 1-31 are heavily over-played and 32-43 are
neglected. They pick patterns on the ticket grid, arithmetic runs, and "lucky"
numbers. A popular combination and an unpopular one win at exactly the same
rate, but conditional on winning, the popular one pays a fraction of the
unpopular one — the shared-jackpot effect is well documented across national
lotteries, and it is the only edge in a lottery that survives contact with the
mathematics.

So the honest framing, and the one every function here sticks to:

    P(win)          unchanged, always, by anything.
    E[payout | win] genuinely improvable, by avoiding what other people play.

**What this model is, precisely.** A heuristic scoring function over the biases
that show up in published research on lottery number choice, with weights this
project cannot calibrate, because doing that needs data on what tickets people
*bought* and no operator publishes it. So `popularity_score` is ordinal, not
absolute: it ranks one combination against another and its numbers do not mean
anything on their own. `expected_winners` turns that into a count only under an
explicitly supplied assumption about ticket volume, and it says so.

Treat the direction as solid (dates are over-played; the effect is large) and
any specific number as a rough estimate. Where a decision only needs the
direction — "should I avoid 1-31?" — that is enough. Where it needs the
magnitude, it is not, and `split_adjusted_value` reports the range instead of a
single figure.
"""

import math
from collections import Counter

import numpy as np
import pandas as pd

from analysis.prizes import category_probabilities, total_combinations
from analysis.tickets import Ticket
from models.common import MAIN_BALLS_DRAWN, MAIN_BALL_RANGE, MAIN_POOL, SUPER_POOL

# Numbers that fit a day of the month. The single largest documented bias in
# lottery number choice: players use birthdays and anniversaries, so 1-31 are
# picked far more often than 32-43 despite being no more likely.
CALENDAR_MAX = 31

# Relative weights of each bias in the popularity score. Ordinal, not
# calibrated — see the module docstring. Their ratios encode the ordering the
# literature agrees on (the date effect dominates everything else), and the
# absolute scale is arbitrary.
BIAS_WEIGHTS = {
    "calendar": 1.00,      # every main number <= 31
    "low_numbers": 0.35,   # clustered in the bottom of the pool
    "consecutive": 0.30,   # runs like 7-8-9, chosen deliberately far too often
    "arithmetic": 0.45,    # even spacing: 5-10-15-20-25
    "lucky_numbers": 0.25,  # 7 and its multiples, 13 avoided in some markets
    "round_decade": 0.20,  # all from one row of the ticket grid
}

LUCKY_NUMBERS = (7, 11, 13, 21)

# How many random combinations `unpopular_ticket` scores before keeping the least
# popular. Kept modest because `evaluate_strategy` generates thousands of tickets
# and every one of them pays this cost; the score is coarse enough that the
# hundredth candidate rarely improves on the sixtieth.
DEFAULT_CANDIDATES = 60


def _calendar_fraction(main):
    return sum(1 for n in main if n <= CALENDAR_MAX) / len(main)


def _low_fraction(main):
    """How far the pick sits toward the bottom of the pool, 0 (top) to 1 (bottom)."""
    midpoint = (MAIN_BALL_RANGE[0] + MAIN_BALL_RANGE[1]) / 2
    return sum(1 for n in main if n < midpoint) / len(main)


def _consecutive_fraction(main):
    ordered = sorted(main)
    runs = sum(1 for a, b in zip(ordered, ordered[1:]) if b - a == 1)
    return runs / (len(ordered) - 1)


def _arithmetic_regularity(main):
    """1.0 when the gaps are perfectly even, 0.0 when they are as ragged as possible.

    An arithmetic progression is a strikingly common human "random" choice, and
    it is visually obvious on a ticket, which is what makes it popular.
    """
    ordered = sorted(main)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    mean = sum(gaps) / len(gaps)
    if mean == 0:
        return 1.0
    variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return max(0.0, 1.0 - math.sqrt(variance) / mean)


def _lucky_fraction(main):
    return sum(1 for n in main if n in LUCKY_NUMBERS) / len(main)


def _decade_concentration(main):
    """Share of the pick sitting in a single block of ten — one row of the grid.

    Plain Python rather than pandas: `unpopular_ticket` scores hundreds of
    candidates per generated ticket and `evaluate_strategy` generates thousands,
    so a `value_counts()` here dominates the runtime of the whole strategy.
    """
    counts = Counter((n - 1) // 10 for n in main)
    return max(counts.values()) / len(main)


BIAS_FUNCTIONS = {
    "calendar": _calendar_fraction,
    "low_numbers": _low_fraction,
    "consecutive": _consecutive_fraction,
    "arithmetic": _arithmetic_regularity,
    "lucky_numbers": _lucky_fraction,
    "round_decade": _decade_concentration,
}


def popularity_components(ticket):
    """Each bias scored 0-1 for one ticket, before weighting. Useful for explaining a score."""
    main = list(ticket.main) if isinstance(ticket, Ticket) else list(ticket)
    return {name: float(fn(main)) for name, fn in BIAS_FUNCTIONS.items()}


def popularity_score(ticket, weights=None):
    """Relative popularity of a combination, 0 (very unusual) to 1 (very commonly played).

    **Ordinal only.** A score of 0.6 does not mean 60% of anything; it means
    this combination is more commonly played than one scoring 0.4. Comparing
    two tickets is what it is for.
    """
    weights = weights or BIAS_WEIGHTS
    components = popularity_components(ticket)
    total = sum(weights.values())
    return sum(components[name] * weight for name, weight in weights.items()) / total


def expected_winners(ticket, tickets_sold, popularity_multiplier=None, weights=None):
    """Expected number of *other* jackpot winners sharing with this combination.

    Under uniform play each combination would be held by
    `tickets_sold / 15,401,568` people. Real play is not uniform, so that
    baseline is scaled by how popular the combination is. `popularity_multiplier`
    is the ratio between the most and least played combinations; the default of
    10 is a deliberately conservative reading of the published date-bias
    literature, where the spread for extreme combinations runs higher.

    `tickets_sold` is the caller's to supply and nothing here estimates it —
    operators publish it inconsistently, and inventing it would turn a documented
    ordering into a fake number.
    """
    multiplier = 10.0 if popularity_multiplier is None else popularity_multiplier
    score = popularity_score(ticket, weights=weights)
    # Map a 0-1 score onto [1/sqrt(m), sqrt(m)] so an average ticket sits at 1x
    # and the extremes are `multiplier` apart end to end.
    factor = multiplier ** (score - 0.5)
    return float(tickets_sold / total_combinations() * factor)


def split_adjusted_value(ticket, jackpot, tickets_sold, popularity_multiplier=None,
                         multiplier_range=(4.0, 25.0), weights=None):
    """What the jackpot is worth to this combination once sharing is priced in.

    Returns the point estimate plus a `low`/`high` band from varying the
    popularity multiplier across `multiplier_range`, because the multiplier is
    the least defensible number in this module and a single figure would hide
    that. If the band is wide enough to change your decision, the honest answer
    is that this model cannot make it for you.

    The share uses `1 / (1 + other winners)`: you always hold one ticket, and
    the expected split is over yourself plus everyone else holding the same
    combination.
    """
    def value(multiplier):
        others = expected_winners(ticket, tickets_sold, multiplier, weights)
        return jackpot / (1.0 + others)

    point = value(10.0 if popularity_multiplier is None else popularity_multiplier)
    low, high = value(multiplier_range[1]), value(multiplier_range[0])
    return {
        "popularity_score": popularity_score(ticket, weights=weights),
        "expected_other_winners": expected_winners(
            ticket, tickets_sold, popularity_multiplier, weights),
        "jackpot": jackpot,
        "expected_jackpot_share": point,
        "share_low": min(low, high),
        "share_high": max(low, high),
        "fraction_of_jackpot": point / jackpot if jackpot else np.nan,
    }


def unpopular_ticket(balls_expanded=None, rng=None, candidates=DEFAULT_CANDIDATES, weights=None):
    """Generate the least commonly played combination out of a random sample.

    Registered in `analysis.tickets.STRATEGIES` so `evaluate_strategy` can hold
    it to the same standard as every other strategy — and it will find, correctly,
    that it does **not** beat chance on hit rate. That is the expected result and
    not a mark against it: this strategy targets `E[payout | win]`, which the hit
    rate does not measure. It is the one strategy in the registry whose value the
    accuracy tests are structurally unable to see.

    `balls_expanded` is accepted and ignored, to match the strategy signature.
    Popularity is about what other players choose, not what the machine drew.
    """
    rng = rng or np.random.default_rng()
    pool = np.arange(MAIN_BALL_RANGE[0], MAIN_BALL_RANGE[1] + 1)

    best, best_score = None, np.inf
    for _ in range(candidates):
        ticket = Ticket(
            main=tuple(int(n) for n in rng.choice(pool, size=MAIN_BALLS_DRAWN, replace=False)),
            super_ball=int(rng.integers(1, SUPER_POOL + 1)),
        )
        score = popularity_score(ticket, weights=weights)
        if score < best_score:
            best, best_score = ticket, score
    return best


def compare_tickets(tickets, jackpot, tickets_sold, popularity_multiplier=None, weights=None):
    """One row per ticket: popularity, expected co-winners, and what the jackpot is worth."""
    rows = []
    for ticket in tickets:
        value = split_adjusted_value(ticket, jackpot, tickets_sold,
                                     popularity_multiplier=popularity_multiplier, weights=weights)
        rows.append({
            "ticket": str(ticket),
            "popularity_score": value["popularity_score"],
            "expected_other_winners": value["expected_other_winners"],
            "expected_jackpot_share": value["expected_jackpot_share"],
            "share_low": value["share_low"],
            "share_high": value["share_high"],
            **popularity_components(ticket),
        })
    return pd.DataFrame(rows).sort_values("expected_jackpot_share",
                                          ascending=False).reset_index(drop=True)


def jackpot_probability():
    """P(matching 5 main + superbalota) — restated here so callers need not import prizes."""
    table = category_probabilities()
    return float(table.loc[(table["main_matches"] == MAIN_BALLS_DRAWN)
                           & table["super_match"], "probability"].iloc[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jackpot", type=float, default=5_000_000_000)
    parser.add_argument("--tickets-sold", type=float, default=3_000_000,
                        help="tickets sold for the draw — no default is authoritative")
    parser.add_argument("--multiplier", type=float, default=10.0)
    args = parser.parse_args()

    examples = [
        Ticket(main=(3, 7, 12, 19, 25), super_ball=8),      # all dates
        Ticket(main=(1, 2, 3, 4, 5), super_ball=6),          # consecutive run
        Ticket(main=(5, 10, 15, 20, 25), super_ball=7),      # arithmetic
        Ticket(main=(2, 14, 23, 31, 38), super_ball=11),     # mixed
        Ticket(main=(33, 36, 38, 41, 43), super_ball=14),    # all above the calendar range
    ]
    pd.set_option("display.width", 160)
    table = compare_tickets(examples, args.jackpot, args.tickets_sold,
                            popularity_multiplier=args.multiplier)
    print(f"\n=== Jackpot {args.jackpot:,.0f} split across {args.tickets_sold:,.0f} tickets sold ===")
    print(table[["ticket", "popularity_score", "expected_other_winners",
                 "expected_jackpot_share", "share_low", "share_high"]].to_string(
        index=False, formatters={
            "popularity_score": "{:.3f}".format, "expected_other_winners": "{:.3f}".format,
            "expected_jackpot_share": "{:,.0f}".format,
            "share_low": "{:,.0f}".format, "share_high": "{:,.0f}".format,
        }))

    best, worst = table.iloc[0], table.iloc[-1]
    ratio = best["expected_jackpot_share"] / worst["expected_jackpot_share"]
    print(f"\nThe least popular of these is worth {ratio:.2f}x the most popular *if it wins*.")
    print(f"Both win with the same probability: {jackpot_probability():.3e} "
          f"(1 in {1 / jackpot_probability():,.0f}). Choosing numbers cannot change that, and this "
          "module does not claim otherwise — it only changes how many people you split with.")
    print("\nThe low/high columns come from varying the popularity multiplier over "
          "4x-25x, which is the least defensible input here. A decision that flips inside "
          "that band is one this model cannot make for you.")
