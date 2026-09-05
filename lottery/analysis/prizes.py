"""Exact prize-category probabilities and expected value per ticket.

This is the part of the project where the math actually answers a decision.
Which numbers will come up is unknowable, but what a ticket is *worth* is
not: the probability of every prize category is exact combinatorics, so
given the current prize table you can compute the expected return of a
$5.700 ticket down to the peso.

A Baloto ticket is 5 distinct numbers from 1-43 plus a superbalota from
1-16, so there are C(43,5) x 16 = 15,401,568 equally likely tickets. Main
matches follow a hypergeometric distribution and the superbalota is
independent of them, which makes each category a simple product.

Payouts are left to the caller because most tiers are set by the operator
and the top prize accumulates — pass in the current prize table rather
than trusting a number hardcoded here.
"""

from math import comb

import pandas as pd

from lottery.models.common import MAIN_BALLS_DRAWN, MAIN_POOL, SUPER_POOL


def total_combinations(main_pool=MAIN_POOL, main_drawn=MAIN_BALLS_DRAWN, super_pool=SUPER_POOL):
    return comb(main_pool, main_drawn) * super_pool


def main_match_probability(k, main_pool=MAIN_POOL, main_drawn=MAIN_BALLS_DRAWN):
    """P(exactly k of your 5 numbers are drawn) — hypergeometric, computed exactly."""
    return (
        comb(main_drawn, k) * comb(main_pool - main_drawn, main_drawn - k)
        / comb(main_pool, main_drawn)
    )


def category_probabilities(main_pool=MAIN_POOL, main_drawn=MAIN_BALLS_DRAWN, super_pool=SUPER_POOL):
    """Every (main matches, superbalota hit) combination with its exact probability."""
    rows = []
    for k in range(main_drawn + 1):
        p_main = main_match_probability(k, main_pool, main_drawn)
        for super_match in (True, False):
            p_super = (1 / super_pool) if super_match else (1 - 1 / super_pool)
            probability = p_main * p_super
            rows.append({
                "main_matches": k,
                "super_match": super_match,
                "category": f"{k} + superbalota" if super_match else f"{k} aciertos",
                "probability": probability,
                "odds_one_in": (1 / probability) if probability > 0 else float("inf"),
            })
    table = pd.DataFrame(rows)
    return table.sort_values(["main_matches", "super_match"], ascending=[False, False]).reset_index(drop=True)


def apply_payouts(prob_table, payouts):
    """Attach a payout per category. `payouts` maps (main_matches, super_match) -> amount."""
    table = prob_table.copy()
    table["payout"] = [
        float(payouts.get((int(row.main_matches), bool(row.super_match)), 0.0))
        for row in table.itertuples()
    ]
    table["expected_contribution"] = table["probability"] * table["payout"]
    return table


def expected_value(prob_table, payouts, ticket_price):
    """Expected return of one ticket: what you get back on average vs. what you pay."""
    table = apply_payouts(prob_table, payouts)
    expected_return = float(table["expected_contribution"].sum())
    win_probability = float(table.loc[table["payout"] > 0, "probability"].sum())
    return {
        "expected_return": expected_return,
        "ticket_price": ticket_price,
        "expected_value": expected_return - ticket_price,
        "return_to_player": (expected_return / ticket_price) if ticket_price else float("nan"),
        "win_any_prize_probability": win_probability,
        "odds_any_prize_one_in": (1 / win_probability) if win_probability > 0 else float("inf"),
        "table": table,
    }


def breakeven_jackpot(prob_table, payouts, ticket_price, jackpot_key=(MAIN_BALLS_DRAWN, True)):
    """How big would the top prize have to be for a ticket to break even?

    Solves expected_return == ticket_price for the jackpot payout, holding
    every other tier fixed. Worth knowing, but a positive expected value on
    paper is not the whole story: with a big jackpot more people play, and
    a shared jackpot pays each winner less than this assumes (taxes and the
    lump-sum discount cut it further).
    """
    other_payouts = {k: v for k, v in payouts.items() if k != jackpot_key}
    table = apply_payouts(prob_table, other_payouts)
    expected_without_jackpot = float(table["expected_contribution"].sum())

    match = prob_table[
        (prob_table["main_matches"] == jackpot_key[0]) & (prob_table["super_match"] == jackpot_key[1])
    ]
    jackpot_probability = float(match["probability"].iloc[0])
    if jackpot_probability <= 0:
        return float("inf")
    return (ticket_price - expected_without_jackpot) / jackpot_probability
