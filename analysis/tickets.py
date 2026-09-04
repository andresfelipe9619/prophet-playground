"""Generate tickets, check them against real draws, and measure whether a
generation strategy does better than picking at random.

This is the experimental surface of the project. Generating a ticket is
legitimate and so is checking it; what no strategy here can do is make a ticket
*more likely* to win than any other, because all C(43,5) x 16 = 15,401,568
tickets are equally probable. `evaluate_strategy` is the instrument that
measures that claim on your own data rather than asking you to take it on faith.

One thing genuinely *is* optimizable: the structure of a **set** of tickets.
Playing five disjoint tickets covers 25 distinct numbers instead of possibly
repeating the same ones, which changes the distribution of outcomes across the
portfolio (not the expected value of any single ticket). See `generate_portfolio`
and `portfolio_coverage`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.prizes import total_combinations
from models.baseline import beats_chance_test
from models.common import (
    MAIN_BALLS_DRAWN,
    MAIN_BALL_RANGE,
    SUPER_BALL_RANGE,
    main_positions,
    super_position,
)

MAIN_NUMBERS = np.arange(MAIN_BALL_RANGE[0], MAIN_BALL_RANGE[1] + 1)
SUPER_NUMBERS = np.arange(SUPER_BALL_RANGE[0], SUPER_BALL_RANGE[1] + 1)


@dataclass(frozen=True)
class Ticket:
    """5 distinct main numbers plus one superbalota."""

    main: tuple
    super_ball: int

    def __post_init__(self):
        if len(set(self.main)) != MAIN_BALLS_DRAWN:
            raise ValueError(f"A ticket needs {MAIN_BALLS_DRAWN} distinct main numbers, got {self.main}")
        for n in self.main:
            if not MAIN_BALL_RANGE[0] <= n <= MAIN_BALL_RANGE[1]:
                raise ValueError(f"Main number {n} outside {MAIN_BALL_RANGE}")
        if not SUPER_BALL_RANGE[0] <= self.super_ball <= SUPER_BALL_RANGE[1]:
            raise ValueError(f"Superbalota {self.super_ball} outside {SUPER_BALL_RANGE}")

    def __str__(self):
        return f"{' - '.join(str(n) for n in sorted(self.main))}  +  {self.super_ball}"

    @property
    def main_set(self):
        return frozenset(self.main)


# --------------------------------------------------------------- generation

def random_ticket(rng=None):
    """Uniform pick. Every ticket is equally likely, so this is the honest default."""
    rng = rng or np.random.default_rng()
    return Ticket(
        main=tuple(int(n) for n in rng.choice(MAIN_NUMBERS, size=MAIN_BALLS_DRAWN, replace=False)),
        super_ball=int(rng.choice(SUPER_NUMBERS)),
    )


def _weighted_pick(weights, rng):
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    probabilities = weights / weights.sum()
    main = rng.choice(MAIN_NUMBERS, size=MAIN_BALLS_DRAWN, replace=False, p=probabilities)
    return tuple(int(n) for n in main)


def _pooled_main_counts(balls_expanded):
    n_columns = balls_expanded.shape[1]
    pooled = balls_expanded.iloc[:, list(main_positions(n_columns))].to_numpy().ravel()
    return pd.Series(pooled).value_counts().reindex(MAIN_NUMBERS, fill_value=0).to_numpy()


def hot_ticket(balls_expanded, rng=None, strength=1.0):
    """Sample main numbers weighted toward those drawn most often.

    A popular heuristic with no predictive edge — included so `evaluate_strategy`
    can measure it against chance instead of arguing about it.
    """
    rng = rng or np.random.default_rng()
    counts = _pooled_main_counts(balls_expanded).astype(float)
    weights = np.power(counts + 1.0, strength)
    super_ball = int(rng.choice(SUPER_NUMBERS))
    return Ticket(main=_weighted_pick(weights, rng), super_ball=super_ball)


def cold_ticket(balls_expanded, rng=None, strength=1.0):
    """The mirror image: weighted toward the least-drawn numbers. Equally unfounded."""
    rng = rng or np.random.default_rng()
    counts = _pooled_main_counts(balls_expanded).astype(float)
    weights = np.power(counts.max() - counts + 1.0, strength)
    super_ball = int(rng.choice(SUPER_NUMBERS))
    return Ticket(main=_weighted_pick(weights, rng), super_ball=super_ball)


def ticket_from_predictions(predictions, rng=None):
    """Turn a model's {position: number} output into a valid ticket.

    Models frequently predict the same number in several positions (a symptom of
    having no signal to tell the slots apart), which would leave fewer than five
    distinct numbers. Missing slots are filled at random, and how many had to be
    filled is worth reporting — see `dashboard`'s collision note.
    """
    rng = rng or np.random.default_rng()
    n_columns = len(predictions)
    main = list(dict.fromkeys(int(predictions[p]) for p in main_positions(n_columns)))

    remaining = [int(n) for n in MAIN_NUMBERS if n not in main]
    while len(main) < MAIN_BALLS_DRAWN:
        main.append(int(rng.choice(remaining)))
        remaining.remove(main[-1])

    return Ticket(main=tuple(main[:MAIN_BALLS_DRAWN]), super_ball=int(predictions[super_position(n_columns)]))


STRATEGIES = {
    "random": lambda balls_expanded, rng: random_ticket(rng),
    "hot": lambda balls_expanded, rng: hot_ticket(balls_expanded, rng),
    "cold": lambda balls_expanded, rng: cold_ticket(balls_expanded, rng),
}


def generate_portfolio(n_tickets, strategy="random", balls_expanded=None, rng=None, disjoint=True):
    """Generate several tickets at once.

    With `disjoint=True` the main numbers never repeat across tickets until the
    43-number pool runs out (8 tickets). This does not improve any single
    ticket's odds — nothing can — but it spreads the portfolio over more of the
    pool, which changes how outcomes are distributed across the set. See
    `portfolio_coverage` for the exact figures.
    """
    rng = rng or np.random.default_rng()
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Available: {sorted(STRATEGIES)}")
    if strategy != "random" and balls_expanded is None:
        raise ValueError(f"Strategy {strategy!r} needs balls_expanded (historical draws)")

    tickets, used = [], set()
    for _ in range(n_tickets):
        if disjoint and len(used) + MAIN_BALLS_DRAWN <= len(MAIN_NUMBERS):
            available = np.array([n for n in MAIN_NUMBERS if n not in used])
            main = tuple(int(n) for n in rng.choice(available, size=MAIN_BALLS_DRAWN, replace=False))
            ticket = Ticket(main=main, super_ball=int(rng.choice(SUPER_NUMBERS)))
        else:
            ticket = STRATEGIES[strategy](balls_expanded, rng)
        used.update(ticket.main)
        tickets.append(ticket)
    return tickets


def portfolio_coverage(tickets):
    """Exact figures for a set of tickets. No estimation involved."""
    covered = set()
    for t in tickets:
        covered |= t.main_set
    n_tickets = len(tickets)
    distinct = len(covered)
    return {
        "n_tickets": n_tickets,
        "distinct_main_numbers": distinct,
        "pool_coverage_pct": 100 * distinct / len(MAIN_NUMBERS),
        # Linearity of expectation: each covered number is drawn with probability 5/43
        "expected_total_main_matches": distinct * MAIN_BALLS_DRAWN / len(MAIN_NUMBERS),
        "jackpot_probability": n_tickets / total_combinations(),
        "jackpot_odds_one_in": total_combinations() / n_tickets if n_tickets else float("inf"),
    }


# ----------------------------------------------------------------- checking

def draw_from_row(balls_expanded_row):
    """Turn one row of balls_expanded into (main set, superbalota)."""
    values = [int(v) for v in balls_expanded_row]
    n_columns = len(values)
    return frozenset(values[p] for p in main_positions(n_columns)), values[super_position(n_columns)]


def check_ticket(ticket, main_drawn, super_drawn):
    """Score one ticket against one draw."""
    main_matches = len(ticket.main_set & frozenset(main_drawn))
    super_match = ticket.super_ball == super_drawn
    return {
        "main_matches": main_matches,
        "super_match": super_match,
        "category": f"{main_matches} + superbalota" if super_match else f"{main_matches} aciertos",
        "matched_numbers": sorted(ticket.main_set & frozenset(main_drawn)),
    }


def check_against_history(ticket, df, balls_expanded):
    """How this exact ticket would have done in every historical draw."""
    rows = []
    for (_, draw_row), date in zip(balls_expanded.iterrows(), df["ds"]):
        main_drawn, super_drawn = draw_from_row(draw_row)
        result = check_ticket(ticket, main_drawn, super_drawn)
        rows.append({"ds": date, **{k: v for k, v in result.items() if k != "matched_numbers"}})
    return pd.DataFrame(rows)


def history_summary(results):
    """Aggregate a check_against_history frame into per-category counts."""
    counts = results["category"].value_counts()
    return pd.DataFrame({
        "category": counts.index,
        "times": counts.values,
        "share_pct": 100 * counts.values / len(results),
    }).reset_index(drop=True)


# ------------------------------------------------- the accuracy measurement

def evaluate_strategy(strategy, df, balls_expanded, n_draws_back=100, tickets_per_draw=1,
                      seed=0, min_history=50):
    """Walk-forward test of a generation strategy against the chance baseline.

    For each of the last `n_draws_back` draws, tickets are generated using only
    the draws that came before it, then scored against what actually came out.
    The resulting hit counts go through the same exact hypergeometric test the
    model backtest uses.

    This is the experiment: if a strategy has an edge, `p_value_greater` is
    small. For a fair lottery it will not be.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Available: {sorted(STRATEGIES)}")

    rng = np.random.default_rng(seed)
    total = len(balls_expanded)
    start = max(min_history, total - n_draws_back)
    if start >= total:
        raise ValueError(
            f"No draws to evaluate: {total} draws with min_history={min_history}. "
            f"Lower min_history below {total}, or use a longer history."
        )

    hits, super_hits = [], []
    for t in range(start, total):
        history = balls_expanded.iloc[:t]
        main_drawn, super_drawn = draw_from_row(balls_expanded.iloc[t])
        for _ in range(tickets_per_draw):
            ticket = STRATEGIES[strategy](history, rng)
            result = check_ticket(ticket, main_drawn, super_drawn)
            hits.append(result["main_matches"])
            super_hits.append(result["super_match"])

    chance = beats_chance_test(hits, MAIN_BALLS_DRAWN)
    return {
        "strategy": strategy,
        "n_tickets_evaluated": len(hits),
        "avg_main_matches": float(np.mean(hits)),
        "chance_avg_main_matches": chance["chance_mean"],
        "z": chance["z"],
        "p_value_better_than_chance": chance["p_value_greater"],
        "beats_chance": bool(chance["p_value_greater"] < 0.05)
                        if pd.notna(chance["p_value_greater"]) else False,
        "super_hit_rate": float(np.mean(super_hits)),
        "best_result": int(np.max(hits)) if hits else 0,
    }


def stability_check(df, balls_expanded, strategy="random", n_seeds=20, alpha=0.05, **kwargs):
    """Repeat the experiment across seeds and count how often it flags a winner.

    A single run of `evaluate_strategy` is one draw from a noisy process: at
    alpha=0.05, a strategy with no edge whatsoever still looks like a winner
    about 1 run in 20. That is not a bug in the test — it is what alpha means —
    and it is the most common way people convince themselves a lottery system
    works.

    Run this before believing any positive result. For `random`, which cannot
    have an edge by construction, `flag_rate` should land near alpha. A strategy
    whose flag rate is not meaningfully above `random`'s has shown nothing.
    """
    flags, p_values = 0, []
    for seed in range(n_seeds):
        result = evaluate_strategy(strategy, df, balls_expanded, seed=seed, **kwargs)
        p = result["p_value_better_than_chance"]
        p_values.append(p)
        flags += bool(pd.notna(p) and p < alpha)

    return {
        "strategy": strategy,
        "n_seeds": n_seeds,
        "times_flagged": flags,
        "flag_rate": flags / n_seeds,
        "expected_flag_rate_if_no_edge": alpha,
        "median_p_value": float(np.median(p_values)),
    }


def compare_strategies(df, balls_expanded, strategies=None, alpha=0.05, **kwargs):
    """Run evaluate_strategy over several strategies and return one table.

    Testing k strategies at once means k chances to get a false positive: at
    alpha=0.05 with three strategies, there is a ~14% chance that at least one
    signal-free strategy looks like a winner. The `beats_chance` column is the
    naive per-test verdict; `beats_chance_corrected` applies a Bonferroni
    threshold (alpha / k) and is the one to read when comparing a table.

    Note on `tickets_per_draw > 1`: for history-dependent strategies (hot, cold)
    the tickets generated for one draw are correlated with each other, so those
    rows carry slightly less information than their ticket count suggests. The
    `random` strategy is unaffected — its match distribution does not depend on
    which numbers were drawn.
    """
    strategies = strategies or list(STRATEGIES)
    rows = [evaluate_strategy(s, df, balls_expanded, **kwargs) for s in strategies]

    corrected = alpha / len(strategies)
    for row in rows:
        p = row["p_value_better_than_chance"]
        row["bonferroni_threshold"] = corrected
        row["beats_chance_corrected"] = bool(p < corrected) if pd.notna(p) else False

    return pd.DataFrame(rows).sort_values("avg_main_matches", ascending=False).reset_index(drop=True)
