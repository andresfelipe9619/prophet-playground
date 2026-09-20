"""Did the price move toward you after you bet?

`evaluation.py` asks whether a model beats the closing line. It is the right
question and it is nearly unanswerable in a season: the per-match differences
are tiny against their own variance, so a few hundred matches cannot separate a
real 2% edge from nothing. `power.py`'s lesson, transplanted.

Closing line value asks something smaller and far easier to measure. You took a
price; the market then went on absorbing money until kick-off; where did it end
up? If prices move toward your bets more often than not, you were systematically
earlier than the market — and that converges in a few hundred bets rather than a
few thousand, because it is a direct measurement rather than a difference of two
noisy scores.

Both ends of the line are already in this project's data contract.
`extra_processor.py` carries opening prices, `processor.py` carries closing
ones; this module is the join between them and the test on what it finds.

Three things about it are load-bearing.

**CLV is measured in de-margined probability, not in raw price.** A bookmaker
who widens their margin between open and close moves every raw price against
every bettor, and a raw-price CLV would read that as everyone losing value. The
margin has to come off both ends first, exactly as it does before a model is
compared to the market at all — `market.py` owns that, and this module does not
reimplement it.

**The join is explicit and refuses rather than guessing.** CLV needs the same
match priced twice. Silently dropping the matches that only appear on one side
would quietly redefine the sample; silently pairing the wrong two matches would
be worse. `join_prices` raises on a frame with no overlap and reports what it
dropped.

**Positive CLV is not profit, and no amount of it is.** It says you were ahead
of the market's own revision, which is evidence that your information was real.
It says nothing about whether the edge survived the margin you paid to take the
bet — that is `value.py`'s question, and the two bars there are still the two
bars.

The endpoint pins the meaning, the way `ensemble.py`'s weight-0 blend does:
**CLV against the price you bet is exactly zero.** If that ever drifts, every
number here stops meaning anything.
"""

import numpy as np
import pandas as pd

from core.significance import bonferroni_threshold, verdicts, z_test_against_null
from football.common import ODDS_COLUMNS, OUTCOMES, outcome_index
from football.market import implied_probabilities
from football.processor import (
    CLOSING_SOURCES,
    ODDS_SOURCES,
    preprocess_matches,
)

DEFAULT_METHOD = "multiplicative"

# The columns a joined frame carries for each side, so the join is describable
# rather than implied by suffixes appearing in a debugger.
BET_ODDS_COLUMNS = tuple(f"bet_{c}" for c in ODDS_COLUMNS)
CLOSING_ODDS_COLUMNS = tuple(f"closing_{c}" for c in ODDS_COLUMNS)

MATCH_KEYS = ("ds", "home_team", "away_team")


class PriceJoinError(ValueError):
    """The two priced frames do not describe the same matches."""


def join_prices(bet_side, closing_side, keys=MATCH_KEYS):
    """Pair each match's taken price with its closing price.

    Both arguments are tidy match frames with odds columns — typically an
    opening-priced frame from `extra_processor.py` and a closing-priced one
    from `processor.py`. The join is on date and both team names, which is the
    only identifier this project's contract guarantees.

    Returns a frame carrying `bet_odds_*` and `closing_odds_*` alongside the
    match keys and the outcome, plus `attrs["n_dropped"]` — the matches that
    appeared on one side only. It **raises** when nothing joins at all, because
    an empty CLV table and a CLV of zero look identical downstream and mean
    opposite things.
    """
    keys = list(keys)
    for frame, label in ((bet_side, "bet"), (closing_side, "closing")):
        missing = [c for c in [*keys, *ODDS_COLUMNS] if c not in frame.columns]
        if missing:
            raise PriceJoinError(f"The {label} frame is missing {missing}.")

    left = bet_side[[*keys, "outcome", *ODDS_COLUMNS]].rename(
        columns=dict(zip(ODDS_COLUMNS, BET_ODDS_COLUMNS, strict=True)))
    right = closing_side[[*keys, *ODDS_COLUMNS]].rename(
        columns=dict(zip(ODDS_COLUMNS, CLOSING_ODDS_COLUMNS, strict=True)))

    joined = left.merge(right, on=keys, how="inner")
    if not len(joined):
        raise PriceJoinError(
            f"No match appears in both frames on {keys}. CLV needs the same fixture priced "
            "twice; an empty table and a CLV of zero look identical downstream and mean "
            "opposite things, so this refuses rather than returning one."
        )

    joined.attrs["n_dropped"] = int(max(len(bet_side), len(closing_side)) - len(joined))
    joined.attrs["bet_odds_source"] = bet_side.attrs.get("odds_source")
    joined.attrs["closing_odds_source"] = closing_side.attrs.get("odds_source")
    return joined


def paired_prices(raw, validate=False):
    """Both ends of the line out of one football-data season file.

    From 2019/20 those files carry the opening columns (`AvgH`, `PSH`, `B365H`)
    **and** the closing ones (`AvgCH`, `PSCH`, `B365CH`) side by side, which is
    what makes CLV measurable here at all without a second data source.

    The contract is not bypassed to get at them. `processor.py` resolves exactly
    one source per frame, on purpose, and that refusal is what stops opening and
    closing prices mixing inside a single model comparison. So this calls it
    **twice**: once on the file as published, which resolves the closing source,
    and once on a copy with the closing columns removed, which can then only
    resolve an opening one. Neither frame can contain both, and the guard that
    makes that true is still the one in `processor.py`.

    Raises if the file has no closing prices — a season before 2019/20, where
    the best available is an opening line and CLV would be measuring an opening
    price against itself.
    """
    closing = preprocess_matches(raw, validate=validate)
    if closing.attrs.get("odds_source") not in CLOSING_SOURCES:
        raise PriceJoinError(
            f"This file's best odds are {closing.attrs.get('odds_source')!r}, which is an "
            "opening line. football-data publishes closing odds only from 2019/20 onward, and "
            "closing line value measured against opening prices on both ends is not CLV."
        )

    closing_columns = [c for name, is_closing, triple in ODDS_SOURCES if is_closing
                       for c in triple if c in raw.columns]
    opening = preprocess_matches(raw.drop(columns=closing_columns), validate=validate)
    if opening.attrs.get("odds_source") in CLOSING_SOURCES:
        raise PriceJoinError(
            "Dropping the closing columns still resolved a closing source — ODDS_SOURCES and "
            "this function disagree about which columns are which."
        )

    return join_prices(opening, closing)


def clv(bet_odds, closing_odds, outcomes, method=DEFAULT_METHOD):
    """Probability the close put on your side, minus the probability you bought.

    Positive means the market moved toward the outcome you backed: your price
    was generous relative to where the line settled.

    De-margined on both sides before differencing. A raw-price version would
    credit or charge a bettor for the bookmaker changing their margin between
    open and close, which is a fact about the book and not about the bet.
    """
    bet = implied_probabilities(np.asarray(bet_odds, dtype=float), method=method)
    close = implied_probabilities(np.asarray(closing_odds, dtype=float), method=method)
    bet, close = np.atleast_2d(bet), np.atleast_2d(close)

    rows = np.arange(len(bet))
    picked = np.array([outcome_index(o) for o in outcomes])
    if picked.size != len(bet):
        raise ValueError(f"{len(bet)} prices against {picked.size} bets.")
    return close[rows, picked] - bet[rows, picked]


def clv_table(joined, bets=None, method=DEFAULT_METHOD):
    """Per-bet CLV for a joined frame.

    `bets` is the outcome backed on each match; leaving it None uses the match's
    actual `outcome`, which measures the CLV of a bettor with perfect hindsight
    and is useful only as a fixture. Any real use passes the model's picks.
    """
    if bets is None:
        bets = list(joined["outcome"])

    values = clv(joined[list(BET_ODDS_COLUMNS)].to_numpy(),
                 joined[list(CLOSING_ODDS_COLUMNS)].to_numpy(),
                 bets, method=method)

    out = joined[list(MATCH_KEYS)].copy()
    out["bet"] = list(bets)
    out["clv"] = values
    out["beat_the_close"] = values > 0
    out.attrs.update(joined.attrs)
    return out


def beats_closing_test(clv_values, alpha=0.05, n_comparisons=1, confidence=0.95):
    """Is the mean CLV distinguishable from zero, in the favourable direction?

    One-sided, like every "beats X" claim in this project: a bettor
    systematically *behind* the market's revision also gets a small two-sided
    p-value, and reading that as a pass turns the worst possible result into
    the best-looking one.

    The null is that the price you took is unbiased about where the line
    settles, so each bet's CLV has mean zero; its variance is estimated from
    the bets themselves, because unlike the lottery's hypergeometric there is
    no exact form for how far a line moves.
    """
    values = np.asarray(clv_values, dtype=float)
    values = values[np.isfinite(values)]
    threshold = bonferroni_threshold(alpha, n_comparisons)

    if values.size < 2:
        return {"n_bets": int(values.size), "mean_clv": float("nan"),
                "hit_rate": float("nan"), "bonferroni_threshold": threshold,
                "beats_closing": False, "beats_closing_corrected": False}

    result = z_test_against_null(values, null_means=0.0,
                                 null_variances=float(np.var(values, ddof=1)),
                                 confidence=confidence)
    verdict = verdicts(result["p_value_greater"], alpha, threshold)
    return {
        "n_bets": int(values.size),
        "mean_clv": float(values.mean()),
        # The share of bets the line moved toward, reported beside the mean
        # because they answer different questions: a hit rate near 1/2 with a
        # good mean is a handful of large wins, which is a different claim
        # about a bettor than being right more often.
        "hit_rate": float((values > 0).mean()),
        "effect": result["effect"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "p_value_greater": result["p_value_greater"],
        "bonferroni_threshold": threshold,
        "beats_closing": verdict["beats_chance"],
        "beats_closing_corrected": verdict["beats_chance_corrected"],
    }


def clv_report(joined, bets=None, method=DEFAULT_METHOD, alpha=0.05, n_comparisons=1):
    """`clv_table` and `beats_closing_test` together, which is how they are read."""
    table = clv_table(joined, bets=bets, method=method)
    result = beats_closing_test(table["clv"], alpha=alpha, n_comparisons=n_comparisons)
    result["bet_odds_source"] = joined.attrs.get("bet_odds_source")
    result["closing_odds_source"] = joined.attrs.get("closing_odds_source")
    result["n_dropped"] = joined.attrs.get("n_dropped")
    return table, result


def summarise_by_outcome(table):
    """CLV split by which outcome was backed — a diagnostic, not a verdict.

    Splitting three ways and reading the best slice is the same mistake as
    running six per-position chi-squares on Baloto and reporting the one that
    fired. It is here because a bettor whose CLV is carried entirely by away
    favourites has learned something about their own process, but the number
    to report is still the pooled one.
    """
    rows = []
    for outcome in OUTCOMES:
        slice_ = table[table["bet"] == outcome]
        rows.append({
            "bet": outcome,
            "n_bets": int(len(slice_)),
            "mean_clv": float(slice_["clv"].mean()) if len(slice_) else float("nan"),
            "hit_rate": float(slice_["beat_the_close"].mean()) if len(slice_) else float("nan"),
        })
    return pd.DataFrame(rows)
