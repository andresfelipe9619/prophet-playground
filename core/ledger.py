"""What was actually staked, settled all at once, with the verdict above the total.

`core/registry.py` records a *forecast* before the event and scores it against
the domain's bar. This records the other half: a **stake** — a selection, a
price, an amount — and what came back. They are deliberately separate files and
separate modules, because they answer different questions and mixing them
produces a row that is both a claim and a bet and is read as whichever suits.

**The headline is the verdict, never the running total.** A profit-and-loss
figure at the top of a betting log is the single most misleading number this
project could put on a screen: it is the one quantity that looks like evidence,
moves every day, and means nothing at the sample sizes a person actually reaches.
So `summary` returns the corrected verdict, the interval and the **minimum
detectable return** in the same row as the money, and every surface that renders
it is required to lead with the first three. Forty settled bets cannot resolve a
5% edge, and a log that says "+18%" without saying that has misinformed its
reader.

**The null is break-even, which is generous on purpose.** A bettor with no edge
paying a bookmaker's margin has a *negative* expected return, so testing against
0 is a bar the no-edge case does not merely fail to clear — it is below it. That
choice is deliberate: the test should not be able to call someone profitable
because the null was set where they already were.

**Variance is measured, not assumed.** Per-bet returns have no closed-form
spread: it depends on the prices taken, which are a property of how someone
bets rather than of the game. So it comes from the settled rows themselves, the
same decision `football/power.py` makes about the paired score difference and
for the same reason.

**Three refusals, the registry's, restated for stakes.**

1. **A stake on an event that has already happened is rejected.** Backdating a
   bet is the one thing that makes a ledger worthless, and it is also the
   easiest thing in the world to do by accident with a date picker.
2. **Rows are append-only.** A settled row is never rewritten; a changed mind
   is a new row under a different label, which leaves both on the record.
3. **Every settleable row is settled, or none are.** `resolve` returning None
   means "not yet" and is the only reason a row may be skipped — picking which
   bets to count is the failure this file exists to prevent.

Domain-free: a selection is an opaque string here, and what counts as a price or
a payout is the caller's business.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from core.significance import bonferroni_threshold, verdicts, z_test_against_null

COLUMNS = ("recorded_at", "event_date", "label", "selection", "stake", "price",
           "note", "settled_at", "won", "payout")

# Filled at settlement and doubling as the "has this been settled" flag.
SETTLED_AT = "settled_at"
RESULT_COLUMNS = ("won", "payout")

# A price below this is not a price. 1.0 returns the stake and nothing else, so
# anything at or under it is either a mistake or not a bet.
MIN_PRICE = 1.0


class LedgerError(RuntimeError):
    """A write that would make the ledger stop being evidence."""


def _now() -> datetime:
    return datetime.now(UTC)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in COLUMNS})


def load(path: str) -> pd.DataFrame:
    """Read the ledger, or an empty frame with the right columns if it does not exist."""
    if not os.path.exists(path):
        return _empty()
    frame = pd.read_csv(path)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    for column in ("stake", "price", "payout"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[list(COLUMNS)]


def write(ledger: pd.DataFrame, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    ledger.sort_values(["event_date", "label", "selection"]).to_csv(path, index=False)


def record(path: str, event_date: Any, label: str, selection: str, stake: float,
           price: float, note: str = "", now: datetime | None = None) -> pd.DataFrame:
    """Add one stake on an event that has not happened yet.

    `now` is injectable so the refusal itself can be tested; leave it alone in
    normal use, where letting the caller choose "now" is precisely the hole this
    is here to close.
    """
    moment = now or _now()
    when = pd.to_datetime(event_date)
    if pd.isna(when):
        raise LedgerError(f"Could not read {event_date!r} as a date.")
    if when.to_pydatetime().replace(tzinfo=UTC) <= moment:
        raise LedgerError(
            f"{when:%Y-%m-%d} is not in the future. A ledger that accepts a stake on an event "
            "that has already happened proves nothing, and one such row makes the file worthless."
        )

    amount, odds = float(stake), float(price)
    if not np.isfinite(amount) or amount <= 0:
        raise LedgerError(f"A stake must be a positive number, got {stake!r}.")
    if not np.isfinite(odds) or odds <= MIN_PRICE:
        raise LedgerError(
            f"A price must exceed {MIN_PRICE}, got {price!r}. At 1.0 the stake comes back and "
            "nothing else, so anything at or below it is not a bet."
        )
    if not str(label).strip():
        raise LedgerError("Every stake carries a label, so the log can be read by strategy.")

    ledger = load(path)
    row = {
        "recorded_at": moment.isoformat(timespec="seconds"),
        "event_date": when, "label": str(label), "selection": str(selection),
        "stake": amount, "price": odds, "note": str(note),
        "settled_at": np.nan, "won": np.nan, "payout": np.nan,
    }
    updated = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    write(updated, path)
    return updated


def settle_pending(path: str, resolve: Callable[[Any], dict[str, Any] | None]) -> pd.DataFrame:
    """Settle every open row whose event has happened. All of them, every time.

    `resolve(row)` returns `{"won": bool, "payout": float}` — payout being the
    **money returned**, stake included, so a loser is 0.0 and an even-money
    winner on 1 unit is 2.0 — or None when the event has not resolved yet. Those
    are the only two cases: a domain that wants to skip a row for any other
    reason is asking for the subset settling this module refuses.
    """
    ledger = load(path)
    if ledger.empty:
        return ledger

    settled = 0
    for i, row in ledger.iterrows():
        if pd.notna(row[SETTLED_AT]):
            continue
        result = resolve(row)
        if result is None:
            continue
        missing = [c for c in RESULT_COLUMNS if c not in result]
        if missing:
            raise LedgerError(
                f"Settling returned no value for {missing}. A half-settled row reads as a "
                "settled one and would be counted as evidence."
            )
        payout = float(result["payout"])
        if not np.isfinite(payout) or payout < 0:
            raise LedgerError(f"A payout must be finite and non-negative, got {result['payout']!r}.")

        ledger.loc[i, SETTLED_AT] = _now().isoformat(timespec="seconds")
        ledger.loc[i, "won"] = bool(result["won"])
        ledger.loc[i, "payout"] = payout
        settled += 1

    if settled:
        write(ledger, path)
    return ledger


def open_bets(path: str | None = None, ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stakes whose event has not happened, or has not been settled yet."""
    if ledger is None:
        if path is None:
            raise LedgerError("open_bets() needs either a path or a ledger frame.")
        ledger = load(path)
    return ledger[ledger[SETTLED_AT].isna()].sort_values("event_date")


def returns(settled: pd.DataFrame) -> np.ndarray:
    """Profit per unit staked, one number per settled bet.

    The per-unit form rather than the money is what makes bets of different
    sizes comparable, and it is what the test below is run on: a ledger with one
    huge bet and forty small ones is not forty-one observations of anything
    unless they are on the same scale.
    """
    stake = settled["stake"].astype(float).to_numpy()
    payout = settled["payout"].astype(float).to_numpy()
    return (payout - stake) / np.where(stake > 0, stake, np.nan)


def minimum_detectable_return(n_bets: int, observed_sd: float, alpha: float = 0.05,
                              power: float = 0.8) -> float:
    """The smallest true per-unit return this many bets could have resolved.

    The other half of a null result, exactly as `lottery/analysis/power.py` is
    for the chance test: "no edge" and "no edge detectable here" are different
    findings and a log that reports the first when it means the second has
    misled its reader. Returns NaN below two bets, where the question has no
    answer rather than a large one.
    """
    from scipy import stats

    if n_bets < 2 or not np.isfinite(observed_sd) or observed_sd <= 0:
        return float("nan")
    z_alpha = float(stats.norm.ppf(1.0 - alpha))
    z_power = float(stats.norm.ppf(power))
    return float((z_alpha + z_power) * observed_sd / np.sqrt(n_bets))


def summary(path: str | None = None, ledger: pd.DataFrame | None = None,
            by_label: bool = False, alpha: float = 0.05) -> pd.DataFrame:
    """Money and verdict in one row, with the verdict first and the resolution beside it.

    The column order is part of the contract: `beats_breakeven_corrected`,
    `p_value_greater` and `min_detectable_return` come before `profit` and
    `roi`, so a surface that renders the frame in order cannot lead with the
    total. `by_label=True` tests each strategy and corrects across them, because
    reading the best of five labels is the same mistake as reading the best of
    five models.
    """
    if ledger is None:
        if path is None:
            raise LedgerError("summary() needs either a path or a ledger frame.")
        ledger = load(path)

    settled = ledger[ledger[SETTLED_AT].notna()] if not ledger.empty else ledger
    columns = ["label", "n_settled", "beats_breakeven", "beats_breakeven_corrected",
               "p_value_greater", "min_detectable_return", "roi", "ci_low", "ci_high",
               "staked", "returned", "profit", "hit_rate", "bonferroni_threshold"]
    if settled.empty:
        return pd.DataFrame(columns=columns)

    groups = (list(settled.groupby("label")) if by_label else [("all", settled)])
    threshold = bonferroni_threshold(alpha, max(len(groups), 1))

    rows = []
    for label, group in groups:
        per_unit = returns(group)
        usable = per_unit[np.isfinite(per_unit)]
        spread = float(np.var(usable, ddof=1)) if usable.size > 1 else 0.0
        test = z_test_against_null(usable, null_means=0.0, null_variances=spread)
        verdict = verdicts(test["p_value_greater"], alpha, threshold)
        staked = float(group["stake"].astype(float).sum())
        returned = float(group["payout"].astype(float).sum())
        rows.append({
            "label": label,
            "n_settled": int(len(group)),
            "beats_breakeven": verdict["beats_chance"],
            "beats_breakeven_corrected": verdict["beats_chance_corrected"],
            "p_value_greater": test["p_value_greater"],
            "min_detectable_return": minimum_detectable_return(
                len(usable), float(np.sqrt(spread)), alpha=alpha),
            "roi": float(usable.mean()) if usable.size else float("nan"),
            "ci_low": test["ci_low"],
            "ci_high": test["ci_high"],
            "staked": staked,
            "returned": returned,
            "profit": returned - staked,
            "hit_rate": float(group["won"].astype(str).str.lower().isin(("true", "1")).mean()),
            "bonferroni_threshold": threshold,
        })
    return pd.DataFrame(rows)[columns]


def status(path: str) -> dict[str, Any]:
    """Counts and dates for a one-glance answer, saying nothing about how it went.

    `summary` is where the verdict lives; this is deliberately mute about
    performance, the same split `core/registry.py:status` makes.
    """
    ledger = load(path)
    settled = ledger[ledger[SETTLED_AT].notna()] if not ledger.empty else ledger
    still_open = open_bets(ledger=ledger) if not ledger.empty else ledger
    next_event = (still_open["event_date"].min() if not still_open.empty else None)
    return {
        "path": path,
        "exists": os.path.exists(path),
        "n_rows": int(len(ledger)),
        "n_settled": int(len(settled)),
        "n_open": int(len(still_open)),
        "next_event": None if next_event is None or pd.isna(next_event) else next_event,
        "first_recorded": (ledger["recorded_at"].min() if not ledger.empty else None),
    }
