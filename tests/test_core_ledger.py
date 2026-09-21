"""The stake ledger — core/ledger.py.

Three kinds of test, and the third is the one that matters most.

**The refusals**, which are `core/registry.py`'s restated for money: a stake on
an event that has already happened, a second write that edits a settled row, and
settling a subset.

**The arithmetic**, which is unglamorous and easy to get subtly wrong — a payout
is money *returned*, stake included, so a loser is 0 and break-even is exactly
the stake back.

**The headline rule.** `summary`'s column order is part of its contract: the
corrected verdict, the p-value and the minimum detectable return come before
the money, so a surface rendering the frame in order cannot lead with a profit
figure. That is not decoration — a P&L at the top of a betting log is the single
most misleading number this project could show, because it looks like evidence,
moves daily, and means nothing at the sample sizes anyone reaches.
"""

import os
import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from core.ledger import (
    COLUMNS,
    LedgerError,
    load,
    minimum_detectable_return,
    open_bets,
    record,
    returns,
    settle_pending,
    status,
    summary,
)

NOW = datetime(2024, 5, 1, tzinfo=UTC)


@pytest.fixture
def path(tmp_path):
    return os.path.join(str(tmp_path), "bets.csv")


def add(path, date="2024-06-01", label="modelo", selection="Team A", stake=1.0, price=2.0):
    return record(path, date, label, selection, stake, price, now=NOW)


def test_a_ledger_starts_empty_with_the_right_columns(path):
    assert list(load(path).columns) == list(COLUMNS)
    assert status(path)["n_rows"] == 0


def test_a_stake_on_an_event_that_already_happened_is_refused(path):
    # The refusal the whole file rests on, and the easiest thing in the world to
    # do by accident with a date picker.
    with pytest.raises(LedgerError, match="not in the future"):
        add(path, date="2024-04-30")
    assert not os.path.exists(path)


def test_the_comparison_is_by_day_not_by_clock(path):
    # An event date carries no time, so comparing a midnight timestamp against
    # the clock refuses everything later today — including tonight's match. The
    # rule is `core/registry.py`'s: normalise to a day, refuse today or earlier.
    # Today stays refused because a date-only "today" may already have happened.
    with pytest.raises(LedgerError, match="today is 2024-05-01"):
        record(path, "2024-05-01", "m", "tonight", 1.0, 2.0,
               now=datetime(2024, 5, 1, 9, tzinfo=UTC))
    # ... and tomorrow is accepted whatever the hour, which is what stops the
    # dashboard form erroring on its own default.
    record(path, "2024-05-02", "m", "tomorrow", 1.0, 2.0,
           now=datetime(2024, 5, 1, 23, 59, tzinfo=UTC))
    assert len(load(path)) == 1


def test_a_stake_and_a_price_have_to_be_real(path):
    for stake in (0.0, -1.0, float("nan")):
        with pytest.raises(LedgerError, match="positive number"):
            add(path, stake=stake)
    for price in (1.0, 0.5, float("inf")):
        with pytest.raises(LedgerError, match="price must exceed"):
            add(path, price=price)
    with pytest.raises(LedgerError, match="label"):
        add(path, label="   ")


def test_a_recorded_bet_is_open_until_it_is_settled(path):
    add(path)
    assert len(open_bets(path)) == 1
    assert status(path)["n_open"] == 1

    settle_pending(path, lambda row: {"won": True, "payout": 2.0})
    assert open_bets(path).empty
    assert status(path)["n_settled"] == 1


def test_a_payout_is_money_returned_not_profit(path):
    # The unglamorous arithmetic: a loser returns 0, break-even returns the
    # stake, and an even-money winner on 1 unit returns 2.
    add(path, selection="loser")
    settle_pending(path, lambda row: {"won": False, "payout": 0.0})
    settled = load(path)
    assert returns(settled) == pytest.approx([-1.0])


def test_returns_are_per_unit_so_bets_of_different_sizes_are_comparable(path):
    add(path, selection="small", stake=1.0, price=3.0)
    add(path, selection="large", stake=10.0, price=3.0)
    settle_pending(path, lambda row: {"won": True, "payout": float(row["stake"]) * 3.0})
    # Both won at 3.0, so both are +2 per unit — a ledger with one huge bet and
    # one small one is not two observations of anything otherwise.
    assert returns(load(path)) == pytest.approx([2.0, 2.0])


def test_a_half_settled_row_is_refused(path):
    add(path)
    with pytest.raises(LedgerError, match="no value for"):
        settle_pending(path, lambda row: {"won": True})


def test_an_unresolved_event_is_skipped_and_nothing_else_is(path):
    add(path, date="2024-06-01", selection="played")
    add(path, date="2024-07-01", selection="not yet")
    settle_pending(path, lambda row: ({"won": True, "payout": 2.0}
                                      if row["selection"] == "played" else None))
    assert list(open_bets(path)["selection"]) == ["not yet"]


def test_resettling_cannot_rewrite_a_settled_row(path):
    add(path)
    settle_pending(path, lambda row: {"won": True, "payout": 2.0})
    settle_pending(path, lambda row: {"won": False, "payout": 0.0})
    assert bool(load(path)["won"].iloc[0]) is True
    assert float(load(path)["payout"].iloc[0]) == 2.0


def test_the_summary_puts_the_verdict_before_the_money(path):
    # The contract this module exists to hold: a surface that renders the frame
    # in column order cannot lead with a profit figure.
    add(path)
    settle_pending(path, lambda row: {"won": True, "payout": 2.0})
    columns = list(summary(path).columns)
    for early in ("beats_breakeven_corrected", "p_value_greater", "min_detectable_return"):
        assert columns.index(early) < columns.index("profit")
        assert columns.index(early) < columns.index("roi") or early == "min_detectable_return"


def test_a_lucky_run_does_not_beat_break_even(path):
    # Six winners at evens is +100% ROI and proves nothing. This is the number a
    # betting log would put at the top, and the verdict beside it is why it must
    # not be the headline.
    for i in range(6):
        add(path, date=f"2024-06-0{i + 1}", selection=f"bet {i}")
    settle_pending(path, lambda row: {"won": True, "payout": 2.0})

    row = summary(path).iloc[0]
    assert row["roi"] == pytest.approx(1.0)
    assert row["profit"] == pytest.approx(6.0)
    # Every bet won, so the realised spread is zero and the test cannot speak;
    # what it must not do is call that a verdict.
    assert not row["beats_breakeven_corrected"]


def test_a_real_edge_is_found_once_there_is_enough_of_it(path):
    # Sixty bets at 2.0 winning 65% of the time is a genuine +30% return, and the
    # test should say so — a detector that never fires is not a conservative
    # detector, it is a broken one.
    rng = np.random.default_rng(0)
    won = rng.random(60) < 0.65
    for i in range(60):
        add(path, date=f"2024-06-{i % 28 + 1:02d}", selection=f"bet {i}")
    outcomes = iter(won)
    settle_pending(path, lambda row: {"won": bool(w := next(outcomes)), "payout": 2.0 if w else 0.0})

    row = summary(path).iloc[0]
    assert row["roi"] > 0.1
    assert row["beats_breakeven_corrected"]
    assert row["ci_low"] > 0


def test_the_minimum_detectable_return_says_what_a_short_log_could_not_see():
    # 10 bets at a per-bet spread of 1.0 cannot resolve anything under ~78%.
    small = minimum_detectable_return(10, 1.0)
    large = minimum_detectable_return(250, 1.0)
    assert small > 0.7 and large < small
    # It falls with the square root of N, so four times the bets halves it.
    assert minimum_detectable_return(40, 1.0) == pytest.approx(small / 2, rel=1e-6)
    assert np.isnan(minimum_detectable_return(1, 1.0))


def test_labels_are_corrected_against_each_other(path):
    for i, label in enumerate(("a", "b", "c")):
        add(path, date=f"2024-06-0{i + 1}", label=label, selection=label)
    settle_pending(path, lambda row: {"won": True, "payout": 2.0})

    table = summary(path, by_label=True)
    assert set(table["label"]) == {"a", "b", "c"}
    # Reading the best of three labels is the same mistake as reading the best
    # of three models, so the threshold tightens with the count.
    assert table["bonferroni_threshold"].iloc[0] == pytest.approx(0.05 / 3)


def test_recording_and_settling_raise_no_pandas_deprecation(path):
    """The columns settling writes into must not be float64.

    An all-empty column reads back as float64 and settling then puts a string
    and a bool in it: pandas 2 warns and upcasts, pandas 3 raises, and
    `pandas>=2.2` in the requirements allows pandas 3. The same deprecation
    covers concatenating onto an all-NA empty frame, which is every ledger's
    first row. Both are errors here so neither can come back quietly.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        add(path)
        settle_pending(path, lambda row: {"won": True, "payout": 2.0})

    stored = load(path)
    assert stored["won"].dtype == object
    assert stored["settled_at"].dtype == object
    # And the string round trip `summary`'s hit rate depends on still works.
    assert summary(path).iloc[0]["hit_rate"] == pytest.approx(1.0)


def test_an_empty_ledger_summarises_to_nothing_rather_than_to_zero_profit(path):
    empty = summary(path)
    assert empty.empty
    assert "profit" in empty.columns
    assert pd.isna(status(path)["next_event"])
