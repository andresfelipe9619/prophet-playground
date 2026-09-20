"""Owner of the cycling price contract: quoted outright odds in, one book out.

`cycling/market.py` does the arithmetic; this owns the file, the same way
`football/processor.py` owns football's odds contract and `market.py` there
owns the de-margining. Splitting them is not tidiness: the arithmetic is
testable on a vector of numbers, and the mistakes that actually happen are
about *which* numbers ended up in the vector.

**The one guard that matters is one book per frame.** This is football's
opening/closing trap in a cycling costume and it is worse here. Two bookmakers
quoting the same Tour do not merely differ in margin; they differ in which
riders they quote at all, and a frame stacking both has a field of 240 runners
for a race with 176 starters and an overround that means nothing. Every
de-margining method in `market.py` normalises over the field it is handed, so a
mixed frame produces a baseline that is confidently wrong about every rider in
it, and nothing in the frame's shape would say so. `preprocess_prices` refuses
a second `book` value and `load_books` refuses to concatenate files that
resolve to different ones.

**One market per frame, for the same reason `preprocess_results` allows one
result kind.** A price on the Tour's general classification and a price on its
seventh stage are prices on different events. Stacked, they normalise against
each other, and the resulting "probabilities" describe a race nobody ran.

**Quoted, not published as a probability.** A book prices who wins and stops
there. Nothing in a price file says how the market would order the rest of the
field, so this module carries prices and `market.worths_from_market` states the
extrapolation — it is not restated here.

**Nothing in this repository fetches prices.** There is no scraper, and
`docs/cycling.md` says so rather than implying a pipeline. What exists is the
shape a price file must have for the verdict to be against the market instead
of against the ranking, so that the day prices are available the baseline is
already expressible.
"""

import os

import numpy as np
import pandas as pd

from cycling.common import RESULT_KINDS
from cycling.market import MIN_DECIMAL_ODDS, overround

REQUIRED_COLUMNS = ("Date", "Race", "Kind", "Rider", "Odds", "Book")
OPTIONAL_COLUMNS = ("Stage",)

PRICE_COLUMNS = ["ds", "race", "kind", "stage", "rider", "odds", "book"]

# The keys that identify one market: prices on two of these are prices on two
# different events and must never normalise against each other.
MARKET_KEYS = ["race", "kind", "stage"]


class PriceFormatError(ValueError):
    """The file does not describe one book on one race, and guessing would be worse."""


def _require_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PriceFormatError(
            f"Missing column(s) {missing}. A price file carries {list(REQUIRED_COLUMNS)}"
            f" and optionally {list(OPTIONAL_COLUMNS)}."
        )


def preprocess_prices(df, validate=True):
    """Raw quoted prices into the tidy shape, or a refusal.

    Returns `PRICE_COLUMNS` with `attrs["book"]`, `attrs["market"]` and
    `attrs["overround"]`. The book and the market are recorded on the frame
    rather than inferred later, so a downstream concatenation has something to
    refuse on — the football lesson: a guard that has to re-derive what a frame
    is will eventually derive it from a frame that is already mixed.
    """
    _require_columns(df)
    out = pd.DataFrame({
        "ds": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "race": df["Race"].astype(str).str.strip(),
        "kind": df["Kind"].astype(str).str.strip(),
        "stage": pd.to_numeric(df["Stage"], errors="coerce") if "Stage" in df else np.nan,
        "rider": df["Rider"].astype(str).str.strip(),
        "odds": pd.to_numeric(df["Odds"], errors="coerce"),
        "book": df["Book"].astype(str).str.strip(),
    })[PRICE_COLUMNS]

    if out["ds"].isna().any():
        bad = df.loc[out["ds"].isna(), "Date"].head(3).tolist()
        raise PriceFormatError(
            f"{int(out['ds'].isna().sum())} date(s) could not be parsed as day-first, e.g. {bad}."
        )

    unknown = sorted(set(out["kind"]) - set(RESULT_KINDS))
    if unknown:
        raise PriceFormatError(
            f"Unknown result kind(s) {unknown}. A price is quoted on one of {list(RESULT_KINDS)},"
            " and a kind this module does not recognise is a kind it cannot match to a result."
        )

    books = sorted(set(out["book"]))
    if len(books) > 1:
        raise PriceFormatError(
            f"This frame holds prices from {len(books)} books ({books}). Two books quote "
            "different fields at different margins, so normalising them together produces a "
            "field that never started and an overround that means nothing. Load one book at a time."
        )

    markets = out[MARKET_KEYS].drop_duplicates(ignore_index=True)
    if len(markets) > 1:
        described = markets.to_dict("records")
        raise PriceFormatError(
            f"This frame holds prices on {len(markets)} different markets ({described}). A price "
            "on a general classification and a price on a stage are prices on different events; "
            "normalised against each other they describe a race nobody ran."
        )

    if validate:
        _validate(out)

    out.attrs["book"] = books[0] if books else None
    out.attrs["market"] = markets.iloc[0].to_dict() if len(markets) else None
    out.attrs["overround"] = overround(out["odds"].to_numpy())
    out.attrs["n_runners"] = int(out["odds"].notna().sum())
    return out.sort_values("odds").reset_index(drop=True)


def _validate(prices):
    """The checks that make a price frame a market rather than a column of numbers."""
    if prices["rider"].duplicated().any():
        repeated = sorted(prices.loc[prices["rider"].duplicated(), "rider"].unique())[:3]
        raise PriceFormatError(
            f"The same rider is priced more than once in one market, e.g. {repeated}. Two quotes "
            "for one runner is either a stale row or two books, and both are refusals here."
        )

    quoted = prices["odds"].dropna()
    if quoted.empty:
        raise PriceFormatError("No usable prices in this frame.")
    if (quoted <= MIN_DECIMAL_ODDS).any():
        raise PriceFormatError(
            f"Decimal odds must exceed {MIN_DECIMAL_ODDS}; the shortest here is {quoted.min():.3f}."
            " A fractional or American column that has not been converted looks exactly like this."
        )

    total = overround(quoted.to_numpy())
    if total <= 1.0:
        raise PriceFormatError(
            f"These prices imply a total of {total:.3f}, at or under 1.0. An outright book always "
            "overrounds, so a total below 1 means the field is partial — and normalising a partial "
            "field spreads the absent riders' probability over the ones that are present."
        )


def load_prices(path, validate=True):
    """Read one price file for one book on one market."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    return preprocess_prices(pd.read_csv(path), validate=validate)


def load_books(paths, validate=True):
    """Concatenate price files, refusing to merge different books.

    This is where the mixing would actually happen: one directory, one race,
    one file per bookmaker, named alike. Markets may differ across files — a
    season of prices is many races — but the book may not, because that is the
    difference the arithmetic downstream cannot see.
    """
    frames = [load_prices(p, validate=validate) for p in paths]
    if not frames:
        raise PriceFormatError("No price files given.")

    books = {f.attrs.get("book") for f in frames}
    if len(books) > 1:
        raise PriceFormatError(
            f"These files hold prices from different books ({sorted(str(b) for b in books)}), so "
            "concatenating them would put two margins and two fields in one market. Load one book "
            "at a time."
        )

    merged = (pd.concat(frames, ignore_index=True)
                .sort_values(["ds", "race", "stage", "odds"], na_position="last")
                .reset_index(drop=True))
    merged.attrs["book"] = frames[0].attrs.get("book")
    merged.attrs["market"] = None if len(frames) > 1 else frames[0].attrs.get("market")
    merged.attrs["n_markets"] = int(merged[MARKET_KEYS].drop_duplicates().shape[0])
    return merged


def market_slice(prices, race, kind, stage=None):
    """The one market's prices out of a multi-race frame, as its own frame.

    Returned with the same `attrs` a single-market file would carry, because
    everything downstream normalises over a field and must be handed one.
    """
    rows = prices[(prices["race"] == race) & (prices["kind"] == kind)]
    rows = rows[rows["stage"].isna()] if stage is None else rows[rows["stage"] == stage]
    if rows.empty:
        raise PriceFormatError(f"No prices for {race!r} ({kind}, stage {stage}).")

    out = rows.reset_index(drop=True)
    out.attrs["book"] = prices.attrs.get("book")
    out.attrs["market"] = {"race": race, "kind": kind, "stage": stage}
    out.attrs["overround"] = overround(out["odds"].to_numpy())
    out.attrs["n_runners"] = int(out["odds"].notna().sum())
    return out


def market_forecaster(prices, method=None, unpriced="longest"):
    """A `walk_forward` forecaster that reads the book instead of the results.

    Returns `f(history, riders, as_of) -> worths`, the shape
    `cycling/evaluation.py` already takes, so the market slots into
    `compare_forecasters` as one more challenger — or, where prices exist, as
    the **baseline**, which is the whole reason this module exists. Every
    cycling verdict in this project is against the pre-race ranking because the
    hard bar could not be expressed; handed a price file, it now can.

    **It looks nothing at `history`**, which is the point: a price quoted before
    the race is already a forecast made without the result, so there is no
    `as_of` to enforce beyond finding the right market. The date is what
    identifies it, and two races priced on the same day are **not scored**
    rather than guessed between — returning None drops that race from both
    sides of the paired test, which is the behaviour `walk_forward` already has
    for a forecaster that cannot answer.
    """
    from cycling.market import DEFAULT_METHOD, field_worths

    method = DEFAULT_METHOD if method is None else method
    table = prices.dropna(subset=["odds"])

    def forecast(history, riders, as_of):  # noqa: ARG001 — a price needs no history
        day = table[table["ds"] == pd.Timestamp(as_of)]
        if day.empty or day[MARKET_KEYS].drop_duplicates().shape[0] != 1:
            return None
        worths, _ = field_worths(riders, list(day["rider"]), day["odds"].to_numpy(),
                                 method=method, unpriced=unpriced)
        return worths

    return forecast
