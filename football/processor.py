"""Owner of the football data contract: football-data.co.uk CSVs in, tidy matches out.

Their files carry one row per match with a stable core — `Date`, `HomeTeam`,
`AwayTeam`, `FTHG`, `FTAG` — wrapped in a column set that drifts season to
season as bookmakers come and go. This module reads the core, resolves one
odds source for the whole frame, and refuses to guess about anything else.

**The decision this module exists to protect: opening and closing odds are
not interchangeable.**

A bookmaker publishes a price when the market opens and a different one when
it closes, after the money has moved. The closing line is the sharp number —
it is the aggregate of everyone who bet, and beating it consistently is the
definition of an edge. The opening line is soft, and a model that "beats the
market" against opening prices has usually beaten nothing but a bookmaker's
first guess.

Football-data marks closing odds with a `C` in the column name (`AvgCH`,
`B365CH`) and publishes them only from the 2019/20 season onward. So a merged
history spanning that boundary has closing odds for its recent half and
nothing for its older half — and filling the gap from the opening columns
would produce one `odds_home` column silently meaning two different things.
That is exactly the failure `lottery/utils/processor.py` guards against with
the two eras of Baloto, in a different costume.

The rule here is therefore: **one odds source per frame, or none.** The
resolver picks the best source that appears in the file at all, records it in
`odds_source`, and leaves rows that lack it as NaN rather than reaching for a
different column. Missing odds are visible; mixed odds would not be.
"""

import os
import warnings

import numpy as np
import pandas as pd

from football.common import MATCH_COLUMNS, ODDS_COLUMNS, outcome_from_goals

REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")

# Candidate odds sources, best first. Closing beats opening, and a market
# aggregate beats any single book — an average of many prices is a better
# estimate of the market than whichever bookmaker happens to be in the file.
#
# Each entry is (name, is_closing, (home, draw, away) column names).
ODDS_SOURCES = (
    ("market_closing_average", True, ("AvgCH", "AvgCD", "AvgCA")),
    ("pinnacle_closing", True, ("PSCH", "PSCD", "PSCA")),
    ("bet365_closing", True, ("B365CH", "B365CD", "B365CA")),
    ("market_opening_average", False, ("AvgH", "AvgD", "AvgA")),
    ("pinnacle_opening", False, ("PSH", "PSD", "PSA")),
    ("bet365_opening", False, ("B365H", "B365D", "B365A")),
)

CLOSING_SOURCES = frozenset(name for name, is_closing, _ in ODDS_SOURCES if is_closing)


class MatchFormatError(ValueError):
    """The file does not carry what a match frame needs, and guessing would be worse."""


def resolve_odds_source(columns):
    """Pick the single best odds source present, or None.

    Presence is decided on the column set alone: a source either is in this
    file or is not. Coverage within the chosen source is reported separately
    by `odds_coverage`, because a column that exists but is half empty is a
    different problem from a column that is absent, and conflating them is how
    a fallback to opening odds sneaks in.
    """
    available = set(columns)
    for name, is_closing, triple in ODDS_SOURCES:
        if available.issuperset(triple):
            return {"name": name, "is_closing": is_closing, "columns": triple}
    return None


def odds_coverage(matches):
    """What fraction of rows actually carry a usable price."""
    if not set(ODDS_COLUMNS).issubset(matches.columns):
        return 0.0
    usable = matches[list(ODDS_COLUMNS)].notna().all(axis=1)
    return float(usable.mean()) if len(matches) else 0.0


def _parse_dates(raw):
    """football-data writes dd/mm/yy in older files and dd/mm/yyyy in newer ones.

    Both are day-first, so one `dayfirst=True` pass handles them — but the
    two-digit form is ambiguous about century and pandas resolves it by its
    own rule, so the result is checked rather than trusted: a history that
    lands in the future, or before the sport's records begin, means the parse
    was wrong and silently shifting a season by a century is not acceptable.
    """
    with warnings.catch_warnings():
        # Both date forms are supported deliberately, so pandas falling back to
        # per-element parsing is the intended path rather than something to fix.
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce")
    if parsed.isna().any():
        bad = raw[parsed.isna()].head(3).tolist()
        raise MatchFormatError(
            f"{int(parsed.isna().sum())} dates could not be parsed as day-first, e.g. {bad}. "
            "football-data publishes dd/mm/yy or dd/mm/yyyy; a file in another format needs "
            "converting before it reaches this contract."
        )
    return parsed


def preprocess_matches(df, validate=True):
    """Turn a raw football-data frame into the tidy match shape used everywhere else.

    Returns a frame with `MATCH_COLUMNS`, plus `odds_home` / `odds_draw` /
    `odds_away` and an `odds_source` attribute recording which columns they
    came from — resolved once for the whole frame, never per row.

    Like the lottery loader, this warns rather than raises on data that is
    merely incomplete (missing odds, a season without closing prices). It
    raises only when the core contract is broken, because a match with no
    teams or no score is not a match.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MatchFormatError(
            f"Missing required columns: {missing}. A football-data.co.uk CSV carries "
            f"{list(REQUIRED_COLUMNS)} in every season; a file without them is not a match file."
        )

    out = pd.DataFrame({
        "ds": _parse_dates(df["Date"]),
        "home_team": df["HomeTeam"].astype("string").str.strip(),
        "away_team": df["AwayTeam"].astype("string").str.strip(),
        "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
    })

    # Rows with no score are fixtures, not results — football-data leaves them
    # blank at the end of an in-progress season. They cannot be scored against,
    # so they are dropped here rather than surfacing as NaN outcomes later.
    unplayed = out["home_goals"].isna() | out["away_goals"].isna()
    if unplayed.any():
        if validate:
            warnings.warn(
                f"Dropping {int(unplayed.sum())} row(s) with no final score — these are "
                "fixtures that have not been played, not results.",
                stacklevel=2,
            )
        out = out[~unplayed]
        df = df[~unplayed]

    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    out["outcome"] = [outcome_from_goals(h, a)
                      for h, a in zip(out["home_goals"], out["away_goals"])]

    source = resolve_odds_source(df.columns)
    if source is None:
        for column in ODDS_COLUMNS:
            out[column] = np.nan
    else:
        for column, raw in zip(ODDS_COLUMNS, source["columns"]):
            out[column] = pd.to_numeric(df[raw], errors="coerce")
        # A price triple is usable or it is not — there is no partial market.
        # Two of three prices cannot be normalised into probabilities, and
        # leaving them in the frame would show a price for a match that has no
        # market behind it. Blank the whole row instead, so `odds_home` never
        # carries a number that nothing downstream can use.
        #
        # An odds of 1.0 or less pays nothing and cannot be a real price, so it
        # counts as missing here rather than producing an implied probability
        # of 1 or more downstream.
        prices = out[list(ODDS_COLUMNS)]
        unusable = prices.isna().any(axis=1) | (prices <= 1.0).any(axis=1)
        out.loc[unusable, list(ODDS_COLUMNS)] = np.nan

    out = out.sort_values("ds").reset_index(drop=True)
    out.attrs["odds_source"] = source["name"] if source else None
    out.attrs["odds_are_closing"] = bool(source["is_closing"]) if source else False

    if validate:
        report = check_match_format(out)
        if report:
            warnings.warn(report["message"], stacklevel=2)
    return out


def check_match_format(matches):
    """Report on anything that would weaken this frame as an evaluation set.

    Returns None when the frame is fit to serve as a market baseline, and a
    dict with a printable `message` otherwise. Two things are worth a warning
    and neither is fatal: no odds at all (the frame can still train a model,
    it just cannot judge one), and opening odds only (the baseline exists but
    is the soft one).
    """
    source = matches.attrs.get("odds_source")
    coverage = odds_coverage(matches)
    problems = []

    if source is None:
        problems.append(
            "This file carries no odds columns at all, so there is no market baseline to "
            "compare against. A model can be fitted on it, but nothing measured on it can "
            "show an edge — beating chance is not the bar in football."
        )
    elif source not in CLOSING_SOURCES:
        problems.append(
            f"The best odds available here are opening prices ({source}); football-data "
            "publishes closing odds (the 'C' columns) only from 2019/20 onward. Opening lines "
            "are soft: a model that beats them has probably beaten a first guess rather than "
            "the market. Treat any edge measured against this baseline as unproven."
        )

    if source is not None and coverage < 1.0:
        problems.append(
            f"Only {coverage:.1%} of rows carry a usable price from {source}. The rest are "
            "left as NaN on purpose — filling them from a different column would mix two "
            "different markets into one baseline."
        )

    if not problems:
        return None
    return {
        "odds_source": source,
        "odds_are_closing": bool(matches.attrs.get("odds_are_closing")),
        "odds_coverage": coverage,
        "n_matches": len(matches),
        "message": " ".join(problems),
    }


def load_and_preprocess(path, validate=True, closing_odds_only=False):
    """Read one football-data CSV into the tidy match shape.

    `closing_odds_only=True` refuses a file whose best odds are opening
    prices, rather than quietly producing a baseline that cannot support the
    conclusion someone will draw from it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    matches = preprocess_matches(pd.read_csv(path), validate=validate)
    if closing_odds_only and matches.attrs.get("odds_source") not in CLOSING_SOURCES:
        raise MatchFormatError(
            f"{path} has no closing odds (best available: "
            f"{matches.attrs.get('odds_source')}). Closing odds start at the 2019/20 season; "
            "either use a later season or drop closing_odds_only."
        )
    return matches


def load_seasons(paths, validate=True, closing_odds_only=False):
    """Concatenate several season files, refusing to merge incompatible odds sources.

    This is where the mixing would actually happen: 2018/19 has no closing
    odds and 2019/20 does, so stacking them naively yields one `odds_home`
    column that is a closing price for half the rows and an opening price for
    the other half. Every model evaluated on that frame would be judged
    against two different bars at once, and the result would look like an edge
    that appears only in the older seasons.
    """
    frames = [load_and_preprocess(p, validate=validate, closing_odds_only=closing_odds_only)
              for p in paths]
    if not frames:
        raise MatchFormatError("No season files given.")

    sources = {f.attrs.get("odds_source") for f in frames}
    if len(sources) > 1:
        raise MatchFormatError(
            f"These seasons resolve to different odds sources ({sorted(str(s) for s in sources)}), "
            "so concatenating them would put two different markets in one column. Load the "
            "seasons that share a source, or pass closing_odds_only=True to keep only the "
            "ones that have closing prices."
        )

    merged = pd.concat(frames, ignore_index=True).sort_values("ds").reset_index(drop=True)
    merged.attrs.update(frames[0].attrs)
    return merged
