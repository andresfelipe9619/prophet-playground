"""The football-data.co.uk "extra" file contract: new/COL.csv and its siblings.

Their main league files carry HomeTeam/AwayTeam/FTHG/FTAG, one league per file,
and — from 2019/20 — closing odds. The "extra" files for the rest of the world
are a different animal: Home/Away/HG/AG, many leagues and seasons stacked in
one download, and OPENING ODDS ONLY (AvgH/D/A market average, PH/D/A Pinnacle,
sometimes B365H/D/A). `football/processor.py` refuses them; this module reads
them onto the same tidy frame, with one rule enforced hard: the odds source is
always an opening one, so `odds_are_closing` is always False and no evaluation
built on these files can claim a corrected edge.

One league per load. The file physically contains several, and stacking (say)
Colombia's Primera A and Primera B is the same mistake as merging opening and
closing odds — a `league` argument is required whenever the file has more than
one.
"""

import os
import warnings

import numpy as np
import pandas as pd

from football.common import ODDS_COLUMNS, outcome_from_goals
from football.processor import MatchFormatError, _parse_dates, check_match_format

REQUIRED_COLUMNS = ("Date", "Home", "Away", "HG", "AG")

# All opening. Best first: market average, then Pinnacle, then Bet365.
EXTRA_ODDS_SOURCES = (
    ("extra_market_average_opening", ("AvgH", "AvgD", "AvgA")),
    ("extra_pinnacle_opening", ("PH", "PD", "PA")),
    ("extra_bet365_opening", ("B365H", "B365D", "B365A")),
)


def available_leagues(path):
    """The distinct `League` values in an extra file, for a UI selector."""
    frame = pd.read_csv(path, usecols=["League"])
    return sorted(frame["League"].dropna().unique().tolist())


def _resolve_extra_source(columns):
    available = set(columns)
    for name, triple in EXTRA_ODDS_SOURCES:
        if available.issuperset(triple):
            return name, triple
    return None, None


def preprocess_extra(df, league=None, validate=True):
    """Turn a raw extra-file frame into the tidy match shape, opening odds only.

    Returns a frame with `MATCH_COLUMNS` plus the three `ODDS_COLUMNS` — the
    same shape as `football/processor.py:preprocess_matches`. `attrs["odds_source"]`
    is always an `extra_*_opening` name (or None if the file carries no prices)
    and `attrs["odds_are_closing"]` is always False: an extra file cannot carry
    closing odds, so nothing measured on it can support a corrected edge claim.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MatchFormatError(
            f"Missing required columns for an extra file: {missing}. Expected "
            f"{list(REQUIRED_COLUMNS)} — this is not a football-data.co.uk new/ file."
        )

    if "League" in df.columns:
        leagues = df["League"].dropna().unique().tolist()
        if len(leagues) > 1 and league is None:
            raise MatchFormatError(
                f"This file carries {len(leagues)} leagues ({leagues}). Pass league= to pick "
                "one — stacking two would put two competitions in one frame, the same mistake "
                "as mixing two odds sources."
            )
        if league is not None:
            df = df[df["League"] == league]
            if df.empty:
                raise MatchFormatError(f"No rows for league {league!r}. Available: {leagues}.")

    out = pd.DataFrame({
        "ds": _parse_dates(df["Date"]),
        "home_team": df["Home"].astype("string").str.strip(),
        "away_team": df["Away"].astype("string").str.strip(),
        "home_goals": pd.to_numeric(df["HG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["AG"], errors="coerce"),
    })

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

    name, triple = _resolve_extra_source(df.columns)
    if name is None:
        for column in ODDS_COLUMNS:
            out[column] = np.nan
    else:
        for column, raw in zip(ODDS_COLUMNS, triple):
            out[column] = pd.to_numeric(df[raw], errors="coerce")
        # A price triple is usable whole or not at all — the same rule as the
        # main contract, so downstream normalisation sees one market or none.
        prices = out[list(ODDS_COLUMNS)]
        unusable = prices.isna().any(axis=1) | (prices <= 1.0).any(axis=1)
        out.loc[unusable, list(ODDS_COLUMNS)] = np.nan

    out = out.sort_values("ds").reset_index(drop=True)
    out.attrs["odds_source"] = name
    out.attrs["odds_are_closing"] = False  # extra files never carry closing odds

    if validate:
        report = check_match_format(out)
        if report:
            warnings.warn(report["message"], stacklevel=2)
    return out


def load_extra(path, league=None, validate=True):
    """Read one extra CSV into the tidy match shape (opening odds only)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    return preprocess_extra(pd.read_csv(path), league=league, validate=validate)
