"""Fetch football-data.co.uk season files into `exported_data/football/`.

Usage:
    python -m football.downloader --seasons 2019/20..2024/25 --leagues E0
    python -m football.downloader --seasons 2324 --leagues E0,SP1,I1 --dry-run

The lottery half of this project scrapes HTML because loterias.com publishes
nothing else. football-data.co.uk publishes CSV directly, so this module
downloads rather than parses — hence `downloader`, not `scraper`. What it
keeps from `lottery/utils/scraper.py` is the posture: **fail loudly**. A
download that quietly saves an HTML error page under a `.csv` name, or a file
whose columns have drifted out of the contract, is the failure mode this is
shaped to avoid, because it surfaces months later as an analysis that has been
running on garbage.

Two decisions worth knowing before changing anything here:

**The file is written exactly as downloaded.** No column pruning, no renaming,
no tidying. `football/processor.py` owns the contract and resolves the odds
source; a downloader that pre-selected columns would become a second, weaker
owner of it — and would silently drop the bookmaker columns a future
`ODDS_SOURCES` entry might want.

**It is validated at download time anyway.** Every file is routed through
`preprocess_matches` before it is written, purely to check that it satisfies
the contract, and the resolved odds source is printed per file. That is when a
format change is cheap to notice. It is also why the summary reports
`odds_source` per file: two seasons with different sources cannot later be
loaded together (`load_seasons` raises), and seeing that at download time
beats discovering it in the middle of an evaluation.

The "extra leagues" files football-data publishes for the rest of the world
(`new/ARG.csv` and friends) are a **different contract** — `Home`/`Away`/`HG`/`AG`
instead of `HomeTeam`/`FTHG`, several leagues stacked in one file — and nothing
in `football/` reads them. This module refuses them by name rather than
downloading something the processor would reject.
"""

import argparse
import io
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from football.common import DEFAULT_DATA_DIR
from football.processor import (
    CLOSING_SOURCES,
    MatchFormatError,
    odds_coverage,
    preprocess_matches,
)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# The main-league files, which all share the contract in football/processor.py.
# Codes are football-data's own; the names are here so an error message can say
# what a code means instead of just rejecting it.
LEAGUES = {
    "E0": "England — Premier League",
    "E1": "England — Championship",
    "E2": "England — League One",
    "E3": "England — League Two",
    "EC": "England — National League",
    "SC0": "Scotland — Premiership",
    "SC1": "Scotland — Championship",
    "SC2": "Scotland — League One",
    "SC3": "Scotland — League Two",
    "D1": "Germany — Bundesliga",
    "D2": "Germany — 2. Bundesliga",
    "I1": "Italy — Serie A",
    "I2": "Italy — Serie B",
    "SP1": "Spain — La Liga",
    "SP2": "Spain — Segunda División",
    "F1": "France — Ligue 1",
    "F2": "France — Ligue 2",
    "N1": "Netherlands — Eredivisie",
    "B1": "Belgium — Jupiler League",
    "P1": "Portugal — Liga I",
    "T1": "Turkey — Süper Lig",
    "G1": "Greece — Super League",
}

# football-data's first season, and the one closing odds start at. Both are
# used in error messages rather than as silent clamps.
FIRST_SEASON_START = 1993
FIRST_CLOSING_ODDS_SEASON_START = 2019

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
}


class DownloadError(RuntimeError):
    """The file could not be fetched, or what came back was not a season file."""


# ------------------------------------------------------------------ seasons

def season_code(start_year):
    """2023 -> '2324', the season code in football-data's URLs."""
    if start_year < FIRST_SEASON_START:
        raise ValueError(
            f"football-data's archive starts at {FIRST_SEASON_START}/"
            f"{(FIRST_SEASON_START + 1) % 100:02d}; {start_year} predates it."
        )
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_start_year(code):
    """'2324' -> 2023. The inverse of `season_code`, for labelling and sorting.

    Two-digit years are ambiguous about century and football-data's archive
    spans 1993 to the present, so the rule is fixed rather than inferred:
    codes from 93 onward are 1900s, everything below is 2000s. That holds until
    2093, by which point this line is somebody else's problem.
    """
    if not re.fullmatch(r"\d{4}", code):
        raise ValueError(f"{code!r} is not a four-digit season code like '2324'.")
    first = int(code[:2])
    return (1900 + first) if first >= (FIRST_SEASON_START % 100) else (2000 + first)


def parse_season(token):
    """One season in any of the forms a person is likely to type -> its code.

    Accepts `2324`, `2023/24`, `2023-24` and a bare start year `2023`.
    """
    token = token.strip()
    if re.fullmatch(r"\d{4}", token):
        # Ambiguous by shape: '2324' is a season code, '2023' is a start year.
        # A code's halves are consecutive years, which no plausible start year
        # in this archive satisfies, so the two never collide in practice.
        first, second = int(token[:2]), int(token[2:])
        if (first + 1) % 100 == second:
            return token
        return season_code(int(token))
    match = re.fullmatch(r"(\d{4})\s*[/-]\s*(\d{2}|\d{4})", token)
    if not match:
        raise ValueError(
            f"Could not read {token!r} as a season. Use 2324, 2023/24, 2023-24 or 2023."
        )
    return season_code(int(match.group(1)))


def parse_seasons(spec):
    """A season spec -> the list of season codes it names, oldest first.

    Ranges are written with `..` (`2019/20..2023/24`), or with `-` between two
    four-digit **years** (`2019-2024`); both bounds are season *start* years, so
    `2019-2024` is 2019/20 through 2024/25 inclusive. The `-` form is deliberately restricted:
    `2023-24` is a single season and `2019-2024` is a range, and the only way to
    tell them apart is the width of the right-hand side. `2019-2020` reads as
    the single season 2019/20 — the natural reading of two consecutive years —
    so write `2019..2020` when a two-season range is what you meant.
    """
    codes = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        bounds = None
        if ".." in part:
            bounds = part.split("..", 1)
        else:
            match = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", part)
            if match and int(match.group(2)) > int(match.group(1)) + 1:
                bounds = [match.group(1), match.group(2)]

        if bounds is None:
            codes.append(parse_season(part))
            continue

        start = season_start_year(parse_season(bounds[0]))
        end = season_start_year(parse_season(bounds[1]))
        if end < start:
            raise ValueError(f"Range {part!r} ends before it starts.")
        codes.extend(season_code(y) for y in range(start, end + 1))

    if not codes:
        raise ValueError(f"No seasons named in {spec!r}.")
    # De-duplicate while keeping chronological order, so '2324,2023/24' is one
    # request rather than two identical downloads.
    return sorted(dict.fromkeys(codes), key=season_start_year)


def parse_leagues(spec):
    """'E0,SP1' -> ['E0', 'SP1'], rejecting codes this contract does not cover."""
    codes = [c.strip().upper() for c in spec.split(",") if c.strip()]
    if not codes:
        raise ValueError(f"No leagues named in {spec!r}.")

    unknown = [c for c in codes if c not in LEAGUES]
    if unknown:
        raise ValueError(
            f"Unknown league code(s) {unknown}. Known codes: {', '.join(sorted(LEAGUES))}. "
            "The 'extra' files for other countries (ARG, BRA, MEX, ...) use a different "
            "column set that football/processor.py does not read."
        )
    return list(dict.fromkeys(codes))


def season_label(code):
    """'2324' -> '2023/24', for printing."""
    start = season_start_year(code)
    return f"{start}/{(start + 1) % 100:02d}"


# ------------------------------------------------------------------- network

def fetch_csv(season, league, session=None, timeout=30, retries=3, backoff=2.0):
    """Fetch one season file's text, retrying transient failures.

    Returns None on 404 — football-data simply has no file for that league and
    season (a league that did not exist yet, or a season not yet started), and
    that is missing data rather than something to retry or die on.

    `requests` is imported here rather than at module scope so the pure parts
    of this module — season specs, the CSV guard, the merge — can be imported
    and tested without it. It is not in requirements-test.txt for that reason.
    """
    import requests  # noqa: PLC0415 — see the docstring

    session = session or requests.Session()
    url = BASE_URL.format(season=season, league=league)

    last_error = None
    for attempt in range(retries):
        try:
            response = session.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code == 404:
                return None
            if response.status_code >= 500:
                raise DownloadError(f"{url} returned {response.status_code}")
            response.raise_for_status()
            return response.text
        except (requests.RequestException, DownloadError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))

    raise DownloadError(f"Could not fetch {url} after {retries} attempts: {last_error}")


# -------------------------------------------------------------- the CSV guard

def read_season_csv(text, label=""):
    """Parse downloaded text as a season file, raising on anything that is not one.

    The check that earns its keep is the first one. A misspelled path on
    football-data can come back as an HTML error page with a 200, and
    `pd.read_csv` will happily turn that into a one-column frame — which is how
    a `.csv` full of `<!DOCTYPE html>` ends up on disk and stays there.
    """
    where = f" for {label}" if label else ""
    head = text.lstrip()[:200].lower()
    if head.startswith("<") or "<html" in head:
        raise DownloadError(
            f"The response{where} is HTML, not CSV. Either the league/season path does not "
            "exist or the site is serving an error page with a 200; open the URL in a browser."
        )

    try:
        df = pd.read_csv(io.StringIO(text), encoding_errors="replace")
    except Exception as exc:  # pandas raises a family of parse errors
        raise DownloadError(f"Could not parse the response{where} as CSV: {exc}") from exc

    # football-data pads its files with trailing blank rows and, in some
    # seasons, trailing unnamed columns. Both are noise, and dropping them here
    # keeps the row count in the summary honest.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:")]]

    if df.empty:
        raise DownloadError(
            f"The file{where} parsed as CSV but has no rows. A season that has not started "
            "yet returns a header-only file; anything else means the format changed."
        )
    return df


def inspect(df, label=""):
    """Route a raw season frame through the contract and report what it resolves to.

    Returns `{n_matches, odds_source, odds_are_closing, odds_coverage}`. Raises
    `MatchFormatError` when the file does not satisfy the contract at all —
    which is the point of doing this at download time rather than at analysis
    time. Validation warnings (a soft market, incomplete coverage) are left to
    `check_match_format`; they describe real data and are not download failures.
    """
    matches = preprocess_matches(df, validate=False)
    source = matches.attrs.get("odds_source")
    return {
        "label": label,
        "n_matches": len(matches),
        "odds_source": source,
        "odds_are_closing": bool(matches.attrs.get("odds_are_closing")),
        "odds_coverage": odds_coverage(matches),
    }


# ----------------------------------------------------------------- the writer

def season_path(season, league, out_dir=DEFAULT_DATA_DIR):
    """Where one league-season lands: `exported_data/football/E0_2324.csv`."""
    return os.path.join(out_dir, f"{league}_{season}.csv")


def write_season(text, path, force=False):
    """Write a downloaded season file, refusing to shrink an existing one.

    An in-progress season legitimately grows on every re-download, so
    overwriting is the normal case. Coming back *smaller* is not: that is a
    truncated transfer or a site-side regression, and quietly replacing a full
    season with half of one is unrecoverable once the original is gone.
    `--force` exists for the case where the site really did drop rows.
    """
    incoming = read_season_csv(text, label=os.path.basename(path))

    if os.path.exists(path) and not force:
        existing = read_season_csv(open(path, encoding="utf-8", errors="replace").read(),
                                   label=f"the existing {path}")
        if len(incoming) < len(existing):
            raise DownloadError(
                f"{path} already holds {len(existing)} rows and the download has "
                f"{len(incoming)}. Refusing to overwrite a longer file with a shorter one — "
                "that is usually a truncated transfer. Pass --force if the source really "
                "did lose rows."
            )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return len(incoming)


# ------------------------------------------------------------------ the driver

def download_seasons(seasons, leagues, out_dir=DEFAULT_DATA_DIR, delay=1.0,
                     dry_run=False, closing_odds_only=False, force=False, session=None):
    """Download every (league, season) pair, validating each file before writing.

    Returns a summary frame with one row per file actually obtained. Missing
    combinations are skipped and listed, for the same reason the lottery
    scraper tolerates a year with no draws: a wide request must not die on the
    first league that did not exist yet. But if *nothing* was obtained, that
    is the parser-is-broken case and it raises.

    `closing_odds_only=True` skips files whose best odds are opening prices
    instead of writing them. That is a narrower rule than it looks: those files
    are perfectly good training data, they just cannot serve as the baseline,
    and mixing them with closing-odds seasons is what `load_seasons` refuses.
    """
    rows, missing, skipped = [], [], []

    first = True
    for league in leagues:
        for season in seasons:
            label = f"{league} {season_label(season)}"
            if not first and delay:
                time.sleep(delay)
            first = False

            print(f"Fetching {label}...", flush=True)
            text = fetch_csv(season, league, session=session)
            if text is None:
                print("  no file published", flush=True)
                missing.append(label)
                continue

            df = read_season_csv(text, label=label)
            report = inspect(df, label=label)

            if closing_odds_only and report["odds_source"] not in CLOSING_SOURCES:
                print(f"  skipped: best odds are {report['odds_source']} (not closing)",
                      flush=True)
                skipped.append(label)
                continue

            path = season_path(season, league, out_dir)
            print(f"  {report['n_matches']} matches, odds {report['odds_source']} "
                  f"({'closing' if report['odds_are_closing'] else 'opening'}), "
                  f"{report['odds_coverage']:.0%} priced", flush=True)

            if not dry_run:
                write_season(text, path, force=force)

            rows.append({
                "league": league,
                "league_name": LEAGUES[league],
                "season": season_label(season),
                "n_matches": report["n_matches"],
                "odds_source": report["odds_source"],
                "odds_are_closing": report["odds_are_closing"],
                "odds_coverage": round(report["odds_coverage"], 4),
                "path": "" if dry_run else path,
            })

    if not rows:
        raise DownloadError(
            f"Nothing was obtained for leagues {list(leagues)} and seasons "
            f"{[season_label(s) for s in seasons]}. Either none of those files exists "
            "(check the codes against https://www.football-data.co.uk/) or the download "
            "path changed."
        )

    if missing:
        print(f"\nNo file published for: {', '.join(missing)} "
              "(other files downloaded fine, so this is missing data, not a broken URL)",
              flush=True)
    if skipped:
        print(f"\nSkipped for having no closing odds: {', '.join(skipped)}. Closing prices "
              f"start at {FIRST_CLOSING_ODDS_SEASON_START}/"
              f"{(FIRST_CLOSING_ODDS_SEASON_START + 1) % 100:02d}.", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", required=True,
                        help="e.g. 2324, 2023/24, 2019/20..2024/25 or 2019-2024")
    parser.add_argument("--leagues", default="E0",
                        help=f"comma-separated codes; known: {', '.join(sorted(LEAGUES))}")
    parser.add_argument("--out-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and validate, write nothing")
    parser.add_argument("--closing-odds-only", action="store_true",
                        help="skip seasons whose best odds are opening prices")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting a file with a shorter one")
    args = parser.parse_args()

    try:
        summary = download_seasons(
            parse_seasons(args.seasons),
            parse_leagues(args.leagues),
            out_dir=args.out_dir,
            delay=args.delay,
            dry_run=args.dry_run,
            closing_odds_only=args.closing_odds_only,
            force=args.force,
        )
    except (DownloadError, MatchFormatError, ValueError) as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print()
    print(summary.drop(columns="league_name").to_string(index=False))

    if summary["odds_source"].nunique(dropna=False) > 1:
        print("\nNote: these files resolve to more than one odds source, so "
              "load_seasons() will refuse to load them together — by design. "
              "Load the ones that share a source, or pass closing_odds_only=True.")
    if args.dry_run:
        print("\nNothing was written (--dry-run).")
