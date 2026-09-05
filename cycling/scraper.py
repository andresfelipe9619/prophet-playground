"""Scrape road-cycling results from procyclingstats.com into the cycling data contract.

Usage:
    python -m cycling.scraper --race tour-de-france --year 2024 --stages 1-21 --dry-run
    python -m cycling.scraper --race tour-de-france --year 2024 --stages 1-21
    python -m cycling.scraper --race tour-de-france --year 2024 --kind gc --stage 21
    python -m cycling.scraper --race milano-sanremo --year 2024 --kind one_day

Same posture as `lottery/utils/scraper.py`: a parser that silently returns
nothing when the markup changes is worse than one that crashes, so an
unreadable page, an unknown rank marker or a table without the columns a result
needs all raise.

**How it reads a table, and why that way.** It does not select on CSS classes.
It finds the tables on the page, reads their header row, and keeps the one whose
headers include a rank and a rider — then addresses every cell by header
position. Class names on a results site churn constantly and a selector that
misses returns zero rows, which is the failure this module is built to avoid;
a header row that no longer says "Rnk" is a change worth stopping on. The
mapping lives in `HEADER_ALIASES` and is the one place to extend when the site
renames a column.

**The times, which are the subtle part.** A results page publishes the winner's
elapsed time and everyone else's *gap* to it, and marks "same time as the rider
above" with a `,,` placeholder. The contract stores total elapsed seconds, so
this module resolves gaps against the winner's time as it walks down the table.
When the winner's own time is missing there is nothing to anchor to, and every
time on the page is left NaN rather than writing gaps into a column that means
totals — the mistake `cycling/processor.py:time_order_violations` exists to
catch after the fact.

**This has never been run against the live site.** The sandbox this was written
in blocks procyclingstats.com at the network policy, so the parser is written
against the site's documented table layout and tested against saved HTML
fixtures, which is not the same as being validated against reality. Run
`--dry-run` first and compare the first rows against the page in a browser.
"""

import argparse
import csv
import os
import re
import sys
import time

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cycling.common import (
    CSV_COLUMNS,
    DEFAULT_DATA_DIR,
    FINISHED,
    GC,
    ONE_DAY,
    RESULT_KINDS,
    STAGE,
    format_seconds,
    normalise_status,
    parse_time_to_seconds,
)
from cycling.processor import ResultFormatError, check_result_format, preprocess_results

BASE_URL = "https://www.procyclingstats.com"

# The page that carries each kind of result.
PAGE_PATHS = {
    STAGE: "race/{race}/{year}/stage-{stage}",
    GC: "race/{race}/{year}/gc",
    ONE_DAY: "race/{race}/{year}/result",
}

# Header text (letters and digits only, lower-cased) -> the field it holds.
# Extend this when the site renames a column; do not reach for CSS classes.
HEADER_ALIASES = {
    "rnk": "rank", "rank": "rank", "pos": "rank", "place": "rank", "": None,
    "rider": "rider", "ridername": "rider", "name": "rider",
    "team": "team", "teamname": "team",
    "time": "time", "timegap": "time", "gap": "time", "gctime": "time",
    "bib": "bib", "bibnr": "bib", "nr": "bib",
    "uci": "uci", "ucipoints": "uci", "pnt": "points", "points": "points",
    "age": "age", "h2h": None, "specialty": None, "prev": None,
}

# A results table must be able to answer "who, and in what order".
REQUIRED_FIELDS = ("rank", "rider")

# Both spellings of "same time as the rider above" that the site uses.
SAME_TIME_MARKERS = {",,", ",,,", "''", '"'}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en;q=0.9",
}


class ScrapeError(RuntimeError):
    """The page could not be fetched, or did not look like a results page."""


# ------------------------------------------------------------------- the table

def _normalise_header(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _header_fields(row):
    """Map a header row's cells to contract fields by position.

    An unrecognised header is mapped to None rather than raising: results pages
    carry decorative and analytic columns (`H2H`, `Specialty`, a flag) that
    nothing here needs, and failing on them would make the parser brittle in
    the one direction that costs nothing. A *missing* required field is the
    error, and that is checked by the caller.
    """
    fields = []
    for cell in row.find_all(["th", "td"]):
        fields.append(HEADER_ALIASES.get(_normalise_header(cell.get_text(strip=True))))
    return fields


def find_results_table(soup):
    """The results table on the page, plus its header -> field mapping.

    Returns `(table, fields)`. Raises when no table on the page has both a rank
    and a rider column, listing what headers were found — the message is the
    whole point, because that is the situation where somebody has to go and look
    at the page.
    """
    candidates, seen_headers = [], []

    for table in soup.find_all("table"):
        header_row = table.find("thead")
        header_row = header_row.find("tr") if header_row else table.find("tr")
        if header_row is None:
            continue

        fields = _header_fields(header_row)
        seen_headers.append([c.get_text(strip=True) for c in header_row.find_all(["th", "td"])])
        if all(field in fields for field in REQUIRED_FIELDS):
            body = table.find("tbody") or table
            recognised = sum(1 for field in fields if field)
            candidates.append(((recognised, len(body.find_all("tr"))), table, fields))

    if not candidates:
        raise ScrapeError(
            "No table on this page has both a rank and a rider column. Headers found: "
            f"{seen_headers}. Either the page is not a results page, or the site renamed a "
            "column — check HEADER_ALIASES against the page before changing anything else."
        )

    # Ranked by how many columns were recognised, then by length. The full
    # classification carries the whole header row (rank, bib, rider, team, time,
    # points); the sidebars that share its layout carry two or three columns, and
    # they can be *longer* than the result on a page published mid-race — so
    # counting rows alone picks the wrong table exactly when it matters.
    _, table, fields = max(candidates, key=lambda c: c[0])
    return table, fields


def _cell_text(cell):
    return cell.get_text(" ", strip=True)


def _rider_name(cell):
    """The rider's name from a result cell.

    The cell usually holds a link to the rider and, in some layouts, the team
    name immediately after it — taking the cell's whole text would produce
    'POGACAR Tadej UAE Team Emirates'. The link is preferred for that reason,
    with the raw text as the fallback for a layout that has no link.
    """
    link = cell.find("a", href=re.compile(r"rider/"))
    name = _cell_text(link) if link else _cell_text(cell)
    return re.sub(r"\s+", " ", name).strip()


def _team_name(cell, row):
    """The team, from the team column if there is one and the row's team link if not."""
    if cell is not None:
        text = _cell_text(cell)
        if text:
            return text
    link = row.find("a", href=re.compile(r"team/"))
    return _cell_text(link) if link else ""


def parse_results_page(html, race, kind, stage=None, date=None, allow_empty=False):
    """Extract contract rows from one results page. No I/O, so it is testable offline.

    Returns a list of dicts keyed by `CSV_COLUMNS`. `date` is the race or stage
    date in dd/mm/yyyy; when it is None the page's own date is used, and a page
    that does not publish one raises — a result with no date cannot be placed
    in a history, and defaulting it to today would quietly corrupt one.
    """
    if kind not in RESULT_KINDS:
        raise ScrapeError(f"Unknown result kind {kind!r}. Known: {list(RESULT_KINDS)}.")

    soup = BeautifulSoup(html, "html.parser")
    table, fields = find_results_table(soup)

    date = date or parse_race_date(html)
    if date is None:
        raise ScrapeError(
            f"No date found on the page for {race} {kind}"
            f"{f' stage {stage}' if stage is not None else ''}, and none was given. "
            "Pass --date dd/mm/yyyy; a result with no date cannot join a history."
        )

    body = table.find("tbody") or table
    rows, winner_seconds, last_gap = [], None, None

    for tr in body.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells or all(not _cell_text(c) for c in cells):
            continue
        by_field = {field: cells[i] for i, field in enumerate(fields)
                    if field and i < len(cells)}
        if "rank" not in by_field or "rider" not in by_field:
            continue  # a spanning row (a group header, a note), not a result

        rank_text = _cell_text(by_field["rank"])
        try:
            # Raises on a marker this parser does not know. Re-raised as a
            # ScrapeError so a caller can catch one exception type for
            # "the page did not read", whatever part of it failed.
            status = normalise_status(rank_text)
        except ValueError as exc:
            raise ScrapeError(f"In {race} {kind}: {exc}") from exc
        rank = int(re.sub(r"[^0-9]", "", rank_text)) if status is None else None

        rider = _rider_name(by_field["rider"])
        if not rider:
            raise ScrapeError(
                f"A row in {race} {kind} has a rank ({rank_text!r}) but no rider name. "
                "The rider column moved or its markup changed."
            )

        total_seconds = None
        if "time" in by_field and status is None:
            raw_time = _cell_text(by_field["time"])
            try:
                gap = (last_gap if raw_time in SAME_TIME_MARKERS or not raw_time
                       else parse_time_to_seconds(raw_time))
            except ValueError as exc:
                raise ScrapeError(f"In {race} {kind}, rider {rank_text}: {exc}") from exc
            if rank == 1:
                # The first row's time is absolute; everything below is a gap
                # to it. Without it there is nothing to anchor gaps to.
                winner_seconds, last_gap = gap, 0.0
                total_seconds = winner_seconds
            else:
                last_gap = gap
                if winner_seconds is not None and gap is not None:
                    total_seconds = winner_seconds + gap

        rows.append({
            "Date": date,
            "Race": race,
            "Kind": kind,
            "Stage": "" if stage is None else stage,
            "Rank": "" if rank is None else rank,
            "Rider": rider,
            "Team": _team_name(by_field.get("team"), tr),
            "Status": FINISHED if status is None else status,
            "TimeSeconds": "" if total_seconds is None else round(float(total_seconds), 3),
        })

    if not rows and not allow_empty:
        raise ScrapeError(
            f"Parsed 0 riders from the {race} {kind} page"
            f"{f' (stage {stage})' if stage is not None else ''}. Either the result is not "
            "published yet or the table layout changed; re-run with --dry-run and inspect "
            "the HTML."
        )

    ranked = [r for r in rows if r["Rank"] != ""]
    if rows and not ranked:
        raise ScrapeError(
            f"Every row on the {race} {kind} page is a non-finisher, which no published "
            "result looks like. The rank column is probably being read from the wrong place."
        )
    return rows


DATE_LABEL = re.compile(r"^\s*date\s*:?\s*$", re.IGNORECASE)


def parse_race_date(html):
    """The race or stage date from the page's info list, as dd/mm/yyyy, or None.

    procyclingstats writes it as a labelled row ('Date: 29 June 2024'). This
    looks for the label and reads the value next to it, in the markup or in the
    same block of text, and returns None when it cannot find one — the caller
    decides whether that is fatal, because a date given on the command line is
    just as good.
    """
    soup = BeautifulSoup(html, "html.parser")

    for node in soup.find_all(["div", "span", "th", "td", "b", "strong"]):
        if not DATE_LABEL.match(node.get_text(strip=True)):
            continue
        sibling = node.find_next_sibling()
        candidates = [sibling.get_text(" ", strip=True) if sibling else "",
                      node.parent.get_text(" ", strip=True) if node.parent else ""]
        for candidate in candidates:
            parsed = _try_date(re.sub(r"^\s*date\s*:?\s*", "", candidate, flags=re.IGNORECASE))
            if parsed:
                return parsed

    match = re.search(r"date\s*:?\s*([0-9]{1,2}\s+\w+\s+[0-9]{4})", soup.get_text(" ", strip=True),
                      flags=re.IGNORECASE)
    return _try_date(match.group(1)) if match else None


def _try_date(text):
    """'29 June 2024' or '29-06-2024' -> '29/06/2024'. None when it is not a date.

    Day-first, because every European results site writes it that way and the
    ambiguous forms (05/06) would otherwise land five months out.
    """
    text = (text or "").strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%d/%m/%Y")


# ------------------------------------------------------------------- the fetch

def page_url(race, year, kind, stage=None):
    if kind == STAGE and stage is None:
        raise ValueError("A stage result needs a stage number.")
    path = PAGE_PATHS[kind].format(race=race, year=year, stage=stage)
    return f"{BASE_URL}/{path}"


def fetch_page(url, session=None, timeout=30, retries=3, backoff=2.0):
    """Fetch one page, retrying transient failures. None on 404.

    `requests` is imported inside the function so the parser above can be
    imported and tested without it — the same reason `football/downloader.py`
    does it. bs4 is imported at module scope because the parsing needs it.
    """
    import requests  # noqa: PLC0415 — see the docstring

    session = session or requests.Session()
    last_error = None
    for attempt in range(retries):
        try:
            response = session.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code == 404:
                return None
            if response.status_code >= 500:
                raise ScrapeError(f"{url} returned {response.status_code}")
            response.raise_for_status()
            return response.text
        except (requests.RequestException, ScrapeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))

    raise ScrapeError(f"Could not fetch {url} after {retries} attempts: {last_error}")


def scrape_results(race, year, kind=STAGE, stages=(None,), date=None, delay=1.5, session=None):
    """Scrape one race's results of one kind, returning a frame of contract rows.

    A stage with no published page (404) or no rows is skipped rather than
    fatal, so `--stages 1-21` works on a race in progress. The distinction that
    keeps that honest is the lottery scraper's: a **missing** page is tolerated
    because other pages worked, while a **structural** problem — an unknown rank
    marker, a table without a rider column — raises from any page, because that
    means the markup changed and dropping the page would hide it.
    """
    rows, missing, empty = [], [], []

    for i, stage in enumerate(stages):
        if i and delay:
            time.sleep(delay)
        url = page_url(race, year, kind, stage)
        print(f"Fetching {url}...", flush=True)

        html = fetch_page(url, session=session)
        if html is None:
            print("  no page published", flush=True)
            missing.append(stage)
            continue

        page_rows = parse_results_page(html, race=race, kind=kind, stage=stage, date=date,
                                       allow_empty=True)
        print(f"  {len(page_rows)} riders", flush=True)
        if page_rows:
            rows.extend(page_rows)
        else:
            empty.append(stage)

    if not rows:
        raise ScrapeError(
            f"No results parsed for {race} {year} ({kind}). Either nothing is published yet "
            "or the page layout changed — re-run with --dry-run on one page you know exists "
            "and inspect the HTML."
        )

    skipped = [s for s in missing + empty if s is not None]
    if skipped:
        print(f"\nNo result for stage(s): {', '.join(str(s) for s in sorted(skipped))} "
              "(other pages parsed fine, so this is missing data, not a broken parser)",
              flush=True)

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


# ------------------------------------------------------------------ the writer

def default_path(race, year, kind, out_dir=DEFAULT_DATA_DIR):
    """`exported_data/cycling/tour-de-france_2024_stage.csv`.

    The kind is in the filename because one file holds one kind — a stage
    result and a general classification in the same file is the mix
    `cycling/processor.py` refuses to load.
    """
    return os.path.join(out_dir, f"{race}_{year}_{kind}.csv")


def merge_into(new_rows, path):
    """Merge scraped rows into `path`, de-duplicating one rider per result.

    The key is (Race, Kind, Stage, Rider): a rider appears once in one
    classification. Existing rows win, so a hand-corrected file is not silently
    overwritten by a re-scrape — the same rule as the lottery scraper.
    """
    frames = []
    if os.path.exists(path):
        existing = pd.read_csv(path)
        frames.append(existing)
        print(f"{len(existing)} existing rows in {path}")
    frames.append(new_rows)

    combined = pd.concat(frames, ignore_index=True)
    key = ["Race", "Kind", "Stage", "Rider"]
    combined = combined.drop_duplicates(subset=key, keep="first")
    combined["_sort"] = pd.to_datetime(combined["Date"], dayfirst=True)
    combined = (combined.sort_values(["_sort", "Stage", "Rank"], na_position="last")
                        .drop(columns="_sort")
                        .reset_index(drop=True))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    combined.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return combined


def parse_stages(spec):
    """'1-21' or '1,3,5' or '7' -> [1, ..., 21] etc."""
    stages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            stages.extend(range(start, end + 1))
        else:
            stages.append(int(part))
    if not stages:
        raise ValueError(f"No stages named in {spec!r}.")
    return stages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", required=True,
                        help="the race's slug on procyclingstats, e.g. tour-de-france")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--kind", default=STAGE, choices=list(RESULT_KINDS))
    parser.add_argument("--stages", help="stage results to fetch, e.g. 1-21 or 1,3,5")
    parser.add_argument("--stage", type=int,
                        help="a single stage; for --kind gc, the stage the standing follows")
    parser.add_argument("--date", help="dd/mm/yyyy, when the page does not publish one")
    parser.add_argument("--out", help=f"defaults to {DEFAULT_DATA_DIR}/<race>_<year>_<kind>.csv")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    parser.add_argument("--dry-run", action="store_true", help="print results, write nothing")
    args = parser.parse_args()

    if args.kind == STAGE and not (args.stages or args.stage):
        parser.error("--kind stage needs --stages or --stage")

    if args.stages:
        stages = parse_stages(args.stages)
    elif args.stage is not None:
        stages = [args.stage]
    else:
        stages = [None]   # a one-day race, or a GC page that is not stage-specific

    try:
        scraped = scrape_results(args.race, args.year, kind=args.kind, stages=stages,
                                 date=args.date, delay=args.delay)
        # Validating here is the point of doing it at scrape time: a contract
        # break costs one re-run now and a re-derived analysis later.
        results = preprocess_results(scraped, validate=True)
    except (ScrapeError, ResultFormatError, ValueError) as exc:
        print(f"\nScrape failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    report = check_result_format(results)
    print(f"\n{len(results)} rows, {results['race'].nunique()} race(s), "
          f"kind {results.attrs['result_kind']}, "
          f"{int((results['status'] != FINISHED).sum())} non-finisher(s)")
    if report:
        print(f"Note: {report['message']}")

    preview = results.head(10).copy()
    preview["time"] = preview["time_seconds"].map(
        lambda s: "" if pd.isna(s) else format_seconds(s))
    print(preview[["ds", "stage", "rank", "rider", "team", "status", "time"]]
          .to_string(index=False))

    if args.dry_run:
        print("\nNothing was written (--dry-run). Check these rows against the page: the "
              "winner's time should be their real elapsed time and each gap should be added "
              "to it, not stored on its own.")
    else:
        path = args.out or default_path(args.race, args.year, args.kind)
        combined = merge_into(scraped, path)
        print(f"\n{len(combined)} total rows written to {path}")
