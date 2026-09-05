"""Scrape historical Baloto results from loterias.com into the project's data contract.

Usage:
    python -m lottery.utils.scraper --years 2020-2025                 # merge into exported_data/final-final.csv
    python -m lottery.utils.scraper --years 2024 --dry-run            # print what was parsed, write nothing

The parser is deliberately loud. A scraper that silently writes an empty file
when the site's markup changes is worse than one that crashes: you only find
out months later, when the analysis is already running on stale data. So a
page that yields no rows, or a draw whose ball count doesn't match the game,
raises instead of being skipped.

`--dry-run` prints the parsed rows without writing anything — use it to eyeball
the first few draws against the website before trusting a scrape.
"""

import argparse
import csv
import os
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottery.models.common import DEFAULT_DATA_PATH, MAIN_BALLS_DRAWN

BASE_URL = "https://www.loterias.com/baloto/resultados/{year}"
BALLS_PER_DRAW = MAIN_BALLS_DRAWN + 1  # 5 main + superbalota, in that order

# Keyed by the first three letters so "may"/"mayo" and "sep"/"set"/"septiembre"
# all resolve; an unknown month raises rather than silently becoming "00".
MONTHS = {
    "ene": "01", "feb": "02", "mar": "03", "abr": "04",
    "may": "05", "jun": "06", "jul": "07", "ago": "08",
    "sep": "09", "oct": "10", "nov": "11", "dic": "12",
}

DATE_PATTERN = re.compile(r"(\d{1,2})\s+([a-zA-ZáéíóúÁÉÍÓÚ]+)\.?\s+(\d{4})")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "es-CO,es;q=0.9",
}


class ScrapeError(RuntimeError):
    """The page could not be fetched, or did not look like a results page."""


def parse_spanish_date(text):
    """'12 oct 2024' -> '12/10/2024' (the dd/mm/yyyy the data contract expects)."""
    match = DATE_PATTERN.search(text)
    if not match:
        raise ScrapeError(f"Could not read a date from {text!r}")

    day, month_name, year = match.groups()
    month = MONTHS.get(month_name[:3].lower())
    if month is None:
        raise ScrapeError(f"Unknown month {month_name!r} in date {text!r}")
    return f"{day.zfill(2)}/{month}/{year}"


def parse_results_page(html, year=None, allow_empty=False):
    """Extract [{Date, Ball, Revancha}] from one year's results page.

    `Ball` is the dash-separated draw in the order the site lists it, which the
    rest of the project reads as 5 main balls followed by the superbalota.
    Kept free of any network access so it can be tested against saved HTML.

    A page with no rows raises unless `allow_empty` — only scrape_years sets
    that, because it can tell "this year has no results" from "the parser is
    broken" by whether *other* years worked.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for tr in soup.find_all("tr"):
        date_cell = tr.find("td", class_="centred")
        numbers_cell = tr.find("td", class_="baloto")
        if not date_cell or not numbers_cell:
            continue

        date_link = date_cell.find("a")
        date = parse_spanish_date(date_link.get_text() if date_link else date_cell.get_text())

        ball_lists = numbers_cell.find_all("ul", class_="balls")
        if not ball_lists:
            raise ScrapeError(f"No <ul class='balls'> found for the draw on {date}")

        draws = [
            [li.get_text(strip=True) for li in ul.find_all("li", class_="ball")]
            for ul in ball_lists
        ]
        main_draw = draws[0]
        if len(main_draw) != BALLS_PER_DRAW:
            raise ScrapeError(
                f"Draw on {date} has {len(main_draw)} balls ({main_draw}), expected "
                f"{BALLS_PER_DRAW} ({MAIN_BALLS_DRAWN} main + superbalota). The site's markup "
                "likely changed — check parse_results_page against the page before trusting it."
            )

        rows.append({
            "Date": date,
            "Ball": "-".join(main_draw),
            "Revancha": "-".join(draws[1]) if len(draws) > 1 else "",
        })

    if not rows and not allow_empty:
        raise ScrapeError(
            f"Parsed 0 draws{f' for {year}' if year else ''}. Either the year has no results "
            "or the page structure changed; re-run with --dry-run and inspect the HTML."
        )
    return rows


def fetch_year(year, session=None, timeout=30, retries=3, backoff=2.0):
    """Fetch one year's results page, retrying transient network/5xx failures.

    Returns None for a 404 — that is "no page for this year", not a failure to
    retry, and a multi-year scrape should move on rather than burn three
    attempts and abort.
    """
    session = session or requests.Session()
    url = BASE_URL.format(year=year)

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


def scrape_years(years, delay=1.5, session=None):
    """Scrape several years, pausing between requests to stay a polite client.

    Years with no published results are skipped rather than fatal, so a wide
    range ("give me everything") does not die on the first year that predates
    the game or the site's archive. The distinction that keeps this honest:

    - A year with **no rows** is only tolerated because other years worked. If
      *no* year yields anything, the parser is broken and this raises.
    - A **structural** problem — wrong ball count, unknown month, unreadable
      date — still raises immediately, from any year. That means the markup
      changed, and silently dropping the year would hide it.
    """
    session = session or requests.Session()
    rows, empty_years, missing_years = [], [], []

    for i, year in enumerate(years):
        if i:
            time.sleep(delay)
        print(f"Fetching {year}...", flush=True)

        html = fetch_year(year, session=session)
        if html is None:
            print("  no page for that year", flush=True)
            missing_years.append(year)
            continue

        year_rows = parse_results_page(html, year=year, allow_empty=True)
        print(f"  {len(year_rows)} draws", flush=True)
        if year_rows:
            rows.extend(year_rows)
        else:
            empty_years.append(year)

    if not rows:
        raise ScrapeError(
            f"No draws parsed from any of {list(years)}. Either none of those years has published "
            "results, or the page structure changed — re-run with --dry-run on a single year you "
            "know has results and inspect the HTML."
        )

    skipped = empty_years + missing_years
    if skipped:
        print(f"\nNo results for: {', '.join(str(y) for y in sorted(skipped))} "
              "(other years parsed fine, so this is missing data, not a broken parser)", flush=True)

    return pd.DataFrame(rows)


def merge_into(new_draws, path):
    """Merge scraped draws into `path`, de-duplicating by date and sorting chronologically.

    Existing rows win on conflict: a hand-corrected local file is not silently
    overwritten by a re-scrape.
    """
    frames = []
    if os.path.exists(path):
        existing = pd.read_csv(path)
        frames.append(existing)
        print(f"{len(existing)} existing draws in {path}")
    frames.append(new_draws)

    combined = pd.concat(frames, ignore_index=True)
    combined["_sort"] = pd.to_datetime(combined["Date"], dayfirst=True)
    combined = (
        combined.drop_duplicates(subset="Date", keep="first")
        .sort_values("_sort")
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    combined.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return combined


def parse_years(spec):
    """'2020-2023' or '2021,2024' or '2024' -> [2020, 2021, 2022, 2023] etc."""
    years = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    return years


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", required=True, help="e.g. 2024, 2020-2025, or 2021,2024")
    parser.add_argument("--out", default=DEFAULT_DATA_PATH)
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    parser.add_argument("--dry-run", action="store_true", help="print results, write nothing")
    args = parser.parse_args()

    try:
        draws = scrape_years(parse_years(args.years), delay=args.delay)
    except ScrapeError as exc:
        print(f"\nScrape failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run:
        print(f"\n{len(draws)} draws parsed (nothing written):")
        print(draws.head(10).to_string(index=False))
        print("\nCheck these against the website before running without --dry-run.")
    else:
        combined = merge_into(draws, args.out)
        print(f"\n{len(combined)} total draws written to {args.out}")
