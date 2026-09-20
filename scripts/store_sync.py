"""Put a draw export into the store, and say what the store now holds.

    python -m scripts.store_sync import --csv exported_data/final-final.csv
    python -m scripts.store_sync status
    python -m scripts.store_sync check          # exit 1 if the store is stale or broken

This is the half of the pipeline that can be automated today. The scraper still
has to be run by hand, but what it produces goes through the contract and into a
store that records the schema version, the dtypes and a fingerprint — so the
next run can tell whether anything actually arrived, and a shape change is
refused at the point where the evidence of what changed still exists.

**`check` is the one written for a scheduled job.** It exits non-zero when the
store is missing, unreadable, written under another schema version, or has not
grown in longer than `--max-age-days`. A silent source is the failure a schedule
is supposed to catch and the one a cron job that only logs will hide: a scraper
whose markup changed keeps exiting 0 and appending nothing, and every downstream
number goes on being computed from a history that stopped.
"""

import argparse
import os
import sys
from datetime import UTC, datetime

import pandas as pd

from lottery.utils.processor import (
    DRAWS_SCHEMA_VERSION,
    import_csv,
    load_and_preprocess,
    store_status,
)

DEFAULT_STORE = os.path.join("exported_data", "draws.sqlite")
DEFAULT_CSV = os.path.join("exported_data", "final-final.csv")

# Baloto draws on Monday, Wednesday and Saturday, so more than this without a
# new draw means the source stopped rather than the week being quiet.
DEFAULT_MAX_AGE_DAYS = 7


def _describe(info):
    if info is None:
        return "No draw table in this store."
    return (f"{info['n_rows']} draws · schema {info['schema_version']} · "
            f"written {info['written_at']} · fingerprint {info['fingerprint'][:12]}")


def run_import(args):
    before = store_status(args.store)
    info = import_csv(args.csv, args.store, on_duplicate=args.on_duplicate)
    added = info["n_rows"] - (before["n_rows"] if before else 0)
    print(f"Imported {args.csv} -> {args.store}")
    print(f"  {added} new draw(s); {_describe(info)}")
    if added == 0:
        # Not an error: re-importing an unchanged export is the ordinary case.
        # Said out loud anyway, because "it ran fine" and "it added nothing" look
        # identical in a log otherwise.
        print("  Nothing new in this export.")
    return 0


def run_status(args):
    print(f"Store: {args.store}")
    print(f"  {_describe(store_status(args.store))}")
    if os.path.exists(args.store):
        df, _ = load_and_preprocess(args.store)
        print(f"  Draws from {df['ds'].min():%Y-%m-%d} to {df['ds'].max():%Y-%m-%d}")
    return 0


def run_check(args):
    """Exit non-zero on the three ways a scheduled pipeline goes quietly wrong."""
    info = store_status(args.store)
    if info is None:
        print(f"FAIL: no draw table in {args.store}.", file=sys.stderr)
        return 1
    if str(info["schema_version"]) != DRAWS_SCHEMA_VERSION:
        print(f"FAIL: store is schema {info['schema_version']}, this code expects "
              f"{DRAWS_SCHEMA_VERSION}.", file=sys.stderr)
        return 1

    df, _ = load_and_preprocess(args.store)
    latest = df["ds"].max()
    age = (pd.Timestamp(datetime.now(UTC).date()) - pd.Timestamp(latest)).days
    print(f"{_describe(info)}; latest draw {latest:%Y-%m-%d} ({age} day(s) ago)")
    if age > args.max_age_days:
        print(f"FAIL: no draw in {age} days, over the {args.max_age_days}-day limit. A scraper "
              "whose markup changed keeps exiting 0 and appending nothing.", file=sys.stderr)
        return 1
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=DEFAULT_STORE)
    sub = parser.add_subparsers(dest="command", required=True)

    importer = sub.add_parser("import", help="import a draw CSV into the store")
    importer.add_argument("--csv", default=DEFAULT_CSV)
    importer.add_argument("--on-duplicate", default="skip", choices=("skip", "error"))
    importer.set_defaults(handler=run_import)

    status = sub.add_parser("status", help="what the store holds")
    status.set_defaults(handler=run_status)

    check = sub.add_parser("check", help="exit 1 if the store is stale, missing or mis-versioned")
    check.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    check.set_defaults(handler=run_check)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
