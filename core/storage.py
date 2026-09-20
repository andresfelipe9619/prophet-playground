"""A store that refuses to lose the shape of what was put in it.

Every dataset in this project lives in a gitignored CSV written by a scraper.
That works until it does not: a CSV has no schema, so a column renamed upstream,
a dtype silently widened, or a second copy of the same draw appended by a
re-scrape all produce a file that loads perfectly and means something different.
The contracts in `lottery/utils/processor.py`, `football/processor.py` and
`cycling/processor.py` exist to catch exactly that at *load* time; this module
catches it at *write* time, which is where the evidence of what changed still
exists.

**SQLite rather than Parquet**, and the reason is appending. Parquet is the
better columnar format and it needs `pyarrow`, but appending to it means writing
another file — and a store whose "append" is "another file in the directory" has
reintroduced the multi-file drift that `load_seasons` and `load_races` already
refuse to concatenate through. SQLite is in the standard library, appends in
place, and holds its own metadata in a sibling table, so a store is one file
that can describe itself. No new dependency and nothing to install in CI.

**The schema version is the point.** `write_frame` records the version, the
column names, the dtypes, the row count and a `core/manifest.py` fingerprint;
`read_frame(expected_version=...)` refuses a store written under a different
one rather than returning a frame that is the wrong shape in a way the caller
will discover three transformations later. `append_frame` refuses a frame whose
columns or dtypes differ from what is already stored — the silent corruption
this module is built around, since pandas will happily concatenate a float
column onto an int one and object onto anything.

**Dtypes are restored, not re-inferred.** SQLite has no datetime type, so a `ds`
column round-trips as text unless something puts it back — and an ISO date
string compares *correctly* against another ISO date string, so the bug survives
every obvious test and fails on the first non-ISO input or the first arithmetic.
The recorded dtype map is replayed on read for that reason, and a column that
cannot be restored raises rather than coming back as text.

Domain-free, like the rest of `core/`: it stores a tidy frame under a name the
caller chooses and knows nothing about what is in it.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from core.manifest import data_fingerprint

# Where this module keeps what it knows about each stored table. Prefixed so it
# cannot collide with a caller's table name.
METADATA_TABLE = "_store_schema"

MODES = ("replace", "append")
DUPLICATE_POLICIES = ("error", "skip")


class StorageError(ValueError):
    """The store cannot hold this frame without changing what it means."""


class SchemaVersionError(StorageError):
    """The stored schema version is not the one the caller expects."""


def _connect(path: str) -> sqlite3.Connection:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    return sqlite3.connect(path)


def _ensure_metadata(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {METADATA_TABLE} ("
        "table_name TEXT PRIMARY KEY, schema_version TEXT NOT NULL, columns TEXT NOT NULL, "
        "dtypes TEXT NOT NULL, n_rows INTEGER NOT NULL, fingerprint TEXT NOT NULL, "
        "written_at TEXT NOT NULL)"
    )


def _dtype_map(frame: pd.DataFrame) -> dict[str, str]:
    return {str(name): str(dtype) for name, dtype in frame.dtypes.items()}


def _read_metadata(connection: sqlite3.Connection, table: str) -> dict[str, Any] | None:
    _ensure_metadata(connection)
    row = connection.execute(
        f"SELECT schema_version, columns, dtypes, n_rows, fingerprint, written_at "
        f"FROM {METADATA_TABLE} WHERE table_name = ?", (table,)).fetchone()
    if row is None:
        return None
    return {
        "table": table, "schema_version": row[0], "columns": json.loads(row[1]),
        "dtypes": json.loads(row[2]), "n_rows": int(row[3]), "fingerprint": row[4],
        "written_at": row[5],
    }


def _write_metadata(connection: sqlite3.Connection, table: str, frame: pd.DataFrame,
                    schema_version: str) -> None:
    _ensure_metadata(connection)
    connection.execute(
        f"INSERT INTO {METADATA_TABLE} "
        "(table_name, schema_version, columns, dtypes, n_rows, fingerprint, written_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(table_name) DO UPDATE SET "
        "schema_version = excluded.schema_version, columns = excluded.columns, "
        "dtypes = excluded.dtypes, n_rows = excluded.n_rows, "
        "fingerprint = excluded.fingerprint, written_at = excluded.written_at",
        (table, str(schema_version), json.dumps([str(c) for c in frame.columns]),
         json.dumps(_dtype_map(frame)), int(len(frame)), data_fingerprint(frame),
         datetime.now(UTC).isoformat(timespec="seconds")))


def _check_shape(stored: dict[str, Any], frame: pd.DataFrame) -> None:
    """Refuse an append whose columns or dtypes differ from what is stored.

    This is the whole reason the module exists. pandas will concatenate a float
    column onto an int one and an object column onto either, producing a table
    in which one column means two things — the same failure the domain
    contracts refuse at load time, arriving from the other direction.
    """
    if list(stored["columns"]) != [str(c) for c in frame.columns]:
        raise StorageError(
            f"Stored columns {list(stored['columns'])} do not match the frame's "
            f"{[str(c) for c in frame.columns]}. A renamed or reordered column appended into "
            "an existing table is a column that means two things."
        )
    incoming = _dtype_map(frame)
    drifted = {c: (stored["dtypes"][c], incoming[c])
               for c in incoming if stored["dtypes"].get(c) != incoming[c]}
    if drifted:
        described = ", ".join(f"{c}: {was} -> {now}" for c, (was, now) in sorted(drifted.items()))
        raise StorageError(
            f"Dtype drift on append ({described}). Concatenating these produces a column that "
            "loads fine and means something different; fix the source or write a new version."
        )


def write_frame(frame: pd.DataFrame, path: str, table: str, schema_version: str,
                mode: str = "replace") -> dict[str, Any]:
    """Write a tidy frame, recording the schema it was written under.

    `mode="replace"` overwrites the table; `append_frame` is the guarded path
    and is what a scraper should use. Returns the metadata that was recorded,
    which is the thing worth logging: version, row count and fingerprint.
    """
    if mode not in MODES:
        raise StorageError(f"Unknown mode {mode!r}. Available: {list(MODES)}")
    if frame.empty:
        raise StorageError(
            "Refusing to write an empty frame. A scraper that writes nothing when the markup "
            "changes is the failure mode this project's parsers are shaped against, and a store "
            "that accepts it moves the failure one step further from where it can be seen."
        )

    with _connect(path) as connection:
        if mode == "append":
            stored = _read_metadata(connection, table)
            if stored is not None:
                _check_shape(stored, frame)
        frame.to_sql(table, connection, if_exists=mode, index=False)
        combined = read_frame(path, table, connection=connection) if mode == "append" else frame
        _write_metadata(connection, table, combined, schema_version)
        return _read_metadata(connection, table) or {}


def append_frame(frame: pd.DataFrame, path: str, table: str, schema_version: str,
                 key: Sequence[str] | None = None, on_duplicate: str = "error") -> dict[str, Any]:
    """Append rows, refusing a shape change and deciding duplicates explicitly.

    `key` names the columns that identify a row — a draw date, a (race, kind,
    stage, rider). Re-scraping an overlapping window is the normal case, not an
    error, but *which* of the two copies survives is a decision: `"error"` says
    the caller has not made it, `"skip"` keeps what is already stored and adds
    only what is new. There is no "overwrite" here on purpose, because silently
    replacing a stored row with a re-scraped one is how a corrected result
    becomes an uncorrected one again.
    """
    if on_duplicate not in DUPLICATE_POLICIES:
        raise StorageError(
            f"Unknown duplicate policy {on_duplicate!r}. Available: {list(DUPLICATE_POLICIES)}")

    existing = read_frame(path, table) if table_exists(path, table) else None
    if existing is None:
        return write_frame(frame, path, table, schema_version, mode="replace")

    if key:
        missing = [column for column in key if column not in frame.columns]
        if missing:
            raise StorageError(f"Key column(s) {missing} are not in the frame.")
        stored_keys = set(map(tuple, existing[list(key)].astype(str).to_numpy()))
        incoming_keys = list(map(tuple, frame[list(key)].astype(str).to_numpy()))
        overlap = [k for k in incoming_keys if k in stored_keys]
        if overlap and on_duplicate == "error":
            raise StorageError(
                f"{len(overlap)} row(s) already stored under this key, e.g. {overlap[:3]}. "
                "Re-scraping an overlapping window is normal; which copy wins is a decision, so "
                "pass on_duplicate='skip' to keep what is stored."
            )
        if overlap:
            keep = [k not in stored_keys for k in incoming_keys]
            frame = frame[keep]
            if frame.empty:
                # Nothing new is a perfectly ordinary outcome for a re-scrape and
                # must not look like a failure — but the store is left untouched.
                return _info_or_empty(path, table)

    return write_frame(frame, path, table, schema_version, mode="append")


def _info_or_empty(path: str, table: str) -> dict[str, Any]:
    info = table_info(path, table)
    return info if info is not None else {}


def table_exists(path: str, table: str) -> bool:
    if not os.path.exists(path):
        return False
    with _connect(path) as connection:
        found = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return found is not None


def table_info(path: str, table: str) -> dict[str, Any] | None:
    """What the store says about a table: version, shape, fingerprint, when."""
    if not os.path.exists(path):
        return None
    with _connect(path) as connection:
        return _read_metadata(connection, table)


def tables(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
    return [row[0] for row in rows if row[0] != METADATA_TABLE]


def _restore_dtypes(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Put back the dtypes the frame was written with.

    SQLite has no datetime type, so a date column comes back as text — and an
    ISO date string compares correctly against another ISO date string, which is
    what makes this bug survive casual testing and then fail on arithmetic or on
    the first non-ISO value. A column whose recorded dtype cannot be restored
    raises rather than being handed back as text.
    """
    out = frame.copy()
    for column, dtype in dtypes.items():
        if column not in out.columns:
            continue
        try:
            out[column] = (pd.to_datetime(out[column]) if dtype.startswith("datetime")
                           else out[column].astype(dtype))
        except (TypeError, ValueError) as exc:
            raise StorageError(
                f"Column {column!r} was written as {dtype} and cannot be restored: {exc}. "
                "Handing it back as text would compare and sort in ways that look right until "
                "they do not."
            ) from exc
    return out


def read_frame(path: str, table: str, expected_version: str | None = None,
               connection: sqlite3.Connection | None = None) -> pd.DataFrame:
    """Read a table back with its dtypes, refusing an unexpected schema version.

    `expected_version` is how a caller says which shape its code was written
    against. Left out, anything is read — which is right for a tool that only
    inspects a store and wrong for anything that computes from it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The store {path} does not exist.")

    own = connection is None
    handle = _connect(path) if connection is None else connection
    try:
        stored = _read_metadata(handle, table)
        if stored is None and not table_exists(path, table):
            raise StorageError(f"No table {table!r} in {path}. Stored: {tables(path)}")
        if (expected_version is not None and stored is not None
                and str(stored["schema_version"]) != str(expected_version)):
            raise SchemaVersionError(
                f"{table!r} was written under schema version {stored['schema_version']!r}, and "
                f"this caller expects {expected_version!r}. Reading it anyway would hand back a "
                "frame whose shape is wrong in a way that only shows up several transformations "
                "later."
            )
        frame = pd.read_sql_query(f"SELECT * FROM {table}", handle)  # noqa: S608 — table is checked above
    finally:
        if own:
            handle.close()

    return _restore_dtypes(frame, stored["dtypes"]) if stored else frame
