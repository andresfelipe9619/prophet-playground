"""Background jobs: a queue that outlives the browser tab that asked for the work.

Every heavy evaluation in the dashboard used to run inside a button click. The
page blocked behind a spinner, the result lived in `st.session_state`, and a
browser refresh threw it away. That shape also turned a slow run into a
dashboard that looked hung and then disconnected, which is what the TimesFM
episode in CLAUDE.md turned out to be sitting on.

This module moves the work out of the request. A caller **submits** a job (a
kind, its parameters and its input frames); a separate **worker** process claims
it, runs it and stores the result; any later page load with the same kind,
parameters and data reads the stored result instead of computing it again.

Four things are load-bearing, and each is a way a queue goes quietly wrong:

**A job is identified by what it would compute, not by who asked.** The key is
a hash of the kind, the parameters and the *fingerprints* of the input frames
(`core/manifest.py`), so the same evaluation on the same data is one job however
many times the button is pressed, and a single corrected cell in the data is a
different job. A row count would not be; it is the stand-in `data_fingerprint`
was written to replace.

**The inputs travel with the job.** The worker never re-reads "the current
data": the frames are stored with the job at submission, so the result
describes the data the reader was looking at when they asked, even if a scraper
has appended a draw since. A queue that re-read its inputs at run time would
attach today's result to yesterday's question.

**A result that cannot be stored exactly is refused.** Results are frames, and
JSON loses dtypes: an object column of `None` comes back as `float64`, which is
the silent shape change `core/storage.py` refuses on append. Every result is
decoded again before it is accepted and compared with what was produced, and a
mismatch fails the job rather than storing a frame that means something else.
JSON rather than pickle because a pickle is code on load, and a results file is
something people open.

**A dead worker does not leave a job "running" forever.** A worker beats a
heartbeat from a background thread while a job runs. `requeue_stale` returns a
job whose heartbeat has stopped to the queue, and after `max_attempts` fails it
instead — a job that kills its worker (out of memory, say) would otherwise be
retried by every worker that picks it up, forever.

Domain-free, like the rest of `core/`: a handler is a function the caller
supplies, `handler(params, inputs, progress) -> dict`, and this module knows
nothing about draws, matches or races.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from core.manifest import data_fingerprint, run_manifest

SCHEMA_VERSION = "1"

# One default for the dashboard, the worker and the nightly run, so all three
# read and write the same queue unless told otherwise. Under exported_data/,
# which is gitignored: this file is a cache of results, not a record — the
# registries and the ledger are the records, and they are committed.
DEFAULT_PATH = "exported_data/jobs.sqlite"

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
STATUSES = (QUEUED, RUNNING, DONE, FAILED)
ACTIVE = (QUEUED, RUNNING)

# A running job whose heartbeat is older than this is presumed orphaned. The
# heartbeat thread beats every HEARTBEAT_SECONDS, so this is several missed
# beats, not one slow one.
HEARTBEAT_SECONDS = 5.0
STALE_AFTER_SECONDS = 60.0
MAX_ATTEMPTS = 3

# A worker counts as alive when it has beaten within this long. Longer than the
# idle poll so a worker between polls is not reported dead.
WORKER_ALIVE_SECONDS = 30.0

Progress = Callable[[str], None]
Handler = Callable[[Mapping[str, Any], Mapping[str, pd.DataFrame], Progress], Mapping[str, Any]]


class JobError(RuntimeError):
    """The queue cannot do what was asked without losing something."""


class JobResultError(JobError):
    """A handler's result would not come back out of the store as it went in."""


# ------------------------------------------------------------------ storage


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _age_seconds(stamp: str | None) -> float:
    if not stamp:
        return float("inf")
    return (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds()


def _connect(path: str) -> sqlite3.Connection:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # isolation_level=None: transactions are opened explicitly with BEGIN
    # IMMEDIATE where two workers could race, and are otherwise autocommit.
    # The timeout covers the other writer holding the lock for a moment.
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS _jobs_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            kind TEXT NOT NULL,
            params TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            heartbeat_at TEXT,
            worker TEXT,
            progress TEXT,
            error TEXT,
            inputs_manifest TEXT NOT NULL,
            worker_manifest TEXT
        );
        CREATE INDEX IF NOT EXISTS jobs_key ON jobs (key, status);
        CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, id);
        CREATE TABLE IF NOT EXISTS job_inputs (
            job_id INTEGER NOT NULL, name TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY (job_id, name));
        CREATE TABLE IF NOT EXISTS job_results (job_id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS workers (
            worker TEXT PRIMARY KEY, pid INTEGER NOT NULL, host TEXT NOT NULL,
            kinds TEXT NOT NULL, started_at TEXT NOT NULL, heartbeat_at TEXT NOT NULL,
            manifest TEXT NOT NULL);
        """
    )
    row = connection.execute("SELECT value FROM _jobs_meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        connection.execute("INSERT INTO _jobs_meta (key, value) VALUES ('schema_version', ?)",
                           (SCHEMA_VERSION,))
    elif row["value"] != SCHEMA_VERSION:
        raise JobError(
            f"The job store is schema {row['value']}; this code expects {SCHEMA_VERSION}. "
            "Delete the file to start a fresh queue — it holds cached results, not records.")


# ------------------------------------------------------- encoding results


def _encode_value(value: Any) -> Any:
    """JSON-shaped copy of a non-frame value, tagging what JSON cannot say."""
    if isinstance(value, pd.Timestamp):
        return {"__timestamp__": value.isoformat()}
    if isinstance(value, datetime):
        return {"__timestamp__": pd.Timestamp(value).isoformat()}
    if isinstance(value, np.ndarray):
        # The dtype travels with it: a float matrix returned as nested lists
        # would come back as an object array, or as a list, and a caller doing
        # arithmetic on it finds out one indexing error later.
        return {"__ndarray__": _encode_value(value.tolist()), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return {"__nan__": True}
    if isinstance(value, Mapping):
        return {str(k): _encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise JobResultError(
        f"Cannot store a {type(value).__name__} in a job result. Return frames, numbers, "
        "strings, dates or lists and dicts of them.")


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__timestamp__"}:
            return pd.Timestamp(value["__timestamp__"])
        if set(value) == {"__nan__"}:
            return float("nan")
        if set(value) == {"__ndarray__", "dtype"}:
            return np.asarray(_decode_value(value["__ndarray__"]), dtype=value["dtype"])
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


# Frames are encoded column by column rather than through `DataFrame.to_json`,
# for two measured reasons. `to_json` rounds floats to at most 15 significant
# digits (10 by default), so a stored match frame was not the frame the page
# held and a worker fitted on odds that differed in the eleventh digit — which
# the first version of this module let through because its guard compared with
# assert_frame_equal's default tolerance. And its table schema cannot say
# "object column holding only None", so read_json turned one into float64.
# Python's own `json` writes each float with the shortest repr that reads back
# bit-identical, and the dtype recorded beside every column is replayed on read.


def _encode_cells(values: Iterable[Any]) -> list[Any]:
    return [None if (v is pd.NA or v is pd.NaT) else _encode_value(v) for v in values]


def _encode_column(series: pd.Series) -> dict[str, Any]:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return {"dtype": "category", "categories": _encode_value(list(dtype.categories)),
                "ordered": bool(dtype.ordered), "codes": series.cat.codes.tolist()}
    if pd.api.types.is_datetime64_any_dtype(dtype):
        # isoformat is exact to the nanosecond and carries the zone if any.
        return {"dtype": str(dtype),
                "values": [None if pd.isna(v) else v.isoformat() for v in series]}
    return {"dtype": str(dtype), "values": _encode_cells(series.tolist())}


def _decode_column(payload: Mapping[str, Any], index: pd.Index) -> pd.Series:
    dtype = payload["dtype"]
    if dtype == "category":
        categories = _decode_value(payload["categories"])
        return pd.Series(pd.Categorical.from_codes(payload["codes"], categories=categories,
                                                   ordered=payload["ordered"]), index=index)
    if dtype.startswith("datetime64"):
        return pd.Series(pd.to_datetime(payload["values"]), index=index).astype(dtype)
    return pd.Series([_decode_value(v) for v in payload["values"]], index=index, dtype=dtype)


def _encode_index(index: pd.Index) -> dict[str, Any]:
    name = _encode_value(index.name)
    if isinstance(index, pd.RangeIndex):
        return {"range": [index.start, index.stop, index.step], "name": name}
    return {"column": _encode_column(pd.Series(index)), "length": len(index), "name": name}


def _decode_index(payload: Mapping[str, Any]) -> pd.Index:
    name = _decode_value(payload["name"])
    if "range" in payload:
        return pd.RangeIndex(*payload["range"], name=name)
    values = _decode_column(payload["column"], pd.RangeIndex(payload["length"]))
    return pd.Index(values, name=name)


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if not all(isinstance(c, str) for c in frame.columns):
        raise JobResultError(
            f"Frame columns must be strings to store; got {list(frame.columns)!r}. Rename them "
            "before returning — a positional column that comes back as '0' is a different column.")
    if frame.columns.duplicated().any():
        raise JobResultError(f"Frame has duplicate column names: {list(frame.columns)!r}.")
    return {
        "__frame__": {"columns": list(frame.columns), "index": _encode_index(frame.index),
                      "data": {c: _encode_column(frame[c]) for c in frame.columns}},
        "attrs": _encode_value(dict(frame.attrs)),
    }


def _decode_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    body = payload["__frame__"]
    index = _decode_index(body["index"])
    frame = pd.DataFrame({c: _decode_column(body["data"][c], index) for c in body["columns"]},
                         index=index, columns=body["columns"])
    frame.attrs = _decode_value(payload["attrs"])
    return frame


def encode_result(result: Mapping[str, Any]) -> str:
    """Serialise a handler's result: a mapping of names to frames or JSON-shaped values."""
    if not isinstance(result, Mapping):
        raise JobResultError(f"A handler must return a mapping; got {type(result).__name__}.")
    encoded = {}
    for name, value in result.items():
        encoded[str(name)] = (_encode_frame(value) if isinstance(value, pd.DataFrame)
                              else {"__value__": _encode_value(value)})
    return json.dumps(encoded)


def decode_result(text: str) -> dict[str, Any]:
    decoded = {}
    for name, payload in json.loads(text).items():
        decoded[name] = (_decode_frame(payload) if "__frame__" in payload
                         else _decode_value(payload["__value__"]))
    return decoded


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, pd.DataFrame):
        if not isinstance(b, pd.DataFrame):
            return False
        try:
            # check_exact: the default tolerance (rtol 1e-5) is how a store that
            # rounded every float to ten digits passed this check the first time.
            pd.testing.assert_frame_equal(a, b, check_dtype=True, check_exact=True)
        except AssertionError:
            return False
        # assert_frame_equal compares object cells with ==, and ("tour", 1) ==
        # ["tour", 1] is a comparison numpy answers True. A tuple that comes back
        # as a list still misses every lookup keyed on the tuple, so the element
        # types of object columns are compared as well.
        for column in a.columns:
            if a[column].dtype == object:
                left = [type(v) for v in a[column] if not _missing(v)]
                right = [type(v) for v in b[column] if not _missing(v)]
                if left != right:
                    return False
        return _same(dict(a.attrs), dict(b.attrs))
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")))
    if isinstance(a, Mapping):
        return isinstance(b, Mapping) and set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    if isinstance(a, (datetime, pd.Timestamp)):
        return bool(pd.Timestamp(a) == pd.Timestamp(b))
    return bool(a == b)


def checked_encode(result: Mapping[str, Any]) -> str:
    """Encode, decode again, and refuse anything that did not survive the trip."""
    text = encode_result(result)
    back = decode_result(text)
    for name, value in result.items():
        if not _same(value, back[str(name)]):
            raise JobResultError(
                f"Result {name!r} does not come back out of the job store as it went in. "
                "Storing it anyway would hand the next reader a different frame under the same "
                "name — fix the result's dtypes (or its column names) instead.")
    return text


# --------------------------------------------------------------- the queue


def job_key(kind: str, params: Mapping[str, Any], inputs: Mapping[str, pd.DataFrame] | None = None) -> str:
    """What a job would compute: kind, parameters and the fingerprint of each input.

    Parameters are canonicalised (sorted keys, tagged dates) so two dicts that
    mean the same thing hash the same, and inputs are named so the same frame
    passed as a different input is a different job.
    """
    digest = hashlib.sha256()
    digest.update(kind.encode())
    digest.update(json.dumps(_encode_value(dict(params)), sort_keys=True).encode())
    for name in sorted(inputs or {}):
        digest.update(name.encode())
        digest.update(data_fingerprint((inputs or {})[name]).encode())
    return digest.hexdigest()


def _row_to_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    job = dict(row)
    job["params"] = _decode_value(json.loads(job["params"]))
    job["inputs_manifest"] = json.loads(job["inputs_manifest"])
    job["worker_manifest"] = json.loads(job["worker_manifest"]) if job["worker_manifest"] else None
    return job


def submit(path: str, kind: str, params: Mapping[str, Any],
           inputs: Mapping[str, pd.DataFrame] | None = None, label: str = "",
           force: bool = False) -> dict[str, Any]:
    """Queue a job, or return the one that already answers the same question.

    An identical job that is queued or running is returned rather than
    duplicated, whatever `force` says — two workers computing the same thing is
    never what anyone wanted. A finished one is returned unless `force`, which
    is how a reader asks for a fresh run of something already answered. A
    failed one never blocks a resubmission.
    """
    inputs = dict(inputs or {})
    # Inputs get the same round-trip check as results, and before anything is
    # written: a worker that computed on a frame the store had quietly changed
    # would attach a correct answer to a question nobody asked.
    encoded_inputs = {}
    for name, frame in inputs.items():
        payload = _encode_frame(frame)
        if not _same(frame, _decode_frame(json.loads(json.dumps(payload)))):
            raise JobResultError(
                f"Input {name!r} does not come back out of the job store as it went in, so a "
                "worker would compute on different data. Fix its dtypes before submitting.")
        encoded_inputs[name] = json.dumps(payload)
    key = job_key(kind, params, inputs)

    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        statuses = ACTIVE if force else (*ACTIVE, DONE)
        placeholders = ",".join("?" * len(statuses))
        existing = connection.execute(
            f"SELECT * FROM jobs WHERE key = ? AND status IN ({placeholders}) "
            "ORDER BY id DESC LIMIT 1", (key, *statuses)).fetchone()
        if existing is not None:
            connection.execute("COMMIT")
            return _row_to_job(existing)  # type: ignore[return-value]

        manifest = {"key": key, "inputs": {name: {"fingerprint": data_fingerprint(frame),
                                                  "n_rows": int(len(frame))}
                                           for name, frame in inputs.items()}}
        cursor = connection.execute(
            "INSERT INTO jobs (key, kind, params, label, status, created_at, inputs_manifest) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, kind, json.dumps(_encode_value(dict(params)), sort_keys=True), label, QUEUED,
             _now(), json.dumps(manifest)))
        job_id = cursor.lastrowid
        for name, payload in encoded_inputs.items():
            connection.execute("INSERT INTO job_inputs (job_id, name, payload) VALUES (?, ?, ?)",
                               (job_id, name, payload))
        connection.execute("COMMIT")
        return get(path, int(job_id))  # type: ignore[arg-type,return-value]
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def get(path: str, job_id: int) -> dict[str, Any] | None:
    connection = _connect(path)
    try:
        return _row_to_job(connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())
    finally:
        connection.close()


def find(path: str, key: str) -> dict[str, Any] | None:
    """The most relevant job for a key: an active one if any, else the latest done or failed.

    Active first because a reader looking at a question that is being answered
    right now wants the progress, not the last answer. The store may not exist
    yet, which is simply "no job".
    """
    if not os.path.exists(path):
        return None
    connection = _connect(path)
    try:
        for statuses in (ACTIVE, (DONE,), (FAILED,)):
            placeholders = ",".join("?" * len(statuses))
            row = connection.execute(
                f"SELECT * FROM jobs WHERE key = ? AND status IN ({placeholders}) "
                "ORDER BY id DESC LIMIT 1", (key, *statuses)).fetchone()
            if row is not None:
                return _row_to_job(row)
        return None
    finally:
        connection.close()


def list_jobs(path: str, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    connection = _connect(path)
    try:
        if status is None:
            rows = connection.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = connection.execute("SELECT * FROM jobs WHERE status = ? ORDER BY id DESC LIMIT ?",
                                      (status, limit))
        return [_row_to_job(row) for row in rows]  # type: ignore[misc]
    finally:
        connection.close()


def counts(path: str) -> dict[str, int]:
    """How many jobs sit in each status."""
    out = dict.fromkeys(STATUSES, 0)
    if not os.path.exists(path):
        return out
    connection = _connect(path)
    try:
        for row in connection.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"):
            out[row["status"]] = int(row["n"])
        return out
    finally:
        connection.close()


def load_inputs(path: str, job_id: int) -> dict[str, pd.DataFrame]:
    connection = _connect(path)
    try:
        rows = connection.execute("SELECT name, payload FROM job_inputs WHERE job_id = ?", (job_id,))
        return {row["name"]: _decode_frame(json.loads(row["payload"])) for row in rows}
    finally:
        connection.close()


def load_result(path: str, job_id: int) -> dict[str, Any] | None:
    connection = _connect(path)
    try:
        row = connection.execute("SELECT payload FROM job_results WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else decode_result(row["payload"])
    finally:
        connection.close()


def claim(path: str, worker: str, kinds: Iterable[str] | None = None,
          worker_manifest: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Atomically take the oldest queued job this worker can run, or None.

    `kinds` restricts the claim to jobs the worker has a handler for, so a
    worker started without one domain's dependencies leaves that domain's jobs
    queued for a worker that can run them rather than failing them.
    """
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        query, args = "SELECT id FROM jobs WHERE status = ?", [QUEUED]
        if kinds is not None:
            kinds = list(kinds)
            if not kinds:
                connection.execute("COMMIT")
                return None
            query += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        row = connection.execute(query + " ORDER BY id LIMIT 1", args).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        now = _now()
        connection.execute(
            "UPDATE jobs SET status = ?, started_at = ?, heartbeat_at = ?, worker = ?, "
            "attempts = attempts + 1, progress = NULL, error = NULL, worker_manifest = ? "
            "WHERE id = ?",
            (RUNNING, now, now, worker,
             json.dumps(_encode_value(dict(worker_manifest))) if worker_manifest else None,
             row["id"]))
        connection.execute("COMMIT")
        job_id = int(row["id"])
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return get(path, job_id)


def heartbeat(path: str, job_id: int, worker: str, progress: str | None = None) -> None:
    """Record that `worker` is still on this job. A job it no longer owns is left alone."""
    connection = _connect(path)
    try:
        if progress is None:
            connection.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND worker = ? AND status = ?",
                               (_now(), job_id, worker, RUNNING))
        else:
            connection.execute(
                "UPDATE jobs SET heartbeat_at = ?, progress = ? WHERE id = ? AND worker = ? AND status = ?",
                (_now(), progress, job_id, worker, RUNNING))
    finally:
        connection.close()


def finish(path: str, job_id: int, worker: str, result: Mapping[str, Any]) -> None:
    """Store a result and mark the job done — only if this worker still owns it.

    A worker whose job was requeued as stale (a long pause, a suspended laptop)
    and then finishes anyway must not overwrite whatever the job has become
    since; the second owner's answer is the one on record.
    """
    payload = checked_encode(result)
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        owned = connection.execute("SELECT 1 FROM jobs WHERE id = ? AND worker = ? AND status = ?",
                                   (job_id, worker, RUNNING)).fetchone()
        if owned is None:
            connection.execute("ROLLBACK")
            return
        connection.execute("INSERT OR REPLACE INTO job_results (job_id, payload) VALUES (?, ?)",
                           (job_id, payload))
        connection.execute("UPDATE jobs SET status = ?, finished_at = ?, progress = NULL WHERE id = ?",
                           (DONE, _now(), job_id))
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def fail(path: str, job_id: int, worker: str | None, error: str) -> None:
    connection = _connect(path)
    try:
        if worker is None:
            connection.execute("UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
                               (FAILED, _now(), error, job_id))
        else:
            connection.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE id = ? AND worker = ?",
                (FAILED, _now(), error, job_id, worker))
    finally:
        connection.close()


def release(path: str, job_id: int, worker: str) -> None:
    """Put a job this worker owns back in the queue, as if it had never been claimed.

    For a worker stopped deliberately mid-job (Ctrl-C). The attempt it used is
    given back, since stopping a worker is not evidence against the job.
    """
    connection = _connect(path)
    try:
        connection.execute(
            "UPDATE jobs SET status = ?, worker = NULL, started_at = NULL, heartbeat_at = NULL, "
            "progress = NULL, attempts = MAX(attempts - 1, 0) WHERE id = ? AND worker = ? AND status = ?",
            (QUEUED, job_id, worker, RUNNING))
    finally:
        connection.close()


def requeue_stale(path: str, stale_after: float = STALE_AFTER_SECONDS,
                  max_attempts: int = MAX_ATTEMPTS) -> list[int]:
    """Return orphaned running jobs to the queue, or fail them after `max_attempts`.

    Returns the ids touched. A job is orphaned when its heartbeat is older than
    `stale_after` — its worker was killed, the machine slept, or the process
    died without a chance to say so.
    """
    if not os.path.exists(path):
        return []
    connection = _connect(path)
    touched = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute("SELECT id, attempts, heartbeat_at, worker FROM jobs WHERE status = ?",
                                  (RUNNING,)).fetchall()
        for row in rows:
            if _age_seconds(row["heartbeat_at"]) <= stale_after:
                continue
            touched.append(int(row["id"]))
            if row["attempts"] >= max_attempts:
                connection.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
                    (FAILED, _now(),
                     f"Its worker stopped responding {row['attempts']} time(s). A job that takes its "
                     "worker down with it (out of memory, say) is not retried again.", row["id"]))
            else:
                connection.execute(
                    "UPDATE jobs SET status = ?, worker = NULL, heartbeat_at = NULL, progress = ? "
                    "WHERE id = ?",
                    (QUEUED, f"Requeued: worker {row['worker']} stopped responding.", row["id"]))
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return touched


# ----------------------------------------------------------------- workers


def new_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"


def register_worker(path: str, worker: str, kinds: Iterable[str],
                    manifest: Mapping[str, Any] | None = None) -> None:
    now = _now()
    connection = _connect(path)
    try:
        connection.execute(
            "INSERT OR REPLACE INTO workers (worker, pid, host, kinds, started_at, heartbeat_at, manifest) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (worker, os.getpid(), socket.gethostname(), json.dumps(sorted(kinds)), now, now,
             json.dumps(_encode_value(dict(manifest or {})))))
    finally:
        connection.close()


def beat_worker(path: str, worker: str) -> None:
    connection = _connect(path)
    try:
        connection.execute("UPDATE workers SET heartbeat_at = ? WHERE worker = ?", (_now(), worker))
    finally:
        connection.close()


def unregister_worker(path: str, worker: str) -> None:
    connection = _connect(path)
    try:
        connection.execute("DELETE FROM workers WHERE worker = ?", (worker,))
    finally:
        connection.close()


def live_workers(path: str, within: float = WORKER_ALIVE_SECONDS) -> list[dict[str, Any]]:
    """Workers that have beaten within `within` seconds, newest first."""
    if not os.path.exists(path):
        return []
    connection = _connect(path)
    try:
        rows = connection.execute("SELECT * FROM workers ORDER BY heartbeat_at DESC").fetchall()
    finally:
        connection.close()
    alive = []
    for row in rows:
        if _age_seconds(row["heartbeat_at"]) <= within:
            worker = dict(row)
            worker["kinds"] = json.loads(worker["kinds"])
            worker["manifest"] = _decode_value(json.loads(worker["manifest"]))
            alive.append(worker)
    return alive


class _Heartbeat(threading.Thread):
    """Beats for a job (and its worker) until stopped, from outside the handler.

    A handler is a long model fit with no natural place to report from; without
    a thread, a job that is working hard is indistinguishable from one whose
    worker died, and `requeue_stale` would hand it to a second worker.
    """

    def __init__(self, path: str, job_id: int, worker: str, every: float) -> None:
        super().__init__(daemon=True)
        self.path, self.job_id, self.worker, self.every = path, job_id, worker, every
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.wait(self.every):
            try:
                heartbeat(self.path, self.job_id, self.worker)
                beat_worker(self.path, self.worker)
            except sqlite3.Error:
                # A locked or briefly unavailable store is retried on the next
                # beat; one missed beat is well inside STALE_AFTER_SECONDS.
                continue


def run_job(path: str, job: Mapping[str, Any], handlers: Mapping[str, Handler], worker: str,
            heartbeat_every: float = HEARTBEAT_SECONDS) -> str:
    """Run one claimed job to DONE or FAILED. Returns the final status.

    Any exception from the handler — or from storing its result — fails the job
    with the traceback, because a job that fails silently is re-run by the next
    reader who presses the button and fails again, which is the loop a spinner
    hides. KeyboardInterrupt releases the job back to the queue and re-raises.
    """
    job_id = int(job["id"])
    handler = handlers.get(job["kind"])
    if handler is None:
        fail(path, job_id, worker, f"No handler for job kind {job['kind']!r} in this worker.")
        return FAILED

    def progress(message: str) -> None:
        heartbeat(path, job_id, worker, progress=str(message))

    beat = _Heartbeat(path, job_id, worker, heartbeat_every)
    beat.start()
    try:
        result = handler(job["params"], load_inputs(path, job_id), progress)
        finish(path, job_id, worker, result)
        return DONE
    except KeyboardInterrupt:
        release(path, job_id, worker)
        raise
    except Exception:  # noqa: BLE001 — any handler failure is the job's result
        fail(path, job_id, worker, traceback.format_exc(limit=20))
        return FAILED
    finally:
        beat.stopped.set()
        beat.join(timeout=heartbeat_every + 1)


def run_here(path: str, kind: str, params: Mapping[str, Any], inputs: Mapping[str, pd.DataFrame],
             handler: Handler, label: str = "", force: bool = False,
             progress: Progress | None = None) -> dict[str, Any]:
    """Submit and run a job in this process, through the same store.

    For a caller with no worker running. The result is stored exactly as a
    worker would store it, so the next reader gets it from the queue either way
    and there is one code path for what a result is, not two. If an identical
    job is already running elsewhere, that job is returned untouched.
    """
    job = submit(path, kind, params, inputs, label=label, force=force)
    if job["status"] in (DONE, RUNNING):
        return job
    worker = f"inline:{new_worker_id()}"
    claimed = _claim_specific(path, int(job["id"]), worker, run_manifest({"worker": worker}))
    if claimed is None:
        return get(path, int(job["id"]))  # type: ignore[return-value]

    def wrapped(params_: Mapping[str, Any], inputs_: Mapping[str, pd.DataFrame],
                report: Progress) -> Mapping[str, Any]:
        def both(message: str) -> None:
            report(message)
            if progress is not None:
                progress(message)
        return handler(params_, inputs_, both)

    run_job(path, claimed, {kind: wrapped}, worker)
    return get(path, int(job["id"]))  # type: ignore[return-value]


def _claim_specific(path: str, job_id: int, worker: str,
                    worker_manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        now = _now()
        cursor = connection.execute(
            "UPDATE jobs SET status = ?, started_at = ?, heartbeat_at = ?, worker = ?, "
            "attempts = attempts + 1, error = NULL, worker_manifest = ? WHERE id = ? AND status = ?",
            (RUNNING, now, now, worker, json.dumps(_encode_value(dict(worker_manifest))), job_id, QUEUED))
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return get(path, job_id) if cursor.rowcount else None


def run_worker(path: str, handlers: Mapping[str, Handler], poll_seconds: float = 2.0,
               once: bool = False, max_jobs: int | None = None,
               stale_after: float = STALE_AFTER_SECONDS, max_attempts: int = MAX_ATTEMPTS,
               heartbeat_every: float = HEARTBEAT_SECONDS,
               log: Callable[[str], None] = print) -> int:
    """Claim and run jobs until stopped. Returns how many jobs it ran.

    `once` drains what is queued now and returns instead of polling, which is
    the shape a scheduled run wants. The worker records the code it started
    from (`core/manifest.py`) on every job it claims: a worker keeps running
    the code it imported, so a job run after the tree was edited was produced
    by the worker's commit, not the tree's, and the store says which.
    """
    worker = new_worker_id()
    manifest = run_manifest({"worker": worker})
    register_worker(path, worker, handlers.keys(), manifest)
    log(f"worker {worker} on {path}: {', '.join(sorted(handlers))}")
    ran = 0
    try:
        while max_jobs is None or ran < max_jobs:
            beat_worker(path, worker)
            for job_id in requeue_stale(path, stale_after=stale_after, max_attempts=max_attempts):
                log(f"requeued or failed stale job {job_id}")
            job = claim(path, worker, handlers.keys(), manifest)
            if job is None:
                if once:
                    break
                time.sleep(poll_seconds)
                continue
            log(f"job {job['id']} {job['kind']} ({job['label'] or 'no label'}) started")
            started = time.monotonic()
            status = run_job(path, job, handlers, worker, heartbeat_every=heartbeat_every)
            ran += 1
            log(f"job {job['id']} {status} in {time.monotonic() - started:.1f}s")
    finally:
        unregister_worker(path, worker)
    return ran
