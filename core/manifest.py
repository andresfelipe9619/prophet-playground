"""What produced a result, recorded beside the result.

Everything else in this project is built around the idea that a number is
worth nothing without the thing it must be compared against. This module is
the same idea pointed at the number's own provenance: a backtest summary is
evidence only for as long as you can say which code, which data and which
library versions produced it. Six months later, "the model did not beat
chance" and "some version of the model did not beat some version of the data"
are different claims, and nothing in a results table's shape distinguishes
them.

Two things live here.

**`run_manifest`** answers "what was running": the commit, whether the tree
was dirty, when, on which Python, and with which versions of the libraries
whose arithmetic could move a result. The dirty flag matters more than the
commit — a result produced from an edited working tree is not reproducible
from any commit at all, and that is exactly the state most results are
produced in.

**`data_fingerprint`** answers "which data": a hash over a frame's values,
columns and dtypes. Row counts and date ranges are the usual stand-ins and
both are too coarse — a single corrected cell leaves them identical.

The module is domain-free, which is the whole reason it is here and not in
one of the domain packages: what counts as an input differs per domain (a
draw history, a set of season files, a race calendar), so the caller names
its own inputs and this module records them without knowing what they are.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd

# The libraries whose version can move a number. Not a dependency list: adding
# streamlit here would record a version that cannot change a result, and the
# point of the list is that everything on it can.
NUMERIC_LIBRARIES = ("numpy", "pandas", "scipy", "statsmodels", "statsforecast", "xgboost")


def _git(*args: str) -> str | None:
    """Run a git command in the repository, or return None if that is not possible.

    Returns None rather than raising: a manifest is metadata, and a result
    computed outside a checkout is still a result. What it must never do is
    report a commit it is not certain of.
    """
    try:
        out = subprocess.run(
            ("git", *args), capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_revision() -> dict[str, Any]:
    """The commit a result was produced from, and whether the tree was edited.

    `dirty` is the load-bearing field. A result produced from a modified
    working tree cannot be reproduced from any commit, so a manifest that
    recorded only the SHA would be quietly wrong in the most common case
    there is — mid-change, which is when most results get looked at.
    """
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"commit": None, "dirty": None, "branch": None}
    status = _git("status", "--porcelain")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": sha,
        # `status` is None only if the second call failed where the first
        # succeeded; unknown is not clean, so it reads as dirty.
        "dirty": True if status is None else bool(status),
        "branch": branch or None,
    }


def library_versions(names: tuple[str, ...] = NUMERIC_LIBRARIES) -> dict[str, str | None]:
    """Installed versions of the libraries whose arithmetic can move a result.

    A library that is not installed records None rather than being omitted:
    "statsforecast was absent" is itself a fact about the run, and a missing
    key would read as an oversight.
    """
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def run_manifest(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to say what produced a result.

    `inputs` is the domain's half: the data fingerprints, the seeds, the
    window counts, the cutoff — whatever a caller would need to run the same
    thing again. This module does not know what those are and does not
    inspect them; it only requires that they be JSON-shaped, so that a
    manifest can be written next to a result rather than only printed.
    """
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": library_versions(),
        "inputs": dict(inputs) if inputs else {},
    }


def data_fingerprint(frame: pd.DataFrame) -> str:
    """A stable hash over a frame's columns, dtypes and values.

    Column order and dtype are part of the fingerprint, not just the values:
    a frame whose superbalota column moved would score differently while
    hashing identically under a values-only digest, and position semantics
    are the thing this codebase is most careful about.

    Object columns are hashed through their string form, which is the only
    representation they reliably have. That makes the digest stable for the
    data this project handles (dates, numbers, team and rider names) and
    says nothing useful about a frame holding arbitrary Python objects —
    which none here does.
    """
    digest = hashlib.sha256()
    digest.update(f"{frame.shape}".encode())
    for column in frame.columns:
        series = frame[column]
        digest.update(str(column).encode())
        digest.update(str(series.dtype).encode())
        if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
            digest.update("\x1f".join(series.astype(str)).encode())
        else:
            # `np.ascontiguousarray` because a sliced or transposed frame can
            # hand over a non-contiguous buffer, and `tobytes` on one is an
            # error rather than a different answer.
            digest.update(np.ascontiguousarray(series.to_numpy()).tobytes())
    return digest.hexdigest()


def combined_fingerprint(frames: Mapping[Any, pd.DataFrame]) -> str:
    """One digest over a set of named frames — a per-position or per-season input.

    The keys are hashed alongside the frames and the mapping is walked in
    sorted key order, so the digest does not depend on dict insertion order
    but does change if the same data arrives under different names. In this
    project a key is usually a column position, and a position that moved is
    a different input by definition.
    """
    digest = hashlib.sha256()
    for key in sorted(frames, key=str):
        digest.update(str(key).encode())
        digest.update(data_fingerprint(frames[key]).encode())
    return digest.hexdigest()
