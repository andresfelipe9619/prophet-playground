"""The background-job controls, shared by every heavy evaluation on every page.

Before this, each page ran its backtest inside the button click: the page
blocked behind a spinner, the result lived in `st.session_state`, and a browser
refresh threw it away. `core/jobs.py` moved the work into a queue; this module
is the one place the dashboard talks to it, for the same reason `ui.py` owns
the copy and `betlog_page.py` owns the stake log — three pages each wiring
their own queue could not be reviewed as a set.

What a page gets from `run_panel`:

- **The stored answer for the current question, with no click.** The job is
  keyed on the evaluation, its parameters and the data's fingerprint, so a
  reader who refreshes, comes back tomorrow, or opens the page after the
  nightly run sees the result immediately. Change a slider and the result goes
  away — a result for other parameters is not the answer to this question,
  which the old session-state version showed anyway.
- **Two ways to run it.** "Enviar a segundo plano" hands the job to a worker
  and keeps the page responsive; "Correr aquí" runs it in this process, for a
  reader with no worker, and stores it exactly as a worker would. Whichever is
  available is the primary button.
- **Provenance on every result**: when it was computed, by which worker, from
  which commit, and whether that tree had uncommitted edits — the dirty flag
  that `core/manifest.py` treats as the load-bearing field.

`sidebar_status` is the shell's view of the queue: live workers, counts, recent
jobs, and a button that starts a worker so a reader does not need a second
terminal to use the queue at all.
"""

import os
import subprocess
import sys
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from core import jobs
from dashboard.ui import HELP

JOBS_PATH = jobs.DEFAULT_PATH
WORKER_LOG = os.path.join("exported_data", "worker.log")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATUS_ES = {jobs.QUEUED: "en cola", jobs.RUNNING: "corriendo", jobs.DONE: "listo",
             jobs.FAILED: "falló"}

# Streamlit 1.37 renamed experimental_fragment to fragment; requirements allow
# 1.36. Without either the panel falls back to a manual refresh button.
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def _ago(stamp):
    if not stamp:
        return "—"
    seconds = (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds()
    if seconds < 90:
        return f"hace {int(seconds)} s"
    if seconds < 5400:
        return f"hace {int(seconds // 60)} min"
    if seconds < 172800:
        return f"hace {int(seconds // 3600)} h"
    return f"hace {int(seconds // 86400)} días"


def _duration(job):
    if not (job.get("started_at") and job.get("finished_at")):
        return None
    return (datetime.fromisoformat(job["finished_at"])
            - datetime.fromisoformat(job["started_at"])).total_seconds()


@st.cache_data(show_spinner=False, max_entries=64)
def _load_result(path, job_id):
    """A finished job's result never changes, so it is cached on the job id."""
    return jobs.load_result(path, job_id)


def provenance(job):
    """One line saying what produced a stored result — the thing a table cannot say."""
    manifest = job.get("worker_manifest") or {}
    git = manifest.get("git") or {}
    commit = (git.get("commit") or "")[:7] or "desconocido"
    dirty = " **con cambios sin guardar**" if git.get("dirty") else ""
    where = "en esta página" if str(job.get("worker", "")).startswith("inline:") else "por un worker"
    took = _duration(job)
    took_text = f", tardó {took:.0f} s" if took is not None else ""
    return (f"Calculado {_ago(job.get('finished_at'))} {where}, a partir del commit `{commit}`"
            f"{dirty}{took_text}. Queda guardado: recargar la página no lo pierde.")


def _workers_for(kind):
    try:
        return [w for w in jobs.live_workers(JOBS_PATH) if kind in w["kinds"]]
    except jobs.JobError:
        return []


def _watch(job_id):
    """Poll one job and rerun the whole page the moment it finishes."""
    job = jobs.get(JOBS_PATH, job_id)
    if job is None or job["status"] not in jobs.ACTIVE:
        st.rerun()
        return
    if job["status"] == jobs.QUEUED:
        workers = _workers_for(job["kind"])
        st.info(f"En cola desde {_ago(job['created_at'])}. "
                + ("Un worker lo tomará en segundos." if workers else
                   "No hay ningún worker activo que pueda correrlo — inicia uno en la barra lateral "
                   "o usa **Correr aquí**."))
    else:
        st.info(f"Corriendo desde {_ago(job['started_at'])}"
                + (f" — {job['progress']}" if job.get("progress") else "")
                + ". Puedes seguir usando el panel; el resultado aparece aquí solo.")


if _fragment is not None:
    _watch_live = _fragment(run_every=2)(_watch)
else:  # pragma: no cover — only on Streamlit 1.36 without experimental_fragment
    _watch_live = _watch


def run_panel(kind, params, inputs, handler, *, label, button, key):
    """Render the run controls for one evaluation and return its stored result, or None.

    `inputs` are the frames the evaluation reads; they are stored with the job,
    so the result describes exactly the data on screen. `key` namespaces the
    buttons, since one page can carry several panels.
    """
    try:
        question = jobs.job_key(kind, params, inputs)
        job = jobs.find(JOBS_PATH, question)
    except jobs.JobError as exc:
        st.error(f"El almacén de trabajos no se puede leer: {exc}")
        return None

    active = job is not None and job["status"] in jobs.ACTIVE
    done = job is not None and job["status"] == jobs.DONE
    workers = _workers_for(kind)

    if not active:
        c1, c2 = st.columns(2)
        queue_label = "Volver a calcular en segundo plano" if done else f"{button} en segundo plano"
        here_label = "Volver a calcular aquí" if done else f"{button} aquí"
        queue_clicked = c1.button(queue_label, key=f"{key}_queue",
                                  type="primary" if workers else "secondary",
                                  help=HELP["jobs_queue"])
        here_clicked = c2.button(here_label, key=f"{key}_here",
                                 type="secondary" if workers else "primary",
                                 help=HELP["jobs_run_here"])
        if not workers:
            st.caption(HELP["jobs_no_worker"])

        if queue_clicked:
            try:
                jobs.submit(JOBS_PATH, kind, params, inputs, label=label, force=done)
            except jobs.JobError as exc:
                st.error(str(exc))
                return None
            st.rerun()
        if here_clicked:
            with st.status("Calculando en esta página…", expanded=True) as status:
                try:
                    finished = jobs.run_here(JOBS_PATH, kind, params, inputs, handler, label=label,
                                             force=done, progress=status.write)
                except jobs.JobError as exc:
                    status.update(label="No se pudo guardar el resultado", state="error")
                    st.error(str(exc))
                    return None
                status.update(label="Listo" if finished["status"] == jobs.DONE else "Falló",
                              state="complete" if finished["status"] == jobs.DONE else "error")
            st.rerun()

    if job is None:
        return None
    if active:
        _watch_live(int(job["id"]))
        return None
    if job["status"] == jobs.FAILED:
        lines = [line for line in (job.get("error") or "").strip().splitlines() if line.strip()]
        st.error(f"La última corrida con estos parámetros falló: {lines[-1] if lines else 'sin detalle'}")
        with st.expander("Detalle técnico"):
            st.code(job.get("error") or "", language="text")
        return None

    st.caption(provenance(job), help=HELP["jobs_provenance"])
    return _load_result(JOBS_PATH, int(job["id"]))


def start_worker():
    """Start a worker process detached from this Streamlit session.

    It outlives the tab and the Streamlit rerun that started it, which is the
    point; its output goes to `exported_data/worker.log`.
    """
    os.makedirs(os.path.dirname(WORKER_LOG), exist_ok=True)
    with open(WORKER_LOG, "a", encoding="utf-8") as log:
        subprocess.Popen(  # noqa: S603 — fixed argv, our own module
            [sys.executable, "-m", "scripts.worker", "--store", JOBS_PATH],
            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)


def sidebar_status():
    """The queue at a glance: who is working, what is waiting, what just finished."""
    try:
        workers = jobs.live_workers(JOBS_PATH)
        counts = jobs.counts(JOBS_PATH)
        recent = jobs.list_jobs(JOBS_PATH, limit=8)
    except jobs.JobError as exc:
        st.caption(f"Trabajos: el almacén no se puede leer ({exc}).")
        return

    st.markdown("**Trabajos en segundo plano**", help=HELP["jobs_sidebar"])
    st.caption(
        f"{len(workers)} worker(s) activo(s) · {counts[jobs.RUNNING]} corriendo · "
        f"{counts[jobs.QUEUED]} en cola · {counts[jobs.FAILED]} fallido(s)")
    if not workers:
        if st.button("Iniciar un worker", key="jobs_start_worker", help=HELP["jobs_start_worker"]):
            start_worker()
            st.toast("Worker iniciado. Tarda unos segundos en aparecer aquí.")
    if recent:
        with st.expander("Últimos trabajos"):
            st.dataframe(
                pd.DataFrame([{"#": j["id"], "Qué": j["label"] or j["kind"],
                               "Estado": STATUS_ES.get(j["status"], j["status"]),
                               "Pedido": _ago(j["created_at"])} for j in recent]),
                hide_index=True, use_container_width=True)
