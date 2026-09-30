"""Run the dashboard's background jobs.

    python -m scripts.worker                     # poll forever, every domain it can import
    python -m scripts.worker --once              # drain what is queued now, then exit
    python -m scripts.worker --kinds football.compare_models

Start one beside `streamlit run dashboard/app.py`, or press **Iniciar un worker**
in the dashboard's sidebar, which runs this module detached. Several can run at
once against the same store; `core/jobs.py` hands each job to exactly one.

**A worker keeps running the code it imported.** Edit a model and restart the
worker, or its jobs are produced by the commit it started from — which the
store records on every job, so the page can say so, but which is still not the
code on disk.

Each domain's handlers are imported separately. A worker on a machine without
one domain's dependencies skips that domain, says so, and leaves its jobs in
the queue for a worker that can run them rather than failing them.
"""

import argparse
import importlib

from core import jobs

DOMAIN_MODULES = ("lottery.jobs", "football.jobs", "cycling.jobs")


def available_handlers(modules=DOMAIN_MODULES, log=print):
    """Every handler this environment can import, domain by domain."""
    handlers = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            log(f"skipping {name}: {exc}")
            continue
        handlers.update(module.HANDLERS)
    return handlers


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=jobs.DEFAULT_PATH)
    parser.add_argument("--once", action="store_true",
                        help="run what is queued now and exit instead of polling")
    parser.add_argument("--poll", type=float, default=2.0, help="seconds between idle polls")
    parser.add_argument("--kinds", default="",
                        help="comma-separated job kinds to accept; default is every one importable")
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser


def main(argv=None, log=print):
    args = build_parser().parse_args(argv)
    handlers = available_handlers(log=log)
    if args.kinds:
        wanted = {kind.strip() for kind in args.kinds.split(",") if kind.strip()}
        unknown = sorted(wanted - set(handlers))
        if unknown:
            log(f"unknown or unavailable job kinds: {', '.join(unknown)}")
            return 2
        handlers = {kind: handlers[kind] for kind in wanted}
    if not handlers:
        log("no job handlers could be imported; nothing to run")
        return 2
    try:
        ran = jobs.run_worker(args.store, handlers, poll_seconds=args.poll, once=args.once,
                              max_jobs=args.max_jobs, log=log)
    except KeyboardInterrupt:
        log("stopped; any job in progress went back to the queue")
        return 0
    log(f"ran {ran} job(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
