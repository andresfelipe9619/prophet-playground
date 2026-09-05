# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It is the condensed version. Full documentation lives in [`docs/`](docs/README.md) — start with [`docs/domain-and-premise.md`](docs/domain-and-premise.md), which explains why several decisions here look wrong until you know the premise. When you change behaviour that these notes describe, update the matching page in `docs/` too.

## What this project is

Analysis and forecasting over historical Colombian Baloto lottery results. A ticket is 5 distinct numbers from 1-43 plus one "superbalota" from 1-16; draws run Monday, Wednesday and Saturday.

**The domain constraint that shapes the whole architecture:** lottery draws are i.i.d. uniform by design, so no model can beat chance. The codebase was deliberately refactored around this. Every model output is paired with the chance baseline it must beat, and every heuristic (hot/cold, "overdue" numbers) carries an explicit note that it has no predictive value. Do not "improve" a model by adding seasonalities, holiday regressors, or tuned hyperparameters that fit historical noise — that is the anti-pattern this repo was moved away from, and [`docs/domain-and-premise.md`](docs/domain-and-premise.md#6-anti-patterns) explains why. If a change makes a model look better on history without beating the chance baseline in `backtest.py`, it made the project worse.

## Setup and commands

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

streamlit run dashboard/app.py          # main entry point
python Prophet.py                        # per-position Prophet forecast
python StatsForecast.py AutoARIMA        # or AutoETS / AutoTheta
python XGBoost.py
python backtest.py --n-windows 20 --min-train 100 [--include-prophet]
python backtest.py --cutoff 2026-07-31 --mode frozen --current-format-only
python -m analysis.power --n-draws 1035          # what edge could this much data detect?
python -m analysis.sensitivity --n-seeds 10      # can the tests detect a planted edge?
python -m analysis.popularity --tickets-sold 3000000    # jackpot splitting by combination
python -m analysis.registry record --label Prophet --main 3-12-19-27-41 --super 8
python -m analysis.registry score                # score every registered draw that has happened
```

There is no test suite, linter, or CI configured. Changes are verified by:

1. `python -m py_compile <files>` for syntax.
2. Running the affected module against synthetic data from `utils.sample_data.load_sample_and_preprocess()`, which returns the same `(df, balls_expanded)` shape as the real loader — no private CSVs needed.
3. `python -m utils.check_docs` after any documentation change — it reimplements github-slugger exactly, because headings here use `·` and `—` and GitHub removes those without collapsing the spaces they leave (`### 6 · Jugadas — generate` anchors as `6--jugadas--generate`, doubled hyphens). A checker that normalises hyphen runs passes links that 404 in the browser.
4. For dashboard changes, launching Streamlit headless and driving it with Playwright (Chromium at `/opt/pw-browsers/chromium`). Streamlit only executes the script when a client connects over the websocket, so an HTTP 200 on `/` proves nothing — you must load the page in a browser and check tab text for `Traceback` / "This app has encountered an error". Note that all tab panels stay mounted in the DOM, so scope Playwright locators to `get_by_role("tabpanel", name=...)` or they match across tabs.

`backtest.py` with `--include-prophet` refits Prophet per position per window and is far slower than the other models; it is off by default for that reason.

## Data contract

Scripts read `exported_data/final-final.csv` — gitignored, not in the repo. Columns: `Date` (dd/mm/yyyy) and `Ball`, six dash-separated numbers where the **last one is the superbalota** (e.g. `3-12-19-27-41-8`). `utils/processor.py:preprocess_draws` is the single owner of that contract; `load_and_preprocess` (CSV), the dashboard's upload path and `utils/sample_data.py` all go through it.

**Two eras of the game.** Baloto changed rules in April 2017 (before: 6 balls from 1-45, no superbalota). Both eras publish as six dash-separated numbers, so a long history has the same *shape* throughout and only the values give the mix away — a real 2010-2026 export splits 707/1035 across the change. `preprocess_draws` warns rather than raises (the rows are real draws; a caller may want them), and `format_violations` / `current_format_mask` / `check_draw_format` in the same module do the detection. Note the two masks differ deliberately: `format_violations` can only flag rows that are *provably* impossible today, so `current_format_mask` cuts at the date of the last violation instead — an old-era draw that happens to fit today's bounds would otherwise survive. Use `load_and_preprocess(..., current_format_only=True)`, `--current-format-only`, or the dashboard's sidebar checkbox (on by default).

`python -m utils.scraper --years 2020-2025` builds that file from loterias.com, merging into whatever is already there. Its parser raises rather than skipping on anything unexpected (no rows, wrong ball count, unknown month) — a scraper that silently writes an empty CSV when the markup changes is the failure mode this module is shaped to avoid. `parse_results_page(html)` takes HTML and does no I/O, so it can be tested against saved pages; nothing in the repo can verify it against the live site, so `--dry-run` exists to eyeball a scrape before writing.

When no real CSV is present the dashboard falls back to synthetic data and says so on screen.

## Cross-cutting invariants

**Position semantics.** Throughout the codebase a "position" is a column index into `balls_expanded`. `models/common.py` is the single source of truth for what each position means: `super_position(n_columns)` (the last column, range 1-16), `main_positions(n_columns)` (all the others, range 1-43), plus `range_for_position` / `clip_to_range` / `series_label`. Derive positions from those helpers rather than writing `n_columns - 1` or `range(5)` — the two only coincide when there are exactly 6 columns, and nothing else in the codebase assumes that. Any new model must clip its raw output through `clip_to_range` or it will emit out-of-range numbers.

Pool sizes (`MAIN_POOL`, `SUPER_POOL`), the draw calendar and `DEFAULT_DATA_PATH` also live in `common.py` and are imported, never re-declared.

**Two time axes.** Prophet and all display code use calendar dates (`ds`). statsforecast models use a sequential integer draw index instead, assigned in `models/common.py:to_long_format`, because draw days are 2 and 3 days apart and no fixed pandas frequency fits them. Never introduce a fixed `freq=` for future dates — use `next_draw_dates(last_date, h, weekdays=infer_draw_weekdays(history))`. `infer_draw_weekdays` reads the schedule off the recent tail of the data so that a pre-Monday history keeps generating Wed/Sat dates rather than phantom Monday draws.

**Backtest scoring is set-based and order-agnostic.** `backtest.py:_score_window` compares the *set* of the 5 main-position predictions against the *set* of actual main balls; slot order is irrelevant. This is what makes the exact hypergeometric baseline in `models/baseline.py` the correct comparison, and it is why order-statistic artifacts in the source data cannot inflate backtest results. Preserve this property in any new scoring code.

**The sorted-data trap.** Official results are often published with the main balls sorted ascending. When that happens each column is an order statistic (min, 2nd-smallest, ...), not a uniform draw, and a per-position chi-square test will report spurious "non-random" structure. `analysis/randomness.py:is_sorted_ascending` detects this and `pooled_uniformity_test` is the sort-proof alternative (it pools columns and only asks whether every number appears equally often). It takes only the positions to pool and derives the range from them, so pooling the 1-16 superbalota with the 1-43 main balls is not expressible — a mistake that value range-checking cannot catch, since 1-16 is a subset of 1-43. Per-position tests are diagnostics; the pooled test is the verdict. Also note six simultaneous per-position tests produce false positives at the usual rate — the dashboard says so, and any new test surface should too.

**Multiple comparisons are corrected everywhere.** A backtest scores k models against the same draws and `compare_strategies` scores k strategies the same way, so both emit `beats_chance` (naive, α = 0.05) *and* `beats_chance_corrected` (Bonferroni, α/k), and every surface points the reader at the corrected column. With six models the naive bar is cleared by luck ~26% of the time — "one of my six models beat chance" is exactly the sentence a lottery system gets built on. A new evaluation table that reports one uncorrected verdict has reintroduced the bug.

**A null result needs its resolution attached.** `analysis/power.py` computes the minimum detectable effect for a given number of draws, and every surface reporting "nothing beat chance" should say what it could have detected: 15 windows cannot see a +75% edge, 1035 draws bottom out at +9%, and the MDE only falls with the square root of N. `analysis/sensitivity.py` is the other half — it plants a known bias and measures how often each detector fires, so "the tests found nothing" is backed by evidence that they can find something. Its `strength = 0` row is the control; a detector firing well above alpha there means *something* is broken, though not necessarily the thing under test — the first time it fired, the fault was in the harness, not the estimator.

**Suspect the harness before the statistic.** When a control arm misbehaves, check how the experiment was wired before concluding the tested code is wrong. `beats_chance_test` was measured under the null and is calibrated (sd(z) = 0.997 at 1 ticket per draw, 0.927 at 5, against a nominal 1.0), including when several tickets share a draw — so do not add a cluster-robust variance for that; one was written, measured against the exact test, and removed. Measure sd(z) under the null before changing an estimator.

**Chance comparisons are one-sided.** `models/baseline.py:beats_chance_test` returns `p_value` (two-sided, "differs from chance") and `p_value_greater` (one-sided, "better than chance"). Only the one-sided value may back a "beats chance" claim — a model significantly *worse* than chance also gets a small two-sided p-value.

## Module layout

- `models/common.py` — ball ranges, draw calendar, position helpers, long-format conversion. Everything else imports its constants from here.
- `models/baseline.py` — exact hypergeometric chance baseline and `beats_chance_test`, the z-test that decides whether a model has real signal. `m_guessed` accepts a per-window list because collisions between positions change the distinct-guess count.
- `models/statsforecast_model.py` — AutoARIMA/AutoETS/AutoTheta via Nixtla. `fit_predict_all` fits **every position and all three models in one call**; don't loop per position for these.
- `models/xgboost_model.py` — chronological splits only. The original code used a shuffled `train_test_split`, leaking future draws into training; keep splits time-ordered.
- `analysis/randomness.py` — frequency, gaps, hot/cold, chi-square (per-position and pooled), runs test, ACF/Ljung-Box.
- `analysis/tickets.py` — generate tickets (random/hot/cold/model/portfolio), check them against draws, and `evaluate_strategy`/`stability_check`, which measure whether a generation strategy beats chance. `stability_check` exists because a single run flags a signal-free strategy ~1 time in 20; `random` is the control whose flag rate is the measured false-positive floor.
- `analysis/power.py` — minimum detectable effect, required draws, power curves. Mirrors the z-test in `baseline.py` exactly, since the point is to characterise that test. The superbalota helper is separate because it is Bernoulli, not hypergeometric — using the wrong variance understates the required data ~3x.
- `analysis/sensitivity.py` — plants a known bias and measures detection rates for `pooled`, `hot` and `random`. `random` must stay near alpha even on biased data (a uniform ticket's expected matches do not depend on the weighting) — that is the control, not a failure. **Its own load-bearing detail:** the draw generator and the ticket generator must be driven by *independent* streams, via `independent_seeds`. Passing one seed to both makes the numbers drawn and the numbers played come out of the same `default_rng`, which is a real ticket/draw dependence — exactly what a lottery test hunts for. That mistake reported a 17.5% false-positive rate on bias-free data and sent a full round of investigation after a phantom defect in `evaluate_strategy` (see `docs/evaluation.md`). Any new detector needing randomness takes its seed from `independent_seeds`.
- `analysis/popularity.py` — jackpot splitting. The only module here that improves anything, and it improves `E[payout | win]`, never `P(win)`. `popularity_score` is **ordinal**, not absolute: its weights cannot be calibrated without data on tickets people bought, which no operator publishes, so `split_adjusted_value` returns a band rather than a figure. Registers `unpopular` in `STRATEGIES`, which correctly fails the hit-rate tests — the accuracy machinery is structurally unable to measure what it targets, and that is documented rather than exempted.
- `analysis/registry.py` — append-only pre-registration of predictions. Refuses a draw date that is not in the future, refuses a second prediction for the same (draw, label), and scores every eligible row or none. Writes `predictions.csv` at the repo root, deliberately **not** gitignored: committing it dates each prediction in version control, which beats any timestamp the file writes about itself.
- `analysis/prizes.py` — exact prize-category probabilities, expected value, RTP, breakeven jackpot. Pure combinatorics, needs no historical data. Prize amounts are caller-supplied, never hardcoded, because tiers are operator-set and the top prize accumulates.
- `backtest.py` — walk-forward evaluation. `run_all` holds out the last N draws; `run_holdout(..., cutoff, mode=)` holds out everything after a date, either refitting per draw (`expanding`) or from one fit at the cutoff (`frozen`). Both score through the same `_score_window`, so their summaries are comparable.
- `dashboard/app.py` — Streamlit UI, the primary surface. Inserts the repo root on `sys.path` so it can import the root-level model scripts. All explanatory tooltip copy lives in one `HELP` dict at the top and is consumed by two helpers: `section(title, key)` (a subheader with its ⓘ) and `chart(fig, title, key)` (a titled line with its ⓘ, then the figure). `chart` moves the title out of the Plotly figure because Plotly's own title has nowhere to hang a help icon — clear it with `title={"text": ""}`, since `title=None` makes Plotly render the literal string `undefined`. A bare `st.plotly_chart` outside that helper means a chart was added without an explanation.
- `summary_charts.py` — the original static matplotlib/seaborn charts, superseded by the dashboard for exploratory use.

`contants.py` is misspelled (not `constants.py`) but is imported under that name; renaming it means updating its importers.

## Language

**English** for code identifiers, docstrings, comments, all documentation (`README.md`, `docs/`, this file) and commit messages. **Spanish** only for user-facing dashboard strings, since the dashboard is the end-user product.

## Documentation map

Keep these in sync when behaviour changes:

| Page | Covers |
| --- | --- |
| [`docs/domain-and-premise.md`](docs/domain-and-premise.md) | The i.i.d. premise, the sorted-data trap, anti-patterns |
| [`docs/architecture.md`](docs/architecture.md) | Layers, data flow, position semantics, two time axes, invariants |
| [`docs/data-pipeline.md`](docs/data-pipeline.md) | Data contract, scraper state machine, source resolution |
| [`docs/models.md`](docs/models.md) | Every predictor and how to add one |
| [`docs/tickets.md`](docs/tickets.md) | Generating, checking and measuring ticket strategies |
| [`docs/evaluation.md`](docs/evaluation.md) | Backtest, chance baseline, randomness tests, known past bugs |
| [`docs/jackpot-splitting.md`](docs/jackpot-splitting.md) | Combination popularity, expected co-winners, split-adjusted value |
| [`docs/registry.md`](docs/registry.md) | Pre-registration: predictions recorded before the draw |
| [`docs/power-and-sensitivity.md`](docs/power-and-sensitivity.md) | Minimum detectable effect, planted-bias detection rates |
| [`docs/dashboard.md`](docs/dashboard.md) | The ten tabs and how to read them |
| [`docs/development.md`](docs/development.md) | Setup, verification workflow, conventions, gotchas |
