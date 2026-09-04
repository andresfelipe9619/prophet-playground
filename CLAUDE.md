# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Analysis and forecasting over historical Colombian Baloto lottery results. A ticket is 5 distinct numbers from 1-43 plus one "superbalota" from 1-16; draws run Monday, Wednesday and Saturday.

**The domain constraint that shapes the whole architecture:** lottery draws are i.i.d. uniform by design, so no model can beat chance. The codebase was deliberately refactored around this. Every model output is paired with the chance baseline it must beat, and every heuristic (hot/cold, "overdue" numbers) carries an explicit note that it has no predictive value. Do not "improve" a model by adding seasonalities, holiday regressors, or tuned hyperparameters that fit historical noise — that is the anti-pattern this repo was moved away from, and `README_ACCURACY.md` explains why. If a change makes a model look better on history without beating the chance baseline in `backtest.py`, it made the project worse.

## Setup and commands

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

streamlit run dashboard/app.py          # main entry point
python Prophet.py                        # per-position Prophet forecast
python StatsForecast.py AutoARIMA        # or AutoETS / AutoTheta
python XGBoost.py
python backtest.py --n-windows 20 --min-train 100 [--include-prophet]
```

There is no test suite, linter, or CI configured. Changes are verified by:

1. `python -m py_compile <files>` for syntax.
2. Running the affected module against synthetic data from `utils.sample_data.load_sample_and_preprocess()`, which returns the same `(df, balls_expanded)` shape as the real loader — no private CSVs needed.
3. For dashboard changes, launching Streamlit headless and driving it with Playwright (Chromium at `/opt/pw-browsers/chromium`). Streamlit only executes the script when a client connects over the websocket, so an HTTP 200 on `/` proves nothing — you must load the page in a browser and check tab text for `Traceback` / "This app has encountered an error". Note that all tab panels stay mounted in the DOM, so scope Playwright locators to `get_by_role("tabpanel", name=...)` or they match across tabs.

`backtest.py` with `--include-prophet` refits Prophet per position per window and is far slower than the other models; it is off by default for that reason.

## Data contract

Scripts read `exported_data/final-final.csv` — gitignored, not in the repo. Columns: `Date` (dd/mm/yyyy) and `Ball`, six dash-separated numbers where the **last one is the superbalota** (e.g. `3-12-19-27-41-8`). `utils/processor.py:load_and_preprocess` splits `Ball` into `balls_expanded`, a DataFrame with one column per position.

When no real CSV is present the dashboard falls back to synthetic data and says so on screen.

## Cross-cutting invariants

**Position semantics.** Throughout the codebase a "position" is a column index into `balls_expanded`. `models/common.py` is the single source of truth for what each position means: `super_position(n_columns)` (the last column, range 1-16), `main_positions(n_columns)` (all the others, range 1-43), plus `range_for_position` / `clip_to_range` / `series_label`. Derive positions from those helpers rather than writing `n_columns - 1` or `range(5)` — the two only coincide when there are exactly 6 columns, and nothing else in the codebase assumes that. Any new model must clip its raw output through `clip_to_range` or it will emit out-of-range numbers.

Pool sizes (`MAIN_POOL`, `SUPER_POOL`), the draw calendar and `DEFAULT_DATA_PATH` also live in `common.py` and are imported, never re-declared.

**Two time axes.** Prophet and all display code use calendar dates (`ds`). statsforecast models use a sequential integer draw index instead, assigned in `models/common.py:to_long_format`, because draw days are 2 and 3 days apart and no fixed pandas frequency fits them. Never introduce a fixed `freq=` for future dates — use `next_draw_dates(last_date, h, weekdays=infer_draw_weekdays(history))`. `infer_draw_weekdays` reads the schedule off the recent tail of the data so that a pre-Monday history keeps generating Wed/Sat dates rather than phantom Monday draws.

**Backtest scoring is set-based and order-agnostic.** `backtest.py:_score_window` compares the *set* of the 5 main-position predictions against the *set* of actual main balls; slot order is irrelevant. This is what makes the exact hypergeometric baseline in `models/baseline.py` the correct comparison, and it is why order-statistic artifacts in the source data cannot inflate backtest results. Preserve this property in any new scoring code.

**The sorted-data trap.** Official results are often published with the main balls sorted ascending. When that happens each column is an order statistic (min, 2nd-smallest, ...), not a uniform draw, and a per-position chi-square test will report spurious "non-random" structure. `analysis/randomness.py:is_sorted_ascending` detects this and `pooled_uniformity_test` is the sort-proof alternative (it pools columns and only asks whether every number appears equally often). It takes only the positions to pool and derives the range from them, so pooling the 1-16 superbalota with the 1-43 main balls is not expressible — a mistake that value range-checking cannot catch, since 1-16 is a subset of 1-43. Per-position tests are diagnostics; the pooled test is the verdict. Also note six simultaneous per-position tests produce false positives at the usual rate — the dashboard says so, and any new test surface should too.

**Chance comparisons are one-sided.** `models/baseline.py:beats_chance_test` returns `p_value` (two-sided, "differs from chance") and `p_value_greater` (one-sided, "better than chance"). Only the one-sided value may back a "beats chance" claim — a model significantly *worse* than chance also gets a small two-sided p-value.

## Module layout

- `models/common.py` — ball ranges, draw calendar, position helpers, long-format conversion. Everything else imports its constants from here.
- `models/baseline.py` — exact hypergeometric chance baseline and `beats_chance_test`, the z-test that decides whether a model has real signal. `m_guessed` accepts a per-window list because collisions between positions change the distinct-guess count.
- `models/statsforecast_model.py` — AutoARIMA/AutoETS/AutoTheta via Nixtla. `fit_predict_all` fits **every position and all three models in one call**; don't loop per position for these.
- `models/xgboost_model.py` — chronological splits only. The original code used a shuffled `train_test_split`, leaking future draws into training; keep splits time-ordered.
- `analysis/randomness.py` — frequency, gaps, hot/cold, chi-square (per-position and pooled), runs test, ACF/Ljung-Box.
- `analysis/prizes.py` — exact prize-category probabilities, expected value, RTP, breakeven jackpot. Pure combinatorics, needs no historical data. Prize amounts are caller-supplied, never hardcoded, because tiers are operator-set and the top prize accumulates.
- `dashboard/app.py` — Streamlit UI, the primary surface. Inserts the repo root on `sys.path` so it can import the root-level model scripts.
- `summary_charts.py` — the original static matplotlib/seaborn charts, superseded by the dashboard for exploratory use.

`contants.py` is misspelled (not `constants.py`) but is imported under that name; renaming it means updating its importers.

## Language

User-facing dashboard strings and README prose are in Spanish. Code identifiers, docstrings and commit messages are in English.
