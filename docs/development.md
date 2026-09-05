# Development

Setup, how work is verified, conventions, and how to extend the project.

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

| Dependency | Used for |
| --- | --- |
| pandas, numpy | Everything |
| scipy | `hypergeom`, `chisquare`, `norm` |
| statsmodels | `acf`, `acorr_ljungbox` |
| prophet (+ cmdstanpy) | `lottery/models/prophet_model.py` |
| statsforecast **≥ 2.0** | AutoARIMA / AutoETS / AutoTheta |
| xgboost, scikit-learn | `lottery/models/xgboost_model.py` |
| streamlit, plotly | Dashboard |
| requests, beautifulsoup4 | Scraper |
| matplotlib, seaborn | Legacy `scripts/summary_charts.py` |

The statsforecast floor is **not cosmetic**: `unique_id` is an index rather than a
column before 2.0. `fit_predict_all()` normalizes it anyway, but the floor keeps the
two in agreement.

Check what is installed with `python -m lottery.utils.lib_detector`.

`pytest` is needed to run the test suite (§3.1).

## 2. Running things

```bash
streamlit run dashboard/app.py                      # main entry point

python -m scripts.prophet_forecast                                    # per-position Prophet forecast
python -m scripts.statsforecast_forecast AutoARIMA                    # or AutoETS / AutoTheta
python -m scripts.xgboost_forecast                                    # held-out evaluation
python -m lottery.backtest --n-windows 20 --min-train 100    # the verdict
python -m lottery.backtest --n-windows 20 --include-prophet  # slower

python -m lottery.utils.scraper --years 2024 --dry-run       # inspect a scrape
python -m lottery.utils.scraper --years 2020-2025             # build/update the CSV
```

The three model scripts additionally expect
`exported_data/exported_data_2024.csv` for their comparison step, and write into
`prophet_results/`, `statsforecast_<model>_results/` and `xgboost_results/`.

## 3. Verification

**There is a test suite; there is still no linter or CI.** Work is verified in
four ways.

```mermaid
flowchart TD
    C["Change some code"] --> S["1 · pytest<br/>the invariants must still hold"]
    S --> D["2 · Behaviour on synthetic data<br/>lottery.utils.sample_data.load_sample_and_preprocess()"]
    D --> U{"Touched the dashboard?"}
    U -->|"no"| V
    U -->|"yes"| B["3 · Drive it in a real browser<br/>Streamlit headless + Playwright"]
    B --> V{"Touched the docs?"}
    V -->|"no"| DONE["Done"]
    V -->|"yes"| L["4 · Links<br/>python -m lottery.utils.check_docs"]
    L --> DONE
```

There is a fifth check that applies to the statistics rather than the code:
`lottery/analysis/sensitivity.py` plants a known bias and confirms the detectors
fire on it. Synthetic i.i.d. data proves the tests do not cry wolf; only a planted
signal proves they can hear one. See
[Power and Sensitivity](power-and-sensitivity.md#2-sensitivity-lotteryanalysissensitivitypy).

Note how §3.1 and that fifth check divide the work: the suite pins what the code
*does*, the sensitivity run checks what the statistics can *see*. Neither
substitutes for the other.

### 3.1 The test suite

```bash
pytest                      # everything
pytest -m "not slow"        # skip the runs that fit real models (~2s)
```

Every test builds its data from a seeded generator — `lottery/utils/sample_data.py`
for uniform draws, `lottery/analysis/sensitivity.py:biased_draws` where a planted
bias is needed — so the suite is deterministic and needs no private CSV. Tests
that fit real models, or that repeat an experiment across many seeds, are marked
`slow`.

What the suite is *for* is worth being explicit about: it does not chase coverage,
it pins the **invariants** listed in [CLAUDE.md](../CLAUDE.md) and throughout these
docs — the ones a refactor breaks silently and no reader would notice:

| Pinned behaviour | Why it needs a test |
| --- | --- |
| Positions derived from the helpers | `n_columns - 1` and `range(5)` coincide only at 6 columns |
| No fixed `freq=` for future dates | A fixed frequency invents draws on non-draw days |
| `current_format_mask` cuts at a **date** | Row-wise validation cannot catch a legal-looking old-era draw |
| One-sided vs two-sided p-values | A model *worse* than chance also gets a small two-sided p-value |
| Bonferroni on every k-way table | Six models clear the naive 5% bar by luck ~26% of the time |
| Set-based, order-agnostic scoring | It is what makes the hypergeometric baseline the right comparison |
| Pooled uniformity refuses mixed ranges | 1-16 is a subset of 1-43, so range-checking cannot catch the mistake |
| Chronological-only XGBoost splits | A shuffled split was the original bug |
| Effect sizes travel with p-values | A p-value cannot separate "no edge" from "no edge detectable here" |
| The MDE falls only with sqrt(N) | Four times the data buys half the resolution |
| The superbalota uses a Bernoulli variance | Reusing the main-ball figures understates the data needed ~3.3x |
| `independent_seeds` splits the streams | One shared seed couples draws to tickets and invents an edge |
| `strength = 0` is a true control | Every positive sensitivity row is meaningless if it is not |
| Popularity moves `E[payout \| win]`, never `P(win)` | The claim the module is careful not to make |
| The registry refuses past dates and duplicates | Its entire value is in what it will not record |
| Scoring the registry is all-or-nothing | Choosing which predictions count is the failure it prevents |

A syntax-only check is still occasionally useful on files the suite does not import:

```bash
python -m py_compile $(find core lottery scripts dashboard tests -name '*.py')
```

### 3.2 Behaviour, against synthetic data

`lottery.utils.sample_data.load_sample_and_preprocess()` returns the same
`(df, balls_expanded)` shape as the real loader, so any module can be exercised
without private CSVs:

```python
from lottery.utils.sample_data import load_sample_and_preprocess
from lottery.models.common import build_position_series
import lottery.backtest as bt

df, balls_expanded = load_sample_and_preprocess(n_draws=300)
position_series = build_position_series(df, balls_expanded)
results = bt.run_all(position_series, balls_expanded.shape[1], n_windows=5, min_train=60)
print(bt.summarize(results).to_string(index=False))
```

Because the synthetic data is genuinely i.i.d., it doubles as a correctness check
on the statistics: the randomness tests **should** report "looks random", and no
model should beat chance significantly.

### 3.3 Dashboard changes

Two traps make casual checking useless here:

> **An HTTP 200 on `/` proves nothing.** Streamlit only executes the script when a
> client connects over the websocket. You must load the page in a real browser.

> **All tab panels stay mounted in the DOM.** A bare locator matches across tabs.
> Scope every locator: `page.get_by_role("tabpanel", name="Forecast").get_by_...`

```bash
streamlit run dashboard/app.py --server.headless true --server.port 8765
```

Then drive it with Playwright (Chromium is under `/opt/pw-browsers/`; check the exact
versioned path, e.g. `/opt/pw-browsers/chromium-1194/chrome-linux/chrome`), check
each tab's text for `Traceback` and "This app has encountered an error", and click
through the Forecast and Backtest buttons — those code paths only execute on click.


### 3.4 Documentation changes

```bash
python -m lottery.utils.check_docs
```

Exits non-zero and names the near-miss anchors. It reimplements github-slugger
exactly rather than approximating: headings here use `·` and `—`, GitHub strips
those characters *without* collapsing the spaces they leave, and so
`### 6 · Jugadas — generate, check, measure` anchors as
`6--jugadas--generate-check-measure` with doubled hyphens. A checker that
normalises runs of hyphens reports success on links that 404 in the browser —
which is how one broken cross-reference survived several review passes here.

## 4. Conventions

| Area | Rule |
| --- | --- |
| **Language** | Code, docstrings, comments, documentation and commit messages in **English**. Dashboard UI strings in **Spanish** (end-user facing). |
| **Constants** | Game rules live in `lottery/models/common.py` and are imported, never re-declared. |
| **Positions** | Derive from `main_positions()` / `super_position()`. Never `range(5)` or `n - 1`. |
| **Clipping** | Every raw model output passes through `clip_to_range()`. |
| **Future dates** | `next_draw_dates(..., weekdays=infer_draw_weekdays(history))`. Never a fixed pandas `freq=`. |
| **Splits** | Chronological only. Never shuffle a time series. |
| **Chance claims** | `p_value_greater` (one-sided), never `p_value`. |
| **Errors** | Raise with a message naming what to check. No bare `except` that swallows a failure into a success message. |

## 5. How to add things

### A model

See [Models §7](models.md#7-adding-a-model). In short: implement in `models/`,
clip the output, add a `@_predictor("Name")` window function in `lottery/backtest.py`,
register in `run_all()`, add to the dashboard selector.

**Never return the actual draw as a fallback** when a model cannot predict —
return `None` and let `_run_windows()` skip the window.

### A statistical test

1. Add it to `lottery/analysis/randomness.py`, taking `position_series` or `balls_expanded`.
2. If it is per-position, note whether it is affected by the sorted-data problem —
   and if so, provide or point at a pooled alternative.
3. Surface it in dashboard tab 4, **with the multiple-comparisons caveat** if it is
   run across all six positions.

### A dashboard tab

Tabs are positional; inserting one shifts every later index. Keep the house rule:
model output and heuristics always appear next to their chance level or a note on
what they do not mean.

## 6. Gotchas

| Thing | Why it bites |
| --- | --- |
| `lottery/constants.py` | Was misspelled `contants.py` at the repo root; corrected during the move to `lottery/`. |
| Two time axes | Prophet/display use calendar dates; statsforecast uses a draw index. Mixing them silently produces wrong dates. See [Architecture §5](architecture.md#5-two-time-axes). |
| `train_predict` vs `forecast_next` | The first predicts draws that already happened. Presenting it as "the next draw" was a real bug. |
| `exported_data/` | Gitignored. Absent on a fresh clone; the dashboard falls back to synthetic data and says so. |
| Prophet in the backtest | Refits per position per window. Off by default for that reason. |
| Streamlit reruns | The whole script executes on every widget interaction. Expensive work must be cached or behind a button. |
| Legacy files | `scripts/summary_charts.py` and `lottery/utils/csv_merger.py` predate the current pipeline. Superseded, kept for existing local workflows. |

## 7. Git workflow

- Commit messages in English, imperative mood, explaining **why** rather than
  restating the diff.
- The default branch is `master`.
- A merged pull request is finished; follow-up work starts from the updated
  `master` as a new change, not as more commits on the merged branch.

---

**Next:** [Architecture](architecture.md) · [Domain and Premise](domain-and-premise.md)
