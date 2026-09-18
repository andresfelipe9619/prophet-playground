# Baloto Analytics

[![Tests](https://github.com/andresfelipe9619/prophet-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/andresfelipe9619/prophet-playground/actions/workflows/tests.yml)

Analysis, forecasting and expected-value tooling for the Colombian **Baloto**
lottery — 5 balls from 1–43 plus a superbalota from 1–16, drawn Monday, Wednesday
and Saturday.

📖 **[Full documentation](docs/README.md)** · 🎯 **[Start with the premise](docs/domain-and-premise.md)**

---

## What this is

A measurement instrument for number-picking strategies. It **generates tickets,
checks them against real draws, and measures whether any way of choosing them
does better than picking at random.**

| It does | It does not |
| --- | --- |
| **Generate tickets** — random, hot, cold, model-driven, or a spread portfolio | Claim any ticket is more likely to win than another |
| **Check tickets** against every historical draw, by prize category | Report accuracy without its chance baseline |
| **Measure strategies** against the exact hypergeometric baseline | Claim to have found a pattern that isn't there |
| Compute **exact** prize probabilities and the expected value of a ticket | Endorse hot/cold heuristics (it labels them and measures them) |

Generating numbers is perfectly legitimate, and so is checking them. What no
strategy can do is make one ticket *more likely* to win than another, because all
15,401,568 combinations are equally probable. **That claim is not asked for on
faith — it is the thing the project measures**, on your own data, with a
significance test.

That is the science here: a null result you can reproduce is a real result. A
pipeline that correctly reports "no signal" on data that provably has none is one
you can trust when the answer is not known in advance — which is why the
forecasting half (walk-forward backtesting, leakage avoidance, multi-series
fitting) is built to the standard it is.

> **The governing rule:** every result is paired with the chance baseline it must
> beat. If a change makes a strategy look better on history without beating that
> baseline, it made the project worse.

The one question with an exact, non-statistical answer is **what a ticket is
worth**: the expected return of a 5,700 COP ticket is arithmetic, and the
dashboard computes it to the peso.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

streamlit run dashboard/app.py
```

No data required — every domain page falls back to synthetic data and says so on
screen. To use real results:

```bash
python -m lottery.utils.scraper --years 2024 --dry-run    # inspect what it parses
python -m lottery.utils.scraper --years 2020-2025          # write exported_data/final-final.csv
```

Then, the question that matters — generate tickets and measure whether the way
you chose them beat chance:

```bash
python -m lottery.backtest --n-windows 20 --min-train 100   # do the models beat chance?

# Or the tangible version: train on everything up to July, then predict the
# draws of August and September that have already happened.
python -m lottery.backtest --cutoff 2026-07-31 --mode frozen --current-format-only
```

```python
from lottery.utils.sample_data import load_sample_and_preprocess
from lottery.analysis.tickets import generate_portfolio, check_against_history, compare_strategies

df, balls = load_sample_and_preprocess(n_draws=800)   # or load_and_preprocess(your_csv)

tickets = generate_portfolio(5)                        # five spread tickets
print(tickets[0])                                      # 8 - 19 - 23 - 34 - 41  +  12

print(check_against_history(tickets[0], df, balls))    # how it would have done, draw by draw
print(compare_strategies(df, balls))                   # random vs hot vs cold, vs chance
```

## The dashboard

One app, three domains, picked from the sidebar. Spanish UI. Full guide in
**[docs/dashboard.md](docs/dashboard.md)**.

| Domain | The baseline | What is there |
| --- | --- | --- |
| 🎯 **Baloto** | exact chance | The ten tabs below: models, chance baseline, backtest, tickets, registry |
| ⚽ **Fútbol** | the closing price | Data, market, a two-team forecast (Dixon-Coles + Elo), a multi-model verdict, and staking |
| 🚴 **Ciclismo** | the pre-race ranking | Data, a race forecast (ranking + Plackett-Luce), and the verdict |

All three can answer "did this beat its baseline?" — but they are **not the same
question**, and the selector names which one. What differs between them is what
can be found there at all: nothing in Baloto, a great deal in the other two.

It is built for a phone as well as a desktop: below 640px the columns stack,
metrics go two-up, the tab strip swipes and chart legends move above the plot.
All of it lives in [`dashboard/mobile.py`](dashboard/mobile.py).

### Reading it on your phone

There is **no hosted instance and no deployment** — this runs on your machine,
which is what lets every model run at full strength on your real data. To read it
on a phone, run the server and open it over your own network:

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0   # then open the Network URL
```

The laptop keeps the data, the models and the GPU; the phone is only a screen.
See **[docs/local-setup.md](docs/local-setup.md)**.

### Baloto's tabs

| Tab | Purpose |
| --- | --- |
| **Resumen** | Draw counts, sorted-data warning, pooled uniformity verdict |
| **Probabilidades y Valor Esperado** | Exact category odds, editable prize table, EV, RTP, breakeven jackpot |
| **Frecuencia y Gaps** | Per-number frequency vs uniform, gap and "overdue" table |
| **Hot / Cold** | Recent vs all-time share |
| **Aleatoriedad** | chi-square, runs test, Ljung–Box, ACF — is there any signal? |
| **Forecast** | Next-draw suggestion from any of the seven models, including Google's TimesFM |
| **Jugadas** | Generate tickets, check one against your whole history, and measure strategies against chance |
| **Backtest vs. Azar** | Walk-forward accuracy against the hypergeometric baseline, over the last N draws or everything after a date you pick |
| **Potencia y Sensibilidad** | What edge this much data could detect, and whether the tests fire on a planted one |
| **Registro** | Predictions recorded before the draw, scored after it |

## Command reference

```bash
streamlit run dashboard/app.py                       # main entry point

python -m lottery.backtest --n-windows 20 --min-train 100    # evaluate vs chance
python -m lottery.backtest --n-windows 20 --include-prophet  # +11% runtime
python -m lottery.backtest --n-windows 20 --include-timesfm  # add Google's TimesFM (zero-shot)
python -m lottery.backtest --cutoff 2026-07-31 --mode frozen # hold out everything after a date
python -m lottery.backtest --current-format-only             # drop pre-2017 draws (rules changed)

python -m scripts.prophet_forecast                                     # per-position Prophet forecast
python -m scripts.statsforecast_forecast AutoARIMA                     # or AutoETS / AutoTheta
python -m scripts.xgboost_forecast                                     # held-out evaluation

python -m lottery.analysis.power --n-draws 1035        # what edge could this data detect?
python -m lottery.analysis.sensitivity --n-seeds 10   # can the tests detect a planted edge?
python -m lottery.analysis.popularity --tickets-sold 3000000  # jackpot splitting
python -m lottery.analysis.registry record --label yo --main 3-12-19-27-41 --super 8
python -m lottery.analysis.registry score             # score the draws that have happened

python -m lottery.utils.scraper --years 2020-2025      # build/update the Baloto dataset
python -m football.downloader --seasons 2019/20..2024/25 --leagues E0   # football seasons
python -m cycling.scraper --race tour-de-france --year 2024 --stages 1-21  # cycling results
python -m lottery.utils.lib_detector                  # print library versions
python -m lottery.utils.check_docs                    # verify documentation links

pytest                                                # the invariant test suite
pytest -m "not slow"                                  # skip runs that fit real models
```

## Repository layout

The tree is split by domain. `core/` is evaluation machinery that knows nothing
about lotteries; `lottery/`, `football/` and `cycling/` are the domains, and each
supplies `core/` with the baseline and the scoring rule it deliberately lacks.

```
core/
  windows.py               Walk-forward and date-cutoff splits
  significance.py          ★ z-test vs a null, Bonferroni correction
football/                  Second domain — real signal, market baseline
  common.py                The three outcomes and their (H, D, A) ordering
  processor.py             ★ football-data.co.uk contract; never mixes opening/closing odds
  market.py                Odds → calibrated probabilities; the baseline to beat
  dixon_coles.py           Attack/defence strengths, the low-score correction
  elo.py                   The cheap baseline; ordered logit fitted for the draw
  ensemble.py              Pooling a model with the market (weight 0 IS the market)
  scoring.py               Brier / RPS / log-loss; ★ RPS is the verdict
  evaluation.py            ★ Paired one-sided test against the market
  backtest.py              Walk-forward; compare_models corrects across models
  value.py                 Edge, the two bars, quarter-Kelly staking
  downloader.py            football-data.co.uk → exported_data/football/
  sample_data.py           Synthetic seasons carrying the generative truth
cycling/                   Third domain — an ordering, not an outcome
  common.py                Result kinds, finish statuses, cycling time parsing
  processor.py             ★ Result contract; one kind per frame, gaps never stored as totals
  baseline.py              Plackett-Luce worths; ★ a uniform draw is NOT a baseline
  scoring.py               ★ PL log score is the verdict; abandons stay in the denominator
  plackett_luce.py         Rider strengths by MM, shrunk toward the field
  evaluation.py            ★ Paired one-sided test against the ranking
  scraper.py               procyclingstats.com → exported_data/cycling/
  sample_data.py           Synthetic stage races carrying the generative truth
lottery/
  backtest.py              Walk-forward evaluation vs chance
  models/
    common.py              ★ Game rules, position helpers, draw calendar
    baseline.py            Hypergeometric chance baseline + significance test
    statsforecast_model.py AutoARIMA / AutoETS / AutoTheta (Nixtla)
    xgboost_model.py       Lag features, chronological splits, forecast_next
    prophet_model.py       Prophet per position
  analysis/
    randomness.py          Frequency, gaps, hot/cold, chi-square, runs, ACF
    structure.py           Sum / parity / calendar split vs their exact distributions
    prizes.py              Exact prize probabilities, EV, RTP, breakeven jackpot
    tickets.py             Generate tickets, check them, measure strategies vs chance
  utils/
    processor.py           ★ The data contract
    scraper.py             loterias.com → project CSV
    sample_data.py         Synthetic i.i.d. draws
scripts/                   CLI entry points (python -m scripts.<name>)
tests/                     pytest suite pinning the invariants
dashboard/
  app.py                   Shell: page config, domain selector, dispatch
  ui.py                    ★ HELP / PLAIN / READ / GLOSSARY + section() / chart()
  baloto_page.py           The ten Baloto tabs
  football_page.py         Datos · Mercado · Pronóstico · ¿Le gana al mercado? · Valor
  cycling_page.py          Datos · Abandonos · Tiempos · Pronóstico · ¿Le gana al ranking?
docs/                      Full documentation
```

## Data

Scripts read `exported_data/final-final.csv` — **gitignored, not in this repo**.
Columns: `Date` (dd/mm/yyyy) and `Ball` (six dash-separated numbers, superbalota
last, e.g. `3-12-19-27-41-8`). Details in
**[docs/data-pipeline.md](docs/data-pipeline.md)**.

Football seasons live in `exported_data/football/` and cycling results in
`exported_data/cycling/`, both also gitignored and both with contracts of their
own — see **[docs/football.md](docs/football.md)** and
**[docs/cycling.md](docs/cycling.md)**.

Always run a scraper with `--dry-run` first and compare its output against the
site — nothing in this repository can verify any of the three parsers against
the live source.

## Contributing

Run `pytest` before and after any change. The suite pins the project's
invariants rather than chasing coverage, and is deterministic — it builds its
data from a seeded generator, so no private CSV is needed.

Conventions, the full verification workflow (see
[docs/development.md §3](docs/development.md#3-verification)) and instructions for
adding models or tests are documented in
**[docs/development.md](docs/development.md)**.

## Disclaimer

This is a research and analysis tool, not a betting system. The expected value of a
Baloto ticket is materially negative, and this project will tell you exactly how
negative. Nothing here improves the odds of any ticket over any other — all
15,401,568 combinations are equally likely, and the **Jugadas** tab exists to
demonstrate that rather than assert it.
