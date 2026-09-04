# Baloto Analytics

Analysis, forecasting and expected-value tooling for the Colombian **Baloto**
lottery — 5 balls from 1–43 plus a superbalota from 1–16, drawn Monday, Wednesday
and Saturday.

📖 **[Full documentation](docs/README.md)** · 🎯 **[Start with the premise](docs/domain-and-premise.md)**

---

## What this is, and what it is not

A fair lottery draw is **independent and identically distributed**. There is no
trend to extrapolate, no season to fit, and no number that is "due". It follows
that **no model in this repository can beat chance**, and none of them do.

So the project is built around that fact rather than against it:

| It does | It does not |
| --- | --- |
| Compute **exact** prize probabilities and the expected value of a ticket | Predict winning numbers |
| Test whether your draw data is statistically consistent with a fair lottery | Claim to have found a pattern |
| Run forecasting models **and report the chance level beside every result** | Report accuracy without its baseline |
| Show hot/cold and "overdue" views, labelled as the gambler's fallacy they are | Endorse those heuristics |

The one question here with an exact answer is **what a ticket is worth**. Which
numbers come up is unknowable; the expected return of a 5,700 COP ticket is
arithmetic, and the dashboard computes it to the peso.

The forecasting half is a genuine engineering exercise — walk-forward backtesting,
leakage avoidance, multi-series fitting — in a domain where the correct answer is
known in advance. A pipeline that correctly reports "no signal" on data that
provably has none is one you can trust elsewhere.

> **The governing rule:** every model output is paired with the chance baseline it
> must beat. If a change makes a model look better on history without beating that
> baseline, it made the project worse.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

streamlit run dashboard/app.py
```

No data required — the dashboard falls back to synthetic draws and says so on
screen. To use real results:

```bash
python -m utils.scraper --years 2024 --dry-run    # inspect what it parses
python -m utils.scraper --years 2020-2025          # write exported_data/final-final.csv
```

Then, the question that matters:

```bash
python backtest.py --n-windows 20 --min-train 100  # does any model beat chance?
```

## The dashboard

Seven tabs, Spanish UI. Full guide in **[docs/dashboard.md](docs/dashboard.md)**.

| Tab | Purpose |
| --- | --- |
| **Resumen** | Draw counts, sorted-data warning, pooled uniformity verdict |
| **Probabilidades y Valor Esperado** | Exact category odds, editable prize table, EV, RTP, breakeven jackpot |
| **Frecuencia y Gaps** | Per-number frequency vs uniform, gap and "overdue" table |
| **Hot / Cold** | Recent vs all-time share |
| **Aleatoriedad** | chi-square, runs test, Ljung–Box, ACF — is there any signal? |
| **Forecast** | Next-draw suggestion from any model |
| **Backtest vs. Azar** | Walk-forward accuracy against the hypergeometric baseline |

## Command reference

```bash
streamlit run dashboard/app.py                       # main entry point

python backtest.py --n-windows 20 --min-train 100    # evaluate vs chance
python backtest.py --n-windows 20 --include-prophet  # include Prophet (slow)

python Prophet.py                                     # per-position Prophet forecast
python StatsForecast.py AutoARIMA                     # or AutoETS / AutoTheta
python XGBoost.py                                     # held-out evaluation

python -m utils.scraper --years 2020-2025             # build/update the dataset
python -m utils.lib_detector                          # print library versions
```

## Repository layout

```
models/
  common.py                ★ Game rules, position helpers, draw calendar
  baseline.py              Hypergeometric chance baseline + significance test
  statsforecast_model.py   AutoARIMA / AutoETS / AutoTheta (Nixtla)
  xgboost_model.py         Lag features, chronological splits, forecast_next
analysis/
  randomness.py            Frequency, gaps, hot/cold, chi-square, runs, ACF
  prizes.py                Exact prize probabilities, EV, RTP, breakeven jackpot
utils/
  processor.py             ★ The data contract
  scraper.py               loterias.com → project CSV
  sample_data.py           Synthetic i.i.d. draws
dashboard/app.py           Streamlit UI
backtest.py                Walk-forward evaluation vs chance
docs/                      Full documentation
```

## Data

Scripts read `exported_data/final-final.csv` — **gitignored, not in this repo**.
Columns: `Date` (dd/mm/yyyy) and `Ball` (six dash-separated numbers, superbalota
last, e.g. `3-12-19-27-41-8`). Details in
**[docs/data-pipeline.md](docs/data-pipeline.md)**.

Always run the scraper with `--dry-run` first and compare its output against the
website — nothing in this repository can verify the parser against the live site.

## Contributing

Conventions, the verification workflow (there is no test suite — see
[docs/development.md §3](docs/development.md#3-verification)) and instructions for
adding models or tests are documented in
**[docs/development.md](docs/development.md)**.

## Disclaimer

This is an analysis tool, not a betting system. The expected value of a Baloto
ticket is materially negative, and this project will tell you exactly how negative.
Nothing here improves the odds of any ticket over any other — all 15,401,568
combinations are equally likely.
