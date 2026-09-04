# Evaluation

How this project decides whether anything works. This is the most important
document in the repository after [Domain and Premise](domain-and-premise.md),
because it is where the honesty is enforced mechanically rather than by intention.

## 1. The governing rule

> **Raw accuracy is meaningless without the chance level next to it.**

"The model matched 0.8 of 5 numbers on average" is not information. Pure luck
already gives you ~0.58. The only interesting quantity is the **difference**, and
whether that difference survives a significance test.

Every accuracy number this project reports is paired with its chance baseline.
`backtest.summarize()` cannot emit an accuracy column without a
`chance_avg_main_hits` column beside it.

## 2. Walk-forward backtesting

`backtest.py`. For each of the last `n_windows` real draws, every model is trained
**only on data before that draw** and scored against what actually came out.

```mermaid
flowchart TD
    START["total draws N<br/>start = max(min_train, N − n_windows)"] --> LOOP{"t = start … N−1"}
    LOOP --> TRUNC["history = draws[0 : t]<br/><i>strictly before t</i>"]
    TRUNC --> PRED["predict_window(position_series, t)"]
    PRED -->|"returns None"| SKIP["skip this window<br/><i>never score against truth</i>"]
    PRED -->|"{position: number}"| SCORE["_score_window vs actual draw t"]
    SCORE --> REC["record main_hits, super_hit, m_guessed"]
    SKIP --> NEXT
    REC --> NEXT["t += 1"]
    NEXT --> LOOP
    LOOP -->|"done"| SUM["summarize() → z-test vs chance"]
```

```bash
python backtest.py --n-windows 20 --min-train 100
python backtest.py --n-windows 20 --include-prophet     # slower
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n-windows` | 15 | How many recent draws to evaluate |
| `--min-train` | 60 | Minimum history before the first evaluated window |
| `--include-prophet` | off | Prophet refits per position per window; far slower |
| `--file` | `exported_data/final-final.csv` | Input |

Writes `backtest_summary.csv`.

### Scoring is set-based

`_score_window()` compares the **set** of the 5 main predictions against the
**set** of the 5 actual main balls. Slot order is irrelevant.

```python
main_pred   = {21, 22, 23, 24}     # note: 4 distinct — two positions collided
main_actual = {3, 12, 19, 24, 41}
main_hits   = 1                     # |intersection|
m_guessed   = 4                     # distinct numbers actually committed to
```

`m_guessed` is tracked per window because **collisions matter**. If two positions
both predict 22, you have committed to four numbers, not five, and your chance of
matching is correspondingly lower. Scoring that against a five-number baseline
would understate the model. This is common: models with no signal converge toward
the same central value in every slot.

### No prediction is ever replaced by the truth

If a model cannot predict a window (too little history), the window is **skipped**.
An earlier version substituted the actual draw as the prediction, handing the model
free 5/5 windows that flowed straight into the significance test.

## 3. The chance baseline

`models/baseline.py`. Computed **exactly**, not simulated.

If you commit to `m` distinct numbers and the lottery draws 5 from a pool of 43,
the count you match follows a **hypergeometric distribution**:

```
matches ~ Hypergeometric(M = 43, n = m, N = 5)
mean = 5m / 43
```

| Numbers committed (`m`) | Expected matches |
| --- | --- |
| 5 | 0.581 |
| 4 | 0.465 |
| 3 | 0.349 |

For the superbalota it is simply `1/16 = 0.0625`.

### The significance test

```python
from models.baseline import beats_chance_test
result = beats_chance_test(observed_hits, m_guessed)
```

Each historical draw is an independent hypergeometric trial — the pool resets every
draw — so the **sum** of observed hits is asymptotically normal around the sum of
per-window chance means:

```
z = (Σ observed − Σ expected) / sqrt(Σ variance)
```

`m_guessed` may be a per-window list, so windows with different collision counts
each contribute their own mean and variance.

### One-sided vs two-sided: this matters

| Returned key | Question |
| --- | --- |
| `p_value` | Two-sided: "does this **differ** from chance?" |
| `p_value_greater` | One-sided: "is this **better** than chance?" |

**Only `p_value_greater` may back a "beats chance" claim.** A model significantly
*worse* than chance also gets a small two-sided p-value. Reporting that as a win
was a real bug here — the FrequencyBaseline, which underperforms chance, was being
labelled a winner.

## 4. Randomness testing

`analysis/randomness.py`. Run **before** trusting any forecast: if the series shows
no exploitable structure, a model that appears to find some is fitting noise.

| Test | Function | What a **high** p-value means |
| --- | --- | --- |
| Chi-square uniformity | `chi_square_uniformity` | Numbers appear equally often — expected for a fair draw |
| Pooled uniformity | `pooled_uniformity_test` | Same, but immune to sort-order artifacts |
| Wald–Wolfowitz runs | `runs_test` | Values sequence randomly around the median |
| Ljung–Box / ACF | `autocorrelation_check` | No autocorrelation — no "memory" to exploit |

Note the inversion relative to normal ML practice: here a **high p-value is the
healthy result**. It means the data behaves like a fair lottery.

A low Ljung–Box p-value would be the one finding that could justify a time-series
model at all. Do not expect one.

### Per-position vs pooled

```mermaid
flowchart TD
    Q{"Is the source data<br/>sorted ascending?"}
    Q -->|"Yes — is_sorted_ascending() true"| S["Per-position chi-square is<br/><b>meaningless</b> (order statistics)"]
    Q -->|"No"| N["Per-position chi-square is a<br/>valid diagnostic"]
    S --> P["Use pooled_uniformity_test<br/>— the verdict"]
    N --> P
```

**Per-position tests are diagnostics; the pooled test is the verdict.** See
[the sorted-data trap](domain-and-premise.md#4-the-sorted-data-trap).

### Multiple comparisons

Six positions are tested simultaneously. At α = 0.05 you should *expect* roughly
one in twenty tests to flag by chance. One or two red flags across six positions is
noise, not a finding. The dashboard says so on screen; any new test surface should
too.

## 5. Expected value: the one exact answer

`analysis/prizes.py`. No historical data required, no model, no uncertainty.

```python
from analysis.prizes import category_probabilities, expected_value, breakeven_jackpot

probs = category_probabilities()                       # all 12 categories, exact
ev = expected_value(probs, payouts, ticket_price=5700)
```

| Output | Meaning |
| --- | --- |
| `expected_return` | Average payout per ticket |
| `expected_value` | `expected_return − ticket_price` — the real number |
| `return_to_player` | Fraction of the stake returned on average (RTP) |
| `odds_any_prize_one_in` | Chance of winning anything |
| `breakeven_jackpot()` | Top prize that would make EV zero |

The 12 categories are every `(main matches 0–5) × (superbalota hit or not)`
combination. Main matches are hypergeometric; the superbalota is independent, so
each category is a product. Probabilities sum to exactly 1.0 and the jackpot comes
out at 1 in 15,401,568 — matching the published odds.

**Payouts are caller-supplied, never hardcoded.** Most tiers are operator-set and
the top prize accumulates, so the dashboard exposes an editable table rather than
inventing numbers.

On `breakeven_jackpot`: a positive EV on paper is not the whole story. A large
jackpot attracts more players, and a shared jackpot pays each winner less than the
calculation assumes; withholding tax cuts it further.

## 6. Forecast-accuracy metrics (Prophet cross-validation)

`Prophet.py:evaluate_model_performance()` wraps Prophet's own cross-validation.

| Metric | Meaning | Direction |
| --- | --- | --- |
| MSE | Mean squared error | lower better |
| RMSE | Root MSE, in ball units | lower better |
| MAE | Mean absolute error | lower better |
| MAPE / MDAPE / SMAPE | Percentage error variants | lower better |
| Coverage | Fraction of actuals inside the prediction interval | should ≈ the interval level |

**Read these with care in this domain.** They measure how numerically close a
prediction was — but a lottery ticket pays on an exact set match, not proximity.
A model that predicts 22 every draw scores a respectable MAE and wins nothing. A
low RMSE here is not evidence of anything useful.

Coverage is the one that is genuinely diagnostic: if you use 95% intervals and
coverage is 0.80, the intervals are too narrow and the model is understating its
own uncertainty.

> These metrics describe a *fit*. `beats_chance_test` describes whether there is
> *signal*. Only the second one answers "is this worth using".

## 7. How to read a backtest result

```
            model  n_windows  avg_main_hits  chance_avg_main_hits  p_value_better_than_chance  beats_chance
        AutoARIMA         15           0.73                  0.46                       0.081         False
          XGBoost         15           0.60                  0.50                       0.561         False
FrequencyBaseline         15           0.40                  0.58                       0.894         False
```

Reading order:

1. **`beats_chance`** — the verdict. Expect `False`.
2. **`avg_main_hits` vs `chance_avg_main_hits`** — the raw gap. AutoARIMA is above
   chance here.
3. **`p_value_better_than_chance`** — 0.081. Not significant. With 15 windows, a
   gap that size is ordinary variance.
4. **`n_windows`** — the sample size behind all of the above. Fifteen is small.
   Treat a single run at 15 windows as an anecdote.

The FrequencyBaseline row is instructive: it is *below* chance, and its one-sided
p-value of 0.894 correctly reports "no evidence this is better". Under the old
two-sided test this same row could read as a win.

## 8. Known failure modes we have already hit

Documented because they are easy to reintroduce and each one produced a
plausible-looking wrong answer rather than a crash.

| Bug | Effect | Fix |
| --- | --- | --- |
| Shuffled `train_test_split` on a time series | Future draws in training; inflated accuracy | Chronological splits everywhere |
| Ground truth as fallback prediction | Free 5/5 windows fed the significance test | Skip the window instead |
| Two-sided p-value for "beats chance" | A model *worse* than chance reported as beating it | `p_value_greater` |
| Fitted value presented as a forecast | The dashboard showed a hindcast of an already-drawn result | `forecast_next()` |
| Pooling the 1-16 superbalota into a 1-43 test | Spurious "not uniform" verdict from the sort-proof test | Range derived from positions; mixing is now inexpressible |
| Per-position chi-square on sorted data | Spurious "non-random" structure | `is_sorted_ascending` + pooled test |

---

**Next:** [Dashboard](dashboard.md) · [Development](development.md)
