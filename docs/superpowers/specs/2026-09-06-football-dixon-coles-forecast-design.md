# Football: Dixon-Coles forecast, scoring, evaluation, and a two-team dashboard view

**Date:** 2026-09-06
**Status:** design, pending implementation
**Branch:** `claude/football-dixon-coles-forecast`

## 1. Goal

Turn the football half of the repository from a data layer into a usable
forecasting tool. Concretely: a dashboard user picks two teams and gets
(a) descriptive head-to-head statistics and (b) a model's 1X2 and
most-likely-scoreline forecast, shown next to the closing-line market
probabilities and, on a holdout, a measured verdict on whether the model
actually beats that market.

Real data, not synthetic: Europe from football-data.co.uk's main contract
(already supported), Colombia from football-data.co.uk's "extra" files
(`new/COL.csv`, a different contract, opening odds only).

## 2. What this is not

- **Not Elo.** `docs/football.md §7` lists Elo as the cheap second baseline
  before Dixon-Coles. We skip it. Elo only yields 1X2; the user asked for a
  likely scoreline, which needs a goals model, and Elo would be built and
  then superseded.
- **Not a forecast shown alone.** The repo's premise is that a football
  forecast is worth nothing until it beats the closing line. Every surface
  that shows model probabilities also shows the market probabilities beside
  them and points at the measured verdict. A page that implied otherwise
  would be claiming a result that does not exist.
- **Not a change to `core/`.** `core/` stays domain-free. The football
  domain supplies the null (the market) and the scoring rule, exactly as
  the lottery supplies the hypergeometric null and set-based hit counting.
- **Not team-name normalisation.** football-data is consistent within its
  own files; cross-source name drift ("Atl Nacional" vs "Atlético
  Nacional") is a known, documented limitation, not solved here.

## 3. Domain layer (`football/`)

No new dependencies: `scipy` (with `scipy.optimize`) is already in both
`requirements.txt` and `requirements-test.txt`.

### 3.1 `football/scoring.py` — proper scoring rules

Pure functions. Input: `probs` as an `(n, 3)` array in `OUTCOMES` order
(`H, D, A`), `outcomes` as a length-`n` sequence of `"H"`/`"D"`/`"A"`.

| Function | Definition | Notes |
| --- | --- | --- |
| `brier_score(probs, outcomes)` | mean over matches of `sum((p - onehot)**2)` | 0 = perfect, 2 = worst; symmetric across outcomes |
| `ranked_probability_score(probs, outcomes)` | mean of `sum_k (cumsum(p)[k] - cumsum(onehot)[k])**2) / (K-1)` over `k = 0..K-2` | the proper score for an **ordered** 3-way result; predicting D when A occurred beats predicting H |
| `log_loss(probs, outcomes)` | mean of `-log(p[true])`, `p` clipped to `[1e-15, 1]` | unbounded; punishes confident misses hardest |
| `skill_score(model_probs, baseline_probs, outcomes, metric="rps")` | `1 - score(model) / score(baseline)` | > 0 means the model beat the baseline on that metric; the headline "vs market" number |

`metric` accepts `"brier"`, `"rps"`, `"log_loss"`. `RPS` is the default
because the outcome is ordered and RPS is the standard football forecasting
metric (Constantinou & Fenton).

The `H, D, A` order is load-bearing for RPS (it uses the cumulative
distribution). `scoring.py` imports `OUTCOMES` / `outcome_index` from
`football/common.py` and never re-declares the order.

Rows with a NaN in `probs` (e.g. a match with no market price) are dropped
pairwise inside `skill_score` so the model and baseline are always scored
on the same matches; the single-forecast functions raise on a NaN rather
than silently averaging over fewer matches.

### 3.2 `football/dixon_coles.py` — the model

The Dixon-Coles bivariate-Poisson-with-low-score-correction model.

**Parameters** (fitted by maximum likelihood):

- `attack[team]` — one per team, log-scale.
- `defence[team]` — one per team, log-scale.
- `home_advantage` (`gamma`) — one scalar, log-scale.
- `rho` — the low-score dependence parameter.

**Identifiability:** `mean(attack) == 0` enforced by fitting `n_teams - 1`
free attack params and setting the last to `-sum(rest)`, or by a penalty
term. Defence is pinned the same way. (Pick the constrained-parameter
form; it is exact and needs no tuning.)

**Expected goals** for a fixture `(i, j)`:

```
lambda_home = exp(attack[i] - defence[j] + home_advantage)
lambda_away = exp(attack[j] - defence[i])
```

**Low-score correction** `tau(h, a, lambda_home, lambda_away, rho)` applied
to the four cells `(0,0), (0,1), (1,0), (1,1)`:

```
tau(0,0) = 1 - lambda_home * lambda_away * rho
tau(0,1) = 1 + lambda_home * rho
tau(1,0) = 1 + lambda_away * rho
tau(1,1) = 1 - rho
tau      = 1   elsewhere
```

**Log-likelihood** over training matches:

```
sum over matches of  w_m * ( log tau(h_m, a_m, ...) + logpmf(h_m; lambda_home) + logpmf(a_m; lambda_away) )
```

`w_m` is the time-decay weight (§3.2.1). Maximised with
`scipy.optimize.minimize(method="L-BFGS-B")` on the negative log-likelihood.
`rho` is bounded to keep every `tau` cell positive over the plausible
scoreline range; a fit that hits the bound warns.

#### 3.2.1 Time decay

`fit(matches, half_life=None)`. `half_life` in **days**: a match `d` days
before the most recent training match gets weight `0.5 ** (d / half_life)`.
`half_life=None` means uniform weights. The dashboard exposes it with a
sensible default (180 days) so recent form is weighted up; tests cover both
`None` and a finite value.

#### 3.2.2 Prediction API

`DixonColes` instance methods, all deriving from one scoreline matrix:

- `scoreline_matrix(home_team, away_team, max_goals=10)` → `(max_goals+1,
  max_goals+1)` array, `M[h, a] = P(home scores h, away scores a)`, with
  `tau` applied and the matrix renormalised to sum to 1 (the truncated tail
  and the correction both cost a little mass).
- `predict_outcome(home_team, away_team)` → `np.array([p_home, p_draw,
  p_away])` in `OUTCOMES` order (`tril`, `diag`, `triu` of the matrix).
- `most_likely_scores(home_team, away_team, n=5)` → list of
  `((h, a), probability)` sorted descending.
- `over_under(home_team, away_team, line=2.5)` → `(p_over, p_under)`.
- `both_teams_to_score(home_team, away_team)` → `p_btts`.
- `predict_matches(matches)` → `(n, 3)` probability array for a frame of
  fixtures, for the evaluator.

**Unknown team:** any predict call with a team not seen in training raises
`UnknownTeamError` (subclass of `KeyError` or `ValueError`) naming the
team. The dashboard catches it and tells the user the team is not in the
loaded seasons.

### 3.3 `football/h2h.py` — descriptive statistics, no model

Pure functions over the tidy match frame (`MATCH_COLUMNS` + odds).

- `team_form(matches, team, last_n=5, as_of=None)` → dict / small frame:
  the last `last_n` matches involving `team` strictly before `as_of` (or
  the end of the frame), each as `W`/`D`/`L` from `team`'s perspective,
  plus goals for / against, points, and the home/away split.
- `head_to_head(matches, home_team, away_team, as_of=None)` → all past
  meetings between the two (either venue), the win/draw/win record, mean
  goals, and the last few results.

`as_of` exists so the dashboard's forecast view can show "form going into
this fixture" without leaking matches after a chosen date; tests pin that
it excludes `>= as_of`.

### 3.4 `football/extra_processor.py` — the `new/COL.csv` contract

football-data's "extra" files use a different, stable contract:

```
Country, League, Season, Date, Time, Home, Away, HG, AG, Res,
PH, PD, PA, MaxH, MaxD, MaxA, AvgH, AvgD, AvgA, ...
```

Several leagues and seasons stacked in one file, `Home`/`Away`/`HG`/`AG`
instead of `HomeTeam`/`FTHG`, and **opening odds only** (`AvgH/D/A` market
average, `PH/D/A` Pinnacle, sometimes `B365H/D/A`).

`preprocess_extra(df, league=None)`:

- Requires `("Date", "Home", "Away", "HG", "AG")`; raises `MatchFormatError`
  (reused from `processor.py`) otherwise.
- If `league` is given, filters to `df["League"] == league` first; if the
  file carries more than one league and `league` is None, raises (the same
  posture as `load_seasons` refusing to merge odds sources — don't silently
  stack Primera A and the Colombian second division).
- Maps to `MATCH_COLUMNS` + `ODDS_COLUMNS` exactly as `preprocess_matches`
  does: `outcome_from_goals`, day-first date parse (reuse `_parse_dates`),
  drop unplayed rows.
- Resolves odds from an **opening-only** source list:
  `("extra_market_average_opening", ("AvgH","AvgD","AvgA"))`,
  `("extra_pinnacle_opening", ("PH","PD","PA"))`,
  `("extra_bet365_opening", ("B365H","B365D","B365A"))`.
  Sets `attrs["odds_source"]` to that name and
  `attrs["odds_are_closing"] = False` unconditionally.
- Same "usable price triple or NaN the whole row" rule, same `<= 1.0`
  guard.
- Runs `check_match_format` (from `processor.py`) which will warn that the
  baseline is soft — that warning is correct and must reach the user.

`load_extra(path, league=None, validate=True)` — read one CSV, no
concatenation (the file already spans seasons).

### 3.5 `football/downloader.py` — `--extra`

- New CLI flag `--extra`. When set, `--leagues` accepts the extra codes
  (start with `COL`; the mechanism is general). Fetches
  `https://www.football-data.co.uk/new/{CODE}.csv`, validates the response
  through `preprocess_extra`, and writes it to
  `exported_data/football/{CODE}.csv`.
- Without `--extra`, the existing refuse-by-name behaviour for `ARG`, `BRA`,
  `MEX`, `COL`, ... is unchanged — the message gains a line pointing at
  `--extra`.
- Same posture as today: HTML-sniff before parse, never replace a longer
  file with a shorter one without `--force`, print the resolved odds source
  per file (it will say "opening").
- The dry-run summary states plainly that extra files are opening-odds-only
  and cannot be evaluated against a sharp baseline.

### 3.6 `football/evaluation.py` — the domain half of the `core/` contract

The football analogue of `lottery/models/baseline.py:beats_chance_test`.

`beats_market_test(model_probs, market_probs, outcomes, metric="rps", alpha=0.05, n_comparisons=1)`:

1. Drop rows where either `model_probs` or `market_probs` has a NaN.
2. Per match, compute the model's and the market's per-match score
   contribution for `metric` (for RPS/Brier that is the per-match term
   before the mean; for log-loss the per-match `-log p[true]`).
3. Form the paired difference `d_m = market_score_m - model_score_m`
   (positive → model better on match `m`).
4. Call `core.significance.z_test_against_null(observed=d, null_means=0.0,
   null_variances=var(d, ddof=1))` — a paired test of "mean improvement > 0".
5. Return the full z-test dict plus `verdicts(p_value_greater, alpha,
   bonferroni_threshold(alpha, n_comparisons))` → `beats_market` and
   `beats_market_corrected`.

One-sided (`p_value_greater`) only, per `core/significance.py`. Effect size
and CI pass straight through.

`n_comparisons` is a parameter because the backtest may score several
de-margining methods (multiplicative / additive / power) as separate
"models" against the same matches; when it does, `k = 3` and the corrected
verdict tightens.

### 3.7 `football/backtest.py` — walk-forward evaluation

Mirrors `lottery/backtest.py` structure and CLI shape.

- `run_all(matches, n_windows, min_train, half_life=None, method="multiplicative")`:
  walk-forward via `core.windows.window_bounds`. For each held-out match:
  fit `DixonColes` on the training slice, `predict_matches` the held-out
  fixture, compute market probs on it via `football/market.py` with
  `method`. Accumulate `(model_probs, market_probs, outcome)` rows, then
  `beats_market_test` once over all of them.
- `run_holdout(matches, cutoff, mode, method=...)`: hold out everything
  after `cutoff`. `mode="expanding"` refits per held-out match;
  `mode="frozen"` fits once at the cutoff and predicts all of them. Both
  score through the same path, so the summaries are comparable — same
  contract as the lottery.
- Reports: model Brier/RPS/log-loss, market Brier/RPS/log-loss, skill
  score, the z-test dict, both verdicts, `n_observations`.
- Fitting DC per window is the slow part. Default `--n-windows` is modest
  (e.g. 30). `run_holdout(..., mode="frozen")` is the fast, concrete
  alternative. The pytest coverage of the walk-forward path is
  `@pytest.mark.slow`.
- CLI: `python -m football.backtest --seasons E0_2324.csv,E0_2223.csv
  --n-windows 30 [--half-life 180] [--method power]` and
  `--cutoff 2024-03-01 --mode frozen`.

## 4. Dashboard (`dashboard/football_page.py`, `dashboard/ui.py`)

Tabs go from three to four: **Datos · Mercado · Pronóstico · Resultados**.

The sidebar gains a data-source toggle: "Europa (football-data)" vs
"Colombia (archivo extra)". Colombia mode loads via `load_extra` with a
league selector; the existing opening-odds warning banner covers it with no
new copy needed beyond naming the file.

### 4.1 Pestaña Pronóstico (new)

- Two `st.selectbox`, home / away, options = sorted unique teams in the
  loaded frame. `st.number_input` for `half_life` (default 180, 0 → None).
- An `st.form` with three **optional** `st.number_input` fields for current
  decimal odds (home / draw / away). Submit button "Calcular pronóstico".
- On submit:
  - **H2H section:** `head_to_head` table + `team_form` for both teams,
    with `as_of` = today (all loaded data is historical, so this is just
    "most recent form").
  - **Model section:** `DixonColes` fitted on the loaded frame, cached with
    `@st.cache_resource` keyed by a hash of `(paths, half_life)`. Shows:
    - 1X2 as three `st.metric` + a bar chart.
    - Scoreline heatmap (`go.Heatmap` over the `scoreline_matrix`, capped at
      `max_goals=6` for legibility) with the modal score annotated.
    - `most_likely_scores(5)` as a small table.
    - Over/Under 2.5 and BTTS as metrics.
  - **Market section:**
    - If the three odds fields are filled: de-margin with the sidebar
      method, show model vs market as a grouped bar + a table of the
      per-outcome difference. Caption: this is **one match**, an
      **uncorrected** comparison — the measured verdict is in *Resultados*.
    - If not: model only, with the standard "sin línea base para este
      partido" caption.
  - `UnknownTeamError` → `st.error` naming the team and the loaded seasons.

### 4.2 Pestaña Resultados (extended)

Keeps the existing outcome-share and observed-vs-market content. Adds:

- Section **"¿Le gana este modelo al mercado?"**, behind a button (the
  backtest is slow), cached with `@st.cache_data`:
  - Runs `football.backtest.run_all` (or `run_holdout` with a date picker)
    on the loaded seasons.
  - Shows model vs market Brier/RPS/log-loss, skill score, effect size +
    CI, `n_observations`, and `beats_market` / `beats_market_corrected`
    with the corrected column emphasised.
  - Only rendered when there is an odds source and enough matches
    (`>= min_train + n_windows`); otherwise a message says why.
  - For a Colombia (opening-odds) frame, an explicit banner: any edge here
    is against a soft baseline and is not evidence of a real edge.
- Model calibration curve (model `p_home` vs observed home-win frequency)
  beside the market calibration curve that already exists.

### 4.3 `dashboard/ui.py`

One `HELP` entry per new section and chart, keys prefixed `fb_`:
`fb_forecast_tab`, `fb_h2h`, `fb_form`, `fb_model_1x2`, `fb_scoreline_grid`,
`fb_most_likely_scores`, `fb_over_under`, `fb_btts`, `fb_your_odds`,
`fb_model_vs_market`, `fb_eval_tab`, `fb_skill_score`, `fb_beats_market`,
`fb_model_calibration`, `fb_half_life`, `fb_source_toggle`.

Every chart goes through `chart(fig, title, key)` and every subheader
through `section(title, key)` — no bare `st.plotly_chart`. Each help string
states what the chart does **not** mean (the scoreline grid is not a
prediction of the exact score; the one-match model-vs-market bar is not a
verdict; a positive skill score on one season is within noise; etc.).

All user-facing strings in Spanish.

## 5. Tests (`tests/`, mirroring the module layout)

- `test_football_scoring.py`: perfect forecast → 0 on all three metrics;
  uniform forecast → known constants; RPS ordering (mass on D when A
  occurred beats mass on H when A occurred, Brier does not distinguish
  them); `skill_score` sign and pairwise-NaN dropping.
- `test_football_dixon_coles.py`: fit on `load_sample_and_preprocess`
  (known `p_true_*`); `scoreline_matrix` sums to ~1; mean `predict_outcome`
  close to mean `p_true_*` within tolerance; `rho` raises the draw
  probability relative to an independent-Poisson baseline on low-scoring
  fixtures; `sum(attack) ≈ 0`; unknown team raises `UnknownTeamError`;
  `half_life` finite vs None both fit and differ. Heavy fits →
  `@pytest.mark.slow`.
- `test_football_h2h.py`: form W/D/L counts from a hand-built frame; h2h
  record consistency when home/away are swapped; `as_of` excludes matches
  on or after the cutoff.
- `test_football_extra_processor.py`: a saved-style `new/COL.csv` fixture
  (checked into `tests/fixtures/`) → correct column mapping,
  `odds_are_closing is False`, opening source name, league filter, day-first
  date parse; a multi-league file with `league=None` raises; a file with
  only closing-style columns still resolves to an opening source name (the
  extra contract has no closing odds by definition).
- `test_football_evaluation.py`: model probs == market probs → effect ≈ 0,
  not significant; a model strictly better on every match → `beats_market`
  true; both verdict keys always present; `n_comparisons=3` tightens the
  corrected threshold.
- `test_football_backtest.py` (`@pytest.mark.slow`): synthetic data with
  `market_noise` high → model beats market, `beats_market` fires; with
  `market_noise=0` → model does **not** beat market (the control); `frozen`
  and `expanding` summaries have the same keys; effect size + CI present.

New test deps: none (`scipy` already in `requirements-test.txt`). If any
test ends up importing something not there, it is added to **both**
requirements files per `CLAUDE.md`.

## 6. Documentation to update (the `CLAUDE.md` doc-map discipline)

| File | Change |
| --- | --- |
| `docs/football.md` | §5: extra files are now reachable with `--extra`, opening-odds-only, one league per load. §7: scoring rules, Dixon-Coles, evaluation and the forecast tab are built; Elo is still deferred and why. New section on the model and the two-team view. |
| `docs/data-pipeline.md` | The extra-file contract and how it resolves odds. |
| `docs/models.md` | Dixon-Coles: parameters, the low-score correction, time decay, how to add another football model against the same scoring rule. |
| `docs/evaluation.md` | `football/backtest.py` and `beats_market_test` beside the lottery backtest; the paired-difference framing. |
| `docs/dashboard.md` §3 | The Pronóstico tab, the source toggle, the evaluation section. |
| `CLAUDE.md` | Module-layout entries for the six new modules. Replace "Football is currently a data layer only … No models yet" and the matching dashboard bullet. New invariants: the scoring rule is proper and treats the outcome as ordered (RPS); model probabilities are never shown without the market beside them and the measured verdict linked; extra files are opening-odds-only and cannot support a corrected edge claim; `football/evaluation.py` is a paired test against the market with `null_mean = 0`. |

`python -m lottery.utils.check_docs` after the doc edits (it checks every
`docs/` link, football pages included).

## 7. Build order (TDD each step)

1. `football/scoring.py` + `tests/test_football_scoring.py`.
2. `football/h2h.py` + `tests/test_football_h2h.py`.
3. `football/dixon_coles.py` + `tests/test_football_dixon_coles.py`.
4. `football/extra_processor.py` + `--extra` in `downloader.py` +
   `tests/test_football_extra_processor.py` + `docs/data-pipeline.md`.
   **Separable:** if Europe-first is preferred, this step can ship as a
   follow-up; steps 5–7 work on the main contract alone.
5. `football/evaluation.py` + `football/backtest.py` + their tests.
6. Dashboard: source toggle, Pronóstico tab, Resultados evaluation section,
   `dashboard/ui.py` HELP keys.
7. Documentation sweep + `CLAUDE.md` + verification.

## 8. Verification (from `CLAUDE.md`)

- `pytest` and `pytest -m "not slow"`.
- `python -m py_compile` on `dashboard/app.py`, `dashboard/football_page.py`
  and any new `scripts/` file.
- `python -m lottery.utils.check_docs`.
- Streamlit headless + Playwright: load the page, select the Fútbol domain,
  open each tab, drive the Pronóstico form with a real pair of teams, and
  check tab text for `Traceback` / "This app has encountered an error".
  Scope locators to `get_by_role("tabpanel", name=...)`.
- After the first real `--extra` download, eyeball one Colombia file the
  way `docs/football.md §5` prescribes for European files.

## 9. Open risks

- **DC fit speed in the walk-forward.** ~40 params, one L-BFGS-B per
  window. Mitigated by a modest default `--n-windows`, the `frozen` mode,
  and warm-starting each window's optimiser from the previous window's
  solution (optional; add only if the default window count is still slow).
- **Colombia odds coverage.** The extra file's `Avg`/`P` columns are sparse
  before ~2018 and the league name string may vary. `check_match_format`
  already warns on low coverage; the league selector is populated from the
  file's own `League` values so a renamed league is visible rather than
  silently empty.
- **Newly promoted teams.** No prior at all → `UnknownTeamError`. Acceptable
  for now; a prior-from-league-average fallback is a future enhancement,
  noted in `docs/models.md`.
