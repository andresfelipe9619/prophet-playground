# Platform Roadmap: from an honest backtest to a defensible platform

> **For agentic workers:** this is the index plan. Each numbered item is a self-contained
> piece of work with its own task list. Work them **in order** — the ordering is by
> ascending effort *and* by dependency, and several later items assume an earlier one
> has landed. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** close the gap between what this repository currently proves (a careful,
well-tested backtest harness over three domains) and what a platform anyone could rely
on in real life would have to prove (calibrated probabilities, a forward record, a
metric that converges inside one season, and results reproducible a year from now).

**The premise does not change.** Nothing here tries to make Baloto beatable; the lottery
domain is finished and every item below either leaves it alone or generalises machinery
*out* of it for the other two. Items 9-11 add model complexity, which is the direction
[`docs/domain-and-premise.md`](../../domain-and-premise.md#6-anti-patterns) warns about —
they are last on purpose and each is gated behind an evaluation surface that already
exists.

**Architecture:** unchanged. `core/` stays domain-free; new shared machinery
(`core/registry.py`, `core/manifest.py`) takes the null and the scoring rule from its
caller exactly as `core/significance.py` does. Everything that mentions a team, a match,
a price, a rider or a ball lands in a domain package.

---

## Global constraints

Copied from `CLAUDE.md` and binding on every item below.

- **`core/` stays domain-free.** No ball, draw, pool, team, match, price, rider or stage.
- **Chance/market/baseline comparisons are one-sided.** Only `p_value_greater` may back a
  "beats X" claim.
- **Every evaluation surface reports the naive and the corrected verdict together** and
  points the reader at the corrected one.
- **Model probabilities are never shown without their baseline beside them** — chance for
  Baloto, the de-margined closing price for football, the pre-race ranking for cycling.
- **A null result needs its resolution attached.** Any new "nothing beat X" surface says
  what it could have detected. This is why item 7 exists.
- **Outcome order is `("H", "D", "A")`**, from `football/common.py:OUTCOMES`.
- **Opening and closing odds never mix**; one odds source per frame or none.
- **Language:** English for code, docstrings, comments, docs and commits. Spanish only
  for user-facing dashboard strings.
- **New test dependency ⇒ add it to BOTH `requirements.txt` and `requirements-test.txt`.**
- **Charts go through `dashboard/ui.py:chart(fig, title, key)` / `section(title, key)`**,
  with all four copy dicts (`HELP`, `PLAIN`, `READ`, `GLOSSARY`) keyed identically.
- **Tests are deterministic**, built from the seeded sample-data generators. Runs that fit
  real models are marked `@pytest.mark.slow`.
- Commit after every green test cycle.

---

## The order, and why

| # | Item | Effort | Why here |
| --- | --- | --- | --- |
| 1 | Lint and type-check | XS | No dependencies, catches defects in every later item. |
| 2 | Packaging and the run manifest | S | Later items produce results; results need provenance. |
| 3 | Leak canaries | S | Proves the pipelines the later items build on do not leak. |
| 4 | `football/calibration.py` | M | Makes every probability downstream trustworthy. |
| 5 | `football/clv.py` | M | The metric that converges inside one season. |
| 6 | `core/registry.py` | M | Generalises the forward record to football and cycling. |
| 7 | Football power and sensitivity | M | Gives football the "could I have detected it?" half. |
| 8 | `football/bankroll.py` | M | Turns an edge into something a person can actually run. |
| 9 | Parameter uncertainty | L | Needs 4 to be readable and 3 to be trusted. |
| 10 | Cycling's market baseline and rider features | L | The domain's stated missing half. |
| 11 | Football shot-level data and features | XL | Gated behind 4, 5, 7 — the anti-pattern risk. |
| 12 | Pipeline automation and a storage layer | L | Ops; wants 2 and 6 in place first. |
| 13 | The dashboard bet/ticket log | M | Pure product work on top of 6. |

---

## Item 1: lint and type-check

**Rationale.** `CLAUDE.md` notes "there is still no linter". Cheapest item on the list and
it pays into all twelve others.

**Files:** `pyproject.toml` (new, shared with item 2), `requirements-test.txt`,
`.github/workflows/tests.yml`, plus whatever the first clean run turns up.

- [x] Add `ruff` config to `pyproject.toml`: line length 100, rules `E,F,I,UP,B`, and an
      explicit per-file ignore for `dashboard/` where Streamlit's import-then-call order
      is load-bearing.
- [x] Add `mypy` config scoped to `core/` only, `strict = true`. `core/` is 200 lines of
      pure numerics and is the one place a type error is silently wrong rather than loud.
- [x] `ruff check --fix` and hand-fix the rest. **No behaviour changes** — if a lint rule
      wants a real change, note it and leave it.
- [x] Add `ruff` and `mypy` to `requirements-test.txt` (and `requirements.txt`).
- [ ] Add a `lint` job to the `static` workflow — **still open**: the snippet is written and
      documented in `docs/development.md`, but pushing a workflow file needs GitHub's `workflow`
      scope, which the session that wrote it did not have. Everything else in this item is done — it needs no third-party install beyond
      the two tools, so it reports in seconds like the rest of that job.
- [x] `pytest` green, docs link check green.

## Item 2: packaging and the run manifest

**Rationale.** Every dependency is pinned `>=`, there is no `pyproject.toml`, and a
backtest result carries nothing that says what produced it. A result you cannot reproduce
in twelve months is not evidence, which is the standard this repository holds itself to
everywhere else.

**Files:** `pyproject.toml`, `core/manifest.py` (new), `tests/test_core_manifest.py` (new),
`lottery/backtest.py`, `football/backtest.py`, `cycling/evaluation.py`, `docs/development.md`.

- [x] `pyproject.toml`: project metadata, `requires-python = ">=3.11"`, dependencies moved
      off the two requirements files' `>=` pins into a single source of truth. Keep the
      requirements files as generated artefacts with a header saying so.
- [x] `core/manifest.py:run_manifest()` — returns git SHA (and a dirty flag), UTC
      timestamp, Python version, the versions of the libraries that affect numerics
      (numpy, pandas, scipy, statsmodels), and a caller-supplied `inputs` dict for the
      data hash and seeds. Domain-free: the caller names its own inputs.
- [x] `core/manifest.py:data_fingerprint(df)` — a stable hash over a frame's values and
      column order, so "the same data" is checkable rather than assumed.
- [x] Every backtest `run_*` attaches the manifest to its result's `attrs`.
- [x] Tests: manifest is JSON-serialisable; fingerprint is stable under re-read and
      changes under a single edited cell; a dirty tree is flagged.
- [x] Document the reproducibility contract in `docs/development.md`.

## Item 3: leak canaries

**Rationale.** The repository has conventions against leakage (chronological splits,
`as_of` cutoffs, `strictly before`) but no test that *proves* a pipeline honours them.
`lottery/analysis/sensitivity.py` is the model to follow: plant the fault, measure that
the detector fires.

**Files:** `tests/test_leakage.py` (new), `docs/evaluation.md`.

- [x] **Shuffled-target canary (football).** Permute outcomes in time, refit Dixon-Coles
      walk-forward, assert `beats_market_corrected` is False and the effect's interval
      straddles zero. A pipeline that leaks still "beats" the market on shuffled targets.
- [x] **Shuffled-target canary (cycling).** The same against `beats_baseline_test`.
- [x] **`as_of` canary.** Assert `form_worths(results, riders, as_of=d)` and
      `team_form(..., as_of=d)` are bit-identical when every row on or after `d` is
      deleted. This is the strongest available statement that the cutoff is real.
- [x] **Future-row canary.** Append an absurd future result to the frame and assert every
      `as_of`-taking function's output is unchanged.
- [x] Mark the refitting ones `slow`; document them in `docs/evaluation.md` beside the
      known-past-bugs section.
- [x] **Positive controls**, added while building it: a canary that cannot fire is decoration,
      so a forecaster that reads the race it forecasts must beat the ranking and a forecast
      that has seen the result must beat the market.

## Item 4: `football/calibration.py`

**Rationale.** `grep -ri calibrat` finds prose, not code. RPS is a lumped score: it cannot
separate a model that ranks matches badly from one that ranks them well and is
overconfident — and Dixon-Coles on thin data is reliably the latter. Calibration is also
what makes `value.py` honest: Kelly on uncalibrated probabilities sizes up exactly when
the model is most confidently wrong, which is what the quarter-stake default is already
hedging against.

**Files:** `football/calibration.py` (new), `tests/test_football_calibration.py` (new),
`football/backtest.py`, `dashboard/football_page.py`, `dashboard/ui.py`, `docs/football.md`,
`docs/evaluation.md`.

- [x] `reliability_curve(probs, outcomes, bins=10)` — binned forecast vs realised
      frequency, per outcome and pooled, with a count per bin. Sparse bins return NaN
      rather than a point, the same rule `structure.py:goodness_of_fit` already follows.
- [x] `expected_calibration_error` and `calibration_in_the_large` (mean forecast minus
      base rate, with its interval through `core/significance.py`).
- [x] `TemperatureScaler` — one parameter, fitted by MLE on the log-odds. The minimal
      recalibrator, and the one that cannot overfit.
- [x] `IsotonicCalibrator` — per-outcome isotonic regression with renormalisation.
- [x] **Fitted walk-forward, never in-sample.** A `calibrate=` option on
      `football/backtest.py` fits the recalibrator on the training window only and applies
      it to the held-out window. Fitting it on the matches you then score is the leak this
      package exists around, and `ensemble.py` already refuses the same temptation.
- [x] Tests: a perfectly calibrated forecast has ECE ≈ 0 and temperature ≈ 1; a
      deliberately over-confident forecast is pulled toward the base rate and its RPS
      improves; the market's own de-margined vector is near-calibrated on sample data;
      calibrating in-sample vs walk-forward gives different answers (pins the gate).
- [x] Dashboard: reliability diagram in **¿Le gana al mercado?**, with `PLAIN` copy whose
      third field says that a calibrated model is not a profitable one.

## Item 5: `football/clv.py`

**Rationale.** The repository measures "does the model beat the closing price on RPS" and
never "did the price move toward me after I bet". Closing line value is the industry's
leading indicator because it converges vastly faster than realised P&L — a few hundred
bets instead of a few thousand. Both ends of the line are **already in the data
contract**: `extra_processor.py` carries opening odds, `processor.py` carries closing.

**Files:** `football/clv.py` (new), `tests/test_football_clv.py` (new),
`dashboard/football_page.py`, `dashboard/ui.py`, `docs/football.md`, `docs/evaluation.md`.

- [x] `clv(bet_odds, closing_odds)` — the de-margined probability shift, not the raw price
      ratio. Raw-price CLV credits you for the bookmaker's margin changing.
- [x] `clv_table(bets, closing)` — per-bet CLV, hit rate (share of bets with positive
      CLV), and the mean shift with its interval.
- [x] `beats_closing_test` — one-sided, through `core/significance.py`, emitting
      `beats_closing` / `beats_closing_corrected`. Same shape as `beats_market_test`.
- [x] **The opening/closing guard applies here too.** CLV needs an opening-priced frame
      and a closing-priced frame for the *same* matches; the join is explicit and raises
      rather than silently producing a one-sided result. A CLV computed against the same
      prices you bet is identically zero, and a test pins that endpoint the way
      `ensemble.py`'s weight-0 endpoint is pinned.
- [x] Note in the docs that Colombian extra files are opening-only, so CLV there needs a
      closing source this project does not yet have.
- [x] **Found while building it:** a European file from 2019/20 on carries *both* column
      families, so CLV is computable from one file after all — `paired_prices` runs the
      contract twice rather than around it. `sample_data.generate_matches` gained
      `opening_noise`, without which none of this is testable.
- [x] Dashboard: CLV beside realised results in **Valor**, never instead of the market
      verdict.

## Item 6: `core/registry.py`

**Rationale.** `lottery/analysis/registry.py` is the most serious module in the
repository, and it serves only the domain where everyone already knows the answer is no.
Football and cycling — where a positive result would mean something — have no forward
record at all. The difference between a backtest tool and a platform is a track record
accumulated in public.

**Files:** `core/registry.py` (new), `lottery/analysis/registry.py` (rewritten onto it),
`football/registry.py` (new), `cycling/registry.py` (new), `tests/test_core_registry.py`
(new), `tests/test_registry.py`, `docs/registry.md`.

- [x] Lift the domain-free half into `core/registry.py`: append-only storage, the refusal
      to record against an event that has already happened, the refusal to record twice
      for the same (event, label), and score-all-or-none. **All three refusals are
      load-bearing and none may soften into a warning.**
- [x] The domain supplies: how to name an event, how to score a prediction against its
      result, and what to report alongside (the lottery attaches its minimum detectable
      effect; football should attach its own once item 7 lands).
- [x] `lottery/analysis/registry.py` keeps its public API and its file, now as a thin
      domain adapter. Its existing tests must pass **unchanged** — that is the refactor's
      proof.
- [x] `football/registry.py`: record a probability vector for a future fixture, score it
      by RPS against the de-margined closing price on the same match, so a registered
      football prediction is a forward market test rather than a forward accuracy number.
- [x] `cycling/registry.py`: record predicted worths for a future race's start list, score
      by the Plackett-Luce log score against the ranking baseline.
- [x] Each domain's file is committed, not gitignored, for the reason the lottery's is.

## Item 7: football power and sensitivity

**Rationale.** `power.py` and `sensitivity.py` exist only for Baloto. Without them,
"the model did not beat the market" and "this backtest could not have detected it if it
had" are indistinguishable — the exact ambiguity those two modules were written to kill,
now reproduced in the domain where it matters more.

**Files:** `football/power.py` (new), `football/sensitivity.py` (new),
`tests/test_football_power.py` (new), `tests/test_football_sensitivity.py` (new),
`dashboard/football_page.py`, `docs/power-and-sensitivity.md`, `docs/football.md`.

- [x] `minimum_detectable_edge(n_matches, metric="rps", alpha=0.05, power=0.8)` — mirrors
      `beats_market_test` exactly, the way `lottery/analysis/power.py` mirrors
      `beats_chance_test`. The per-match RPS difference's variance is estimated from data
      rather than assumed, because unlike the hypergeometric case there is no exact form.
- [x] `required_matches(edge)` and a power curve.
- [x] `football/sensitivity.py`: plant a known edge by blending the generative truth from
      `football/sample_data.py` into the model's forecast at a known weight, then measure
      how often `beats_market_corrected` fires. **`strength = 0` is the control** and must
      sit near alpha.
- [x] **Independent seeds.** The match generator and any sampling in the detector take
      their seeds from separate streams, for the reason documented in
      `lottery/analysis/sensitivity.py` — one shared `default_rng` manufactured a 17.5%
      false-positive rate there and cost a full investigation.
- [x] Every football "did not beat the market" surface gains the resolution line.

## Item 8: `football/bankroll.py`

**Rationale.** `value.py` gives a per-outcome Kelly fraction and stops. A 3% edge with a
40% chance of a 50% drawdown is not something a person can actually run, and nothing on
screen currently says so.

**Files:** `football/bankroll.py` (new), `tests/test_football_bankroll.py` (new),
`dashboard/football_page.py`, `dashboard/ui.py`, `docs/football.md`.

- [x] `simulate_bankroll(bets, stake_fraction, n_paths, seed)` — bankroll paths over a
      realised or bootstrapped bet sequence.
- [x] `drawdown_distribution`, `risk_of_ruin`, and a bootstrapped ROI interval.
- [x] **Show the zero-edge path beside every simulation.** A bankroll chart of a model
      with no measured edge is the football twin of presenting a lottery hindcast as a
      prediction, and the only thing that stops it reading as a promise is the null path
      drawn next to it.
- [x] Tests: the null bleeds (**not** as this plan first assumed — a zero-edge bettor makes no
      bets at all, so the null had to be redefined as the same bets in a world without the edge);
      full Kelly's drawdown distribution dominates the quarter's; risk of ruin rises with
      stake fraction.
- [x] Dashboard: gated behind the measured verdict, exactly as the staking surface
      already is.

## Item 9: parameter uncertainty

**Rationale.** `DixonColes` and `PlackettLuce` are MLE point estimates with no standard
errors. Six matches into a season the point estimate is confidently wrong and nothing on
any surface signals it. This also feeds item 8: a stake on a probability whose interval
spans break-even is not a bet, it is noise.

**Files:** `football/dixon_coles.py`, `cycling/plackett_luce.py`, `football/uncertainty.py`
(new), `tests/test_football_uncertainty.py` (new), dashboard pages, `docs/models.md`.

- [x] Bootstrap band on predicted probabilities (resample matches within the training
      window, refit, take the quantiles). Cheaper and more honest than a Hessian on a
      likelihood with a bounded `rho`.
- [x] Propagate the band to `value.py:classify` — a `value` classification whose band
      crosses the break-even probability is downgraded to `disagreement_only`.
- [x] Cycling: the same for rider worths, where the Gamma prior's shrinkage already makes
      the thin-data case visible and the band makes it quantitative.
- [x] Tests: the band narrows as the training window grows. **Not** "the point sits inside its
      own band" — measured, it often does not, and forcing that would have been papering over
      a real property of a shrunk ratio-scale estimator rather than pinning one.
- [x] Dashboard: bands on the forecast charts, never a bare point.

## Item 10: cycling's market baseline and rider features

**Rationale.** `CLAUDE.md` states it outright: "the market baseline is still missing, so
every verdict is against the ranking and says so." The ranking is the soft bar, and the
`uniform_worths` joke only lands once a hard bar exists. Separately, `cycling/sample_data.py`
models terrain while the real model does not — the synthetic tests are currently harder
than the model they test.

**Files:** `cycling/market.py` (new), `cycling/features.py` (new), `cycling/scraper.py`,
`cycling/plackett_luce.py`, `tests/`, `dashboard/cycling_page.py`, `docs/cycling.md`.

- [x] `cycling/market.py` — outright prices to de-margined worths. Cycling's book has
      ~180 runners and an overround far above football's; `football/market.py`'s three
      normalisations disagree most exactly here, so `compare_methods` is not optional.
- [x] The contract for a price source (`cycling/prices.py`) — the source itself does not exist and the docs say so, with the same "one source per frame" guard
      the other two domains enforce.
- [x] `cycling/features.py` — parcours/terrain, rider specialisation, team strength (excluding
      the rider it describes), days of accumulated fatigue. Specialisation shipped as **two**
      classes, climb and sprint, not four: the label is inferred from how a race finished, and a
      bunch share separates mountain from flat cleanly while nothing in a result distinguishes a
      rouleur from a time triallist. Four classes would need a roadbook, which is the same
      missing input as everything else here.
- [x] Terrain-conditional worths in `plackett_luce.py` (`TerrainPlackettLuce`), gated behind
      `beats_baseline_test` — against the unconditional fit, since no price source exists to
      make the market the bar in practice. Measured with and without planted specialists.
- [x] The uniform draw stays in every comparison table, for the reason it always has.

## Item 11: football shot-level data and features

**Rationale.** Goals plus odds is the 2005 feature set, and Dixon-Coles has a known
ceiling on it. Attack/defence strengths converge roughly three times faster on xG than on
goals, and the model currently has no memory across seasons — a promoted side starts from
nothing every August.

**This item is last and gated.** It is the one that most resembles the anti-pattern this
repository was refactored away from. Do not start it until items 4, 5 and 7 are green, so
that every addition is measured against the closing line, in CLV, with a stated minimum
detectable effect.

**Files:** a shot-level source module, `football/features.py` (new),
`football/dixon_coles.py`, tests, `docs/football.md`, `docs/data-pipeline.md`.

- [ ] A shot-level / xG source with its own contract module and download-time validation,
      following `football/downloader.py`'s shape (sniff for an HTML error page served 200,
      refuse to shorten a file without `--force`).
- [ ] `football/features.py` — rest days, travel, red-card timing, congestion.
- [ ] xG-based attack/defence in `DixonColes`, as an **option**, scored against the
      goals-based fit over the same held-out matches through `compare_models`.
- [ ] Season-to-season strength carry-over with shrinkage, and a promoted-team prior.
- [ ] **Each feature lands only if it clears the corrected market test.** A feature that
      improves the in-sample fit and not the verdict is reverted, and the docs say which
      ones were.

## Item 12: pipeline automation and a storage layer

**Rationale.** Everything is a manual scraper writing a gitignored CSV. `final-final.csv`
is a single point of failure for a project whose entire thesis is data discipline.

**Files:** `.github/workflows/refresh.yml` (new), a storage module, the three scrapers,
`docs/data-pipeline.md`.

- [ ] Parquet or SQLite with an explicit schema version, behind the existing
      `load_and_preprocess` API so nothing downstream changes.
- [ ] A scheduled job: scrape → validate against the contract → append → re-score the
      registries (item 6) → fail loudly on a contract violation. **A scraper that writes
      an empty file when the markup changes is the failure mode these modules are already
      shaped against** — the schedule must not reintroduce it by swallowing the raise.
- [ ] Alert on: contract violation, a source gone silent, a registry row now scoreable.

## Item 13: the dashboard bet/ticket log

**Rationale.** The dashboard is stateless — nothing remembers a ticket you generated or a
bet you would have placed. For real-life use this is the product gap, and on top of item 6
it is mostly UI.

**Files:** `dashboard/` pages, the registry modules, `docs/dashboard.md`.

- [ ] Record a generated ticket or a classified bet straight into its domain registry from
      the dashboard, with the same refusals enforced.
- [ ] A log view: what was recorded, what has been scored, realised CLV and P&L, and the
      minimum detectable effect for the number of rows so far.
- [ ] **The log's headline is the corrected verdict, not the running total.** A P&L figure
      at the top of a betting log is the single most misleading number this project could
      put on a screen.
