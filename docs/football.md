# Football

The second domain. Read [Domain and Premise](domain-and-premise.md) first —
this page is mostly about how football differs from it, and the differences
run deeper than the data.

## 1. The premise, inverted

| | Baloto | Football |
| --- | --- | --- |
| The process | i.i.d. uniform **by design** | real, persistent signal |
| Can a model beat the baseline? | No. Provably not. | Yes — that is the point |
| The baseline | exact hypergeometric, computed from the rules | the **closing betting line**, estimated from prices |
| A good result | the honest null | a measured, corrected, out-of-sample edge |
| What failure looks like | a system sold on a lucky backtest | a model that beats Elo and loses to the market |

The lottery half of this repository exists to show that nothing works. The
football half can show that something does — which makes it *more* dangerous,
not less, because now a positive result is possible and therefore worth
faking to yourself.

Everything in `core/` carries over unchanged: walk-forward splits, the
one-sided test, the Bonferroni correction, the effect size and its interval.
What changes is the null the domain hands it, and the scoring rule.

## 2. The baseline is the closing line

Not "50/50". Not "always back the home team". Not an Elo rating.

The market price at kick-off aggregates every public model, every injury
report, every professional bettor and everyone with an opinion and money. It
is the strongest freely available forecast of a football match that exists.
Beating it consistently, after the margin, is the entire definition of an
edge — and a model that beats a weaker baseline has demonstrated nothing that
anyone would pay for.

This is the football analogue of "no model can beat chance": a claim that
sounds pessimistic, is true, and is the reason the code is shaped the way it
is.

### Odds are not probabilities

Decimal odds of 2.00 look like a 50% chance. But the three implied
probabilities of a match sum to **more than 1** — typically 1.02 to 1.08.
That excess is the *overround*, the bookmaker's margin, and it must be
removed before the prices mean anything as a forecast.

A model compared against raw `1/odds` is being compared against a baseline
that is deliberately wrong in the bookmaker's favour. It will look better than
it is, by roughly the size of the margin.

`football/market.py` removes it three ways, because they disagree most
exactly where it matters — on longshots:

| Method | What it assumes | Trade-off |
| --- | --- | --- |
| `multiplicative` (default) | margin applied proportionally | simple, standard; overstates longshots |
| `additive` | margin split equally across outcomes | corrects the other way; can go negative on an extreme favourite |
| `power` | solves `sum(p^k) = 1` | fits observed longshot bias best; costs a root-find per match |

None is correct. Pick one, say which, and run `compare_methods` to check the
conclusion does not flip when you switch. **If it flips, the finding is about
the margin model, not about the model.**

## 3. The data contract

`football/processor.py` owns it. Source: [football-data.co.uk](https://www.football-data.co.uk/),
one CSV per league per season, in `exported_data/football/`, fetched by
`football/downloader.py` ([§5](#5-getting-real-data)).

Required columns, stable across every season: `Date`, `HomeTeam`, `AwayTeam`,
`FTHG`, `FTAG`. Everything else drifts as bookmakers come and go.

Output is one tidy frame:

```
ds  home_team  away_team  home_goals  away_goals  outcome  odds_home  odds_draw  odds_away
```

with `matches.attrs["odds_source"]` and `matches.attrs["odds_are_closing"]`
recording where the prices came from.

### The trap: never mix opening and closing odds

This is the football counterpart of [Baloto's two eras](data-pipeline.md), and
it is just as invisible in the shape of the frame.

Bookmakers publish a price when a market **opens** and a different one when it
**closes**. Closing prices are sharp. Opening prices are soft, and a model that
"beats the market" against opening prices has usually beaten a bookmaker's
first guess rather than the market.

football-data marks closing odds with a `C` (`AvgCH`, `B365CH`) and publishes
them **only from 2019/20 onward**. So a merged history spanning that boundary
has closing odds for its recent half and none for its older half. Filling that
gap from the opening columns produces one `odds_home` column that silently
means two different things, and every model evaluated on it is judged against
two different bars at once — which shows up as an edge that exists only in the
older seasons.

The rule is therefore **one odds source per frame, or none**:

- `resolve_odds_source` picks the best source present in the file, preferring
  closing over opening and a market average over any single book.
- Rows missing that source stay `NaN`. There is **no per-row fallback**. A
  visible hole is recoverable; a silently mixed baseline is not.
- A price triple is usable or it is not — two of three prices cannot be
  normalised, so the whole row is blanked.
- `load_seasons` **raises** rather than concatenating files that resolve to
  different sources.
- `load_and_preprocess(..., closing_odds_only=True)` refuses a soft file
  outright.

`check_match_format` warns (never raises) when a frame is merely weak — no
odds at all, opening-only, or incomplete coverage — because those rows are
still real matches and a caller may want them for training even when they
cannot support an evaluation.

## 4. Synthetic data, with a known answer

`football/sample_data.py`, mirroring `lottery/utils/sample_data.py`: seeded,
deterministic, no private CSV needed.

But it does something the lottery generator cannot. Lottery draws are
generated from the null itself, so the tests can be checked for crying wolf.
Football has real signal, so this generates from a real model — per-team
attack and defence strengths, home advantage, independent Poisson goals — and
**carries the generative truth alongside** as `p_true_home` / `p_true_draw` /
`p_true_away`, computed exactly by summing the scoreline grid rather than
simulated.

On real data nobody knows the right answer, so a model can only be compared to
another model. Here the answer is known, which allows the two checks that
matter: a forecast that *is* the truth must beat the market, and the market
must be calibrated but beatable.

`market_noise` is the knob. At `0` the simulated book prices the truth exactly
and is unbeatable except for its margin — the realistic pessimistic case, and
the setting where the de-margining round-trip is checked. Turning it up
produces a soft market. **Nothing here claims a real bookmaker is beatable at
any setting**; the parameter exists so a test can tell "the code found an edge"
apart from "the code cannot find an edge that was planted".

Calibration is against the Premier League's long-run figures — roughly 45%
home wins, 25% draws, 30% away wins, 1.54 / 1.19 goals per side. Data that did
not look like football would make every downstream test easier to pass and
less informative.

**One known limitation, stated rather than hidden.** Goals are two independent
Poisson variables, the standard first model, known to under-produce draws and
low-scoring correlated scorelines. Dixon-Coles exists to correct exactly that,
so this is a fair test bed for it — but it is not a substitute for real
results when the question is about score dependence itself.

## 5. Getting real data

`football/downloader.py` fetches the season files. football-data publishes CSV
directly, so this downloads rather than scrapes — hence `downloader`, not
`scraper` — but it keeps the [lottery scraper's posture](data-pipeline.md#31-design-principle-fail-loudly):
fail loudly.

```bash
# Always look first.
python -m football.downloader --seasons 2019/20..2024/25 --leagues E0 --dry-run

python -m football.downloader --seasons 2019/20..2024/25 --leagues E0
python -m football.downloader --seasons 2324 --leagues E0,SP1,I1,D1,F1
python -m football.downloader --seasons 2015-2024 --leagues E0 --closing-odds-only
```

Seasons are written as `2324`, `2023/24`, `2023-24` or `2023`; ranges use `..`,
or `-` between two four-digit years (both bounds are season *start* years).
Files land in `exported_data/football/E0_2324.csv`.

Four decisions in it are worth knowing:

- **The file is written exactly as downloaded** — no column pruning, no
  renaming. `processor.py` owns the contract, and a downloader that pre-selected
  columns would become a second, weaker owner of it.
- **It is validated at download time anyway.** Every file goes through
  `preprocess_matches` before it is written, and the resolved odds source is
  printed per file. A format change is cheapest to notice now, and the summary
  says up front when two seasons resolve to different sources — those cannot
  later be loaded together, and finding that out mid-evaluation is worse.
- **An HTML error page never reaches disk under a `.csv` name.** A wrong path
  can come back as HTML with a 200, and `pd.read_csv` will happily turn that
  into a one-column frame. The response is sniffed before it is parsed.
- **A longer file is never replaced by a shorter one** without `--force`. An
  in-progress season legitimately grows on every re-download; coming back
  smaller is a truncated transfer.

### The "extra league" files

The rest of the world (`new/COL.csv`, `new/ARG.csv` and friends) is published on
a **different contract** — `Home`/`Away`/`HG`/`AG`, several leagues and seasons
stacked in one file, and **opening odds only** (`AvgH` / `PH` / `B365H`, never a
`C` column). `football/processor.py` does not read them.

```bash
python -m football.downloader --leagues COL --extra          # writes exported_data/football/COL.csv
```

`football/extra_processor.py` owns this contract — `preprocess_extra(df, league=None)`,
`load_extra(path, league=None)`, `available_leagues(path)` — and maps it onto the
same tidy frame the main processor produces, so the model, market, scoring and
dashboard consume it unchanged. Two rules are enforced hard:

- **One league per load.** A file with more than one `League` value raises
  `MatchFormatError` unless `league=` names one — stacking two competitions is the
  same mistake as merging two odds sources.
- **`odds_are_closing` is always `False`.** An extra file can never resolve to a
  closing source; `odds_source` is always an `extra_*_opening` name.

`--seasons` is ignored on the `--extra` path (these files are not per-season), and
without `--extra` the codes are refused by name. **The limit:** opening odds mean
the market baseline is the soft one, so **no corrected edge claim is possible on
Colombian data** — a model that beats these prices has probably beaten a
bookmaker's first guess rather than the market.

### The limit of this verification

The sandbox this was built in blocks football-data.co.uk at the network policy,
so **neither the parser nor the downloader has ever run against a real file
here.** Both are written against the documented format and covered by tests
built from that format, which is not the same thing as being validated against
reality.

So after the first real download, eyeball one file:

```bash
python -c "
from football.processor import load_and_preprocess
from football.market import market_probabilities
m = market_probabilities(load_and_preprocess('exported_data/football/E0_2324.csv'))
print(m.attrs['odds_source'], m.attrs['odds_are_closing'])
print(m.head().to_string())
"
```

Check the odds source is a closing one, the dates span the right season, and
the home-win rate lands near 45%. The parser raises on a broken date and warns
on a soft or incomplete market, so a file that produces neither is probably
fine — but a first look costs a minute and this code has never seen reality.

## 6. The dashboard page

`streamlit run dashboard/app.py`, then pick **⚽ Fútbol** in the sidebar. Five
numbered tabs — **1 · Datos**, **2 · Mercado**, **3 · Pronóstico**,
**4 · ¿Le gana al mercado?**, **5 · Valor** — and a **Europa / Colombia** source
toggle at the top. Datos and Mercado are descriptive: which odds source a file
resolved to, how big the margin is, that the market is calibrated, how the three
de-margining methods differ on one match, plus the model's own calibration curve.
**Pronóstico** is the two-team view ([§8](#8-the-model-and-the-two-team-view)),
now with the Elo beside Dixon-Coles and its rating table.
**¿Le gana al mercado?** scores every selected model in one pass and prints the
corrected threshold as 0.05 divided by however many ran. **Valor** is the staking
surface of [§11](#11-value-and-staking), and it shows no stake at all until the
verdict from the previous tab is on screen beside it. In Colombia mode the page
shows the opening-odds warning, and every model surface shows the market beside
it. Selecting season files that resolve to different odds sources renders the
refusal instead of merging them. Details in
[Dashboard §3](dashboard.md#3-fútbol-the-data-contract-the-market-and-the-model).

## 7. What is built, and what is not

The scoring rules, Dixon-Coles, the market evaluation and the forecast tab now
exist:

- **`football/scoring.py`** — Brier, ranked probability score and log-loss for the
  three-way outcome, plus a skill score. RPS is the default because H–D–A is
  ordered ([§8](#the-scoring-rule)).
- **`football/dixon_coles.py`** — attack/defence strengths with the low-score
  correction, optional time decay, one fitted model feeding 1X2, correct score,
  over/under and both-teams-to-score.
- **`football/h2h.py`** — recent form and head-to-head, descriptive only.
- **`football/evaluation.py`** — `beats_market_test`, a paired one-sided
  proper-score test against the market through `core/significance.py`, naive and
  corrected verdict together.
- **`football/elo.py`** — Elo ratings with an ordered-logit map to H–D–A
  ([§9](#9-elo-the-cheap-baseline)).
- **`football/ensemble.py`** — linear and logarithmic pooling of a model with the
  market ([§10](#10-pooling-with-the-market)).
- **`football/value.py`** — edge, the two bars, and Kelly staking
  ([§11](#11-value-and-staking)).
- **`football/backtest.py`** — walk-forward (`run_all`), date-cutoff
  (`run_holdout`, `expanding` / `frozen`) and multi-model (`compare_models`)
  evaluation.
- **`football/calibration.py`** — whether a forecast's 30% is a real 30%, and a
  recalibration that is never fitted on what it is then scored on
  ([§12](#12-calibration-a-different-question-from-edge)).

**What is still missing.** Lineups and injuries — the single largest thing the
closing price knows and no model here does; in-play data; and any notion of a
team's form being about *who* is playing rather than about recent results.

## 9. Elo: the cheap baseline

Elo was skipped for a long time on the grounds that it only produces a 1X2
vector while Dixon-Coles produces that *and* a scoreline, so a separate Elo would
be a weaker duplicate of one output. That reasoning was about Elo as a
**forecast**, and it holds. It is wrong about Elo as a **baseline**.

"Dixon-Coles beats the market" is a claim. "Dixon-Coles beats the market while
also beating one number per team, updated after each match" is a much more
interesting one — and if the expensive model *cannot* separate itself from Elo,
that is the most useful thing a run can tell you, because Elo costs a single pass
over the data.

### The draw is the whole difficulty

Classic Elo answers "who wins", which is two-way; the target is three-way.
Splitting the win probability by some fixed draw share would be a made-up number
wearing a rating system's credibility. Instead the rating gap goes through an
**ordered logit** whose two cut points and scale are fitted by maximum likelihood
on the training matches (the standard treatment, Hvattum & Arntzen). That has the
property the outcome needs: H, D and A are ordered, so the draw sits between the
two wins by construction rather than by assumption. The gap between the cut
points is optimised in logs, so they cannot cross and the result cannot stop
being a distribution.

### One matchday at a time

Ratings advance a **date** at a time, not a row at a time. Every fixture on a
date is predicted from the ratings as they stood before that date, and the date's
updates are applied together afterwards. Updating match by match would make the
model depend on the order rows happen to sit in within a matchday — an ordering
that does not exist, since the fixtures are played simultaneously — and would let
one 3pm result inform another 3pm forecast. A test shuffles the frame and demands
a bit-identical fit.

## 10. Pooling with the market

`football/ensemble.py`, and it asks a sharper question than the backtest does.

"Does the model beat the market?" sets a bar almost nothing clears, and a no
answers very little: a model can be genuinely informative and still lose to a
price that already contains everything it knows plus team news, lineups and
money. **Does the model know anything the market does not** is the better
question, and a blend answers it directly.

A blend at weight 0 **is** the market. It scores identically, and the paired test
returns an effect of exactly zero — not a degenerate case to guard against, but
the null the comparison is built on. If putting weight on the model improves the
score from there, the model carries information the price does not, whether or
not it could ever stand alone. A test pins that endpoint, because the whole
reading collapses if it drifts.

Two pooling rules, for the same reason `market.py` offers three de-margining
methods. **Linear** averages the probabilities and hedges: the result always
lands between its inputs and is never more confident than the more confident
source. **Logarithmic** takes a weighted geometric mean and renormalises, which
leans harder on whatever both sources favour and is far harsher on an outcome one
of them nearly ruled out — a source saying 2% drags the pool most of the way down
instead of being averaged away. Report which one you used; if a conclusion flips
between them it is about the pooling rule, not about the model.

Nothing here is fitted. The weight is a parameter the caller chooses and the
backtest measures — fitting it on the same matches you then score would be the
purest form of the leak this package exists to avoid.

## 11. Value and staking

`football/value.py` is football's counterpart to the lottery's jackpot splitting:
the module that answers "so what do I actually do", and the one most able to do
harm. Jackpot splitting is safe because it improves a quantity that is real
whether or not any model works. Nothing here is safe in that way.

### The two bars, which are different

This distinction is the whole module:

- To judge a **model**, compare it against the **de-margined** price. That is
  what `market.py` produces and `evaluation.py` tests, because the bookmaker's
  margin is not a forecast.
- To judge a **bet**, compare it against the raw **`1/odds`**. You pay the
  margin.

Between the two sits a band, and most betting systems live in it: the model is
more optimistic than the market and the disagreement is not big enough to pay for
the spread. `classify` returns that as a named state — `no_value`,
`disagreement_only`, `value` — rather than leaving it to a footnote, and
`margin_cost` prints the band's width.

### Kelly is not a safety feature

It is the stake that maximises long-run growth *given a true edge*. Applied to an
edge that is not real it does not merely fail to help: it sizes up precisely when
the model is most confidently wrong, which is how a staking plan turns a small
negative expectation into a fast one. `kelly_fraction` therefore defaults to a
**quarter** stake, and the dashboard will not show a stake at all until the
measured verdict is on screen next to it.

**Scope.** One outcome at a time. Backing two outcomes of the same match
simultaneously is a different optimisation — the bets are mutually exclusive, so
the single-bet formula over-stakes — and pretending otherwise would be the same
kind of quiet wrongness this package exists to avoid.

## 8. The model and the two-team view

### The model

`football/dixon_coles.py`. One maximum-likelihood fit on a tidy match frame
produces five things: a baseline scoring rate `mu`, a `home_advantage`, an
`attack` and a `defence` strength per team (each constrained to sum to zero), and
`rho`, the low-score dependence parameter.

```python
from football.dixon_coles import DixonColes
model = DixonColes.fit(matches, half_life=None)      # half_life in days → exponential time decay
model.predict_outcome("Team A", "Team B")            # (P_home, P_draw, P_away), H-D-A order
model.scoreline_matrix("Team A", "Team B")           # the full goals grid
model.most_likely_scores(...); model.over_under(..., line=2.5); model.both_teams_to_score(...)
model.predict_matches(frame)                         # (n, 3) for a whole frame
```

`DixonColes.fit` takes `matches` and optional `half_life` and `max_iter=200`; an
unknown team at prediction time raises `UnknownTeamError`.

**The low-score correction.** Two independent Poisson goal counts
(`independent_poisson_matrix`) under-produce the 0–0, 1–0, 0–1 and 1–1 scorelines
that real football clusters on. `_tau` multiplies exactly those four cells of the
grid by a factor in `rho` before renormalising; `rho` is bounded to ±0.4 and the
fit warns if it hits the bound (the correction is then at its limit and the model
is straining).

**Time decay.** With `half_life` set, each historical match is weighted
`0.5 ** (age_in_days / half_life)` in the likelihood, so a season-old result
counts for less than last week's. Left off, every match counts equally.

### The scoring rule

`football/scoring.py`. The outcome H–D–A is **ordered** — a forecast that put its
mass on H when the result was A is more wrong than one that favoured D. Ranked
probability score charges for that distance; Brier does not. So **RPS is the
verdict** and Brier is a diagnostic, the same shape as the lottery's "the pooled
test is the verdict". `METRICS = ("brier", "rps", "log_loss")`;
`per_match_scores(probs, outcomes, metric)` and `skill_score` are the entry
points.

### Pronóstico: model beside market

The tab takes two teams and, optionally, the three current decimal prices. It
shows head-to-head and recent form (descriptive), then the Dixon-Coles 1X2
vector, a scoreline heatmap, over/under and BTTS. **The model-vs-market columns
appear only when odds are entered** — a de-margined market vector next to the
model's, and their difference. No forecast is shown on its own.

### Resultados: the measured verdict

A gated section runs `football/backtest.py` walk-forward and reports
`beats_market` and `beats_market_corrected` (corrected emphasised), the model and
market scores, the effect size and its interval — exactly as the lottery backtest
does, and through the same `core/` machinery. The evaluation is a
**paired-difference** test: per match, `market_score − model_score`, tested
one-sided against a null mean of 0. See [Evaluation](evaluation.md#9-football-dixon-coles-vs-the-market).

---

**Next:** [Cycling](cycling.md) · [Architecture](architecture.md) · [Evaluation](evaluation.md) · [Development](development.md)


## 12. Calibration: a different question from edge

`scoring.py` returns one number, and one number cannot separate two very
different failures. A model that ranks matches well but states its case too
strongly and a model that is appropriately humble about matches it has no idea
about can post the same RPS. The first is fixable with one parameter; the
second needs a better model.

That distinction is not academic, because of what sits downstream.
`value.py` turns a probability into a stake, and an over-confident probability
sizes up **precisely when the model is most confidently wrong** — which is what
the quarter-Kelly default was already hedging against without being able to say
so. A calibration curve is the thing that can say so.

### The one sentence that has to travel with it

**A calibrated model is not a profitable one.** A forecast that simply copies
the de-margined closing price is perfectly calibrated and has no edge
whatsoever. Calibration asks whether the numbers mean what they claim; the
verdict is still `beats_market_test`. This is why the dashboard renders the
curve *after* the backtest verdict rather than beside it, and why every piece
of copy on that surface says it outright. Treating a good calibration plot as a
result is the football version of presenting a lottery hindcast as a
prediction.

### What it measures

| Function | Question | Note |
| --- | --- | --- |
| `reliability_curve` | Does a 30% happen 30% of the time? | A bin below `MIN_BIN_COUNT` reports its count and **no** frequency |
| `expected_calibration_error` | How far off, on average? | `coverage` says what share of forecasts it could measure — read it beside the error |
| `calibration_in_the_large` | Is the whole model listing to one side? | **Two-sided**, and both verdicts |

The sparse-bin rule is `lottery/analysis/structure.py:goodness_of_fit`'s, in a
new costume: three matches cannot measure a frequency, and drawing them as if
they could is how a reliability diagram invents a story. The only values a
three-match bin can take are 0, ⅓, ⅔ and 1, none of which says anything about a
forecast of 0.27.

`calibration_in_the_large` being two-sided is a deliberate departure from the
rest of this package, where only `p_value_greater` may back a claim because a
model *worse* than its baseline is not a finding. Here it is: forecasting home
wins at 50% when they happen 40% of the time, and forecasting them at 30%, are
both mis-calibration, and a one-sided test waves one of them through. It is also
three tests over one set of matches, so it emits `miscalibrated` and
`miscalibrated_corrected` together like everything else here.

### Fixing it, honestly

`TemperatureScaler` raises every probability to 1/T and renormalises. That is
the entire model, and its smallness is the point: it has nowhere to put noise,
so a temperature fitted on a few hundred matches is a claim about
over-confidence and nothing else. If a model's only fault is stating its case
too strongly, this fixes it completely; if the fix is large, the diagnosis was
wrong. `IsotonicCalibrator` is the free-form upper bound — run it to find out
what recalibration could buy at most, and if it barely beats the temperature,
the model's problem is not its confidence.

**The gate.** `prequential_calibrate` fits each row's correction on the rows
strictly before it and applies it to that row alone. The leading `min_fit` rows
pass through untouched, because the alternative is fitting a correction on a
handful of matches and calling the result out-of-sample. This is the same
refusal `ensemble.py` makes about its blend weight, for the same reason: fitting
a parameter on the matches you then score is the leak this package exists
around, and a calibrator fitted in-sample improves every score it touches and
means nothing.

`compare_models(..., calibrate="temperature")` wires it in, and does one more
thing that is easy to get wrong: it **trims every model, the market and the
outcomes by the same `min_fit`**. Scoring the whole series would mix rows the
correction reached with rows it could not and pull any difference toward zero —
and trimming one model and not another would break the rule that several models
are scored on the same held-out set or not compared at all.
