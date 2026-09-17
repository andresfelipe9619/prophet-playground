# Cycling

The third domain. Read [Domain and Premise](domain-and-premise.md) and
[Football](football.md) first — this page is about how cycling differs from
both, and the differences are in the shape of the target, not in the discipline.

The contract, the scraper and seeded synthetic races, and on top of them the
layer that lets a forecast mean something: a **ranking baseline**
([§6](#6-the-baseline-cyclingbaselinepy)), a **proper scoring rule** for an
ordering ([§7](#7-scoring-an-ordering-cyclingscoringpy)), a fitted
**Plackett-Luce** rider-strength model ([§8](#8-the-model-cyclingplackett_lucepy))
and the paired evaluation against that baseline
([§9](#9-evaluation-cyclingevaluationpy)). What is still missing is in
[§11](#11-what-is-not-built-yet) — most of all the market, which is the bar
wherever a price exists.

## 1. Where it sits between the other two

| | Baloto | Football | Cycling |
| --- | --- | --- | --- |
| The process | i.i.d. uniform **by design** | real, persistent signal | real, persistent signal |
| The target | 5 numbers from 43 | one of three outcomes | an **ordering** of ~180 riders |
| The baseline | exact hypergeometric | the closing betting line | the market where it exists, otherwise the pre-race ranking |
| Can a model beat it? | No. Provably not. | Yes — that is the point | Yes, but the baseline is already very strong |
| The characteristic self-deception | a lucky backtest | beating opening odds | scoring only the riders who finished |

Riders differ enormously and persistently — this is the least random of the
three domains. Which is exactly why it is the easiest to fool yourself in: a
model that knows nothing except "Pogačar is fast" will look impressive against
a bad baseline, and almost every naive baseline in cycling is bad.

**What is not a baseline:** a uniform draw over the start list. Predicting a
random order among 180 riders is trivially beaten by anything, it makes any
model look brilliant, and a result reported against it says nothing.

**What is:** a betting price where one exists (for a stage winner or the
overall, the same closing-line argument as football's), and otherwise the
**pre-race ranking** — UCI or PCS points, or a start-list quality score. That
baseline is already hard, because "the best riders finish near the front"
explains most of a bike race, and any model has to earn the difference.

## 2. The data contract

`cycling/processor.py` owns it. Source:
[procyclingstats.com](https://www.procyclingstats.com/), scraped by
`cycling/scraper.py` into `exported_data/cycling/`.

One row per rider per result:

```csv
Date,Race,Kind,Stage,Rank,Rider,Team,Status,TimeSeconds
29/06/2024,tour-de-france,stage,1,1,POGAČAR Tadej,UAE Team Emirates,FIN,15322.0
29/06/2024,tour-de-france,stage,1,,VINGEGAARD Jonas,Team Visma,DNF,
```

| Column | Format | Notes |
| --- | --- | --- |
| `Date` | `dd/mm/yyyy` | Day first, like the lottery contract. |
| `Race` | slug | The race's own slug, e.g. `tour-de-france`. |
| `Kind` | `stage` / `one_day` / `gc` | **One kind per file.** See [§3](#3-the-three-traps). |
| `Stage` | integer or blank | Blank for a one-day race; for `gc`, the stage the standing follows. |
| `Rank` | integer or blank | Blank for every non-finisher. |
| `Status` | `FIN` `DNF` `DNS` `DSQ` `OTL` `NR` | Why a rider has no rank. |
| `TimeSeconds` | float or blank | **Total** elapsed seconds, never a gap. |

`preprocess_results` turns that into the tidy frame every consumer sees —
`ds, race, kind, stage, rank, rider, team, status, time_seconds` — and records
the kind in `results.attrs["result_kind"]`.

It **raises** when the contract is broken (a mixed file, a rank that
contradicts its status, two riders sharing a rank) and **warns** when the data
is merely weak, which is the same line `lottery/utils/processor.py` draws: a
weak file still holds real results and a caller may want them.

## 3. The three traps

### One kind of result per frame

A rank of 4 in a stage result and a rank of 4 in a general classification are
different quantities: one is a sprint finish, the other is three weeks of
accumulated time. They are published by the same site, in the same table
layout, for the same riders, on adjacent pages. A frame holding both has a
`rank` column that means two things and every model fitted on it is fitted on
a mixture.

This is the cycling counterpart of
[football's opening/closing odds](football.md#the-trap-never-mix-opening-and-closing-odds),
and it is handled the same way: `preprocess_results` raises on a mixed frame,
`load_races` raises rather than concatenating files of different kinds, and the
scraper puts the kind in the filename so the two cannot land in one file by
accident.

### Non-finishers stay in the frame

A fifth of a Grand Tour's start list can fail to reach the end, and the
abandons are **not random** — they concentrate among the riders whose form was
worst, which is to say the ones a model was least sure about. Dropping them
turns "predict the finishing order" into "predict the order among those who
finished", which is a strictly easier problem and one nobody can bet on.

So a non-finisher stays, with `rank` NaN and a `status` saying why. Filtering
is an explicit `finishers()` call or `load_and_preprocess(..., finishers_only=True)`,
never a side effect of loading — and the filter runs *after* the checks, so
the "this frame has no abandons at all" warning still fires on the file as
published.

### `time_seconds` is a total, never a gap

Results pages publish the winner's elapsed time and everyone else's **gap** to
it (`4:15:22`, then `0:14`, then `,,` for "same time as the rider above").
Storing those gaps in a column that means totals produces a frame that looks
completely normal and has rank 2 finishing four hours ahead of rank 1.

Resolving them is the scraper's job. `time_order_violations` looks for the
fingerprint afterwards — a rider timed *faster* than someone placed ahead of
them — and `check_result_format` reports it. When the winner's own time is
missing there is nothing to anchor to, and every time on that page is left
NaN rather than written as a gap.

## 4. The scraper

```bash
# Always look first. Nothing here has ever run against the live site.
python -m cycling.scraper --race tour-de-france --year 2024 --stages 1-21 --dry-run

python -m cycling.scraper --race tour-de-france --year 2024 --stages 1-21
python -m cycling.scraper --race tour-de-france --year 2024 --kind gc --stage 21
python -m cycling.scraper --race milano-sanremo --year 2024 --kind one_day
```

Each run writes `exported_data/cycling/<race>_<year>_<kind>.csv`, merging into
whatever is there (keyed on race, kind, stage and rider; existing rows win, so
a hand-corrected file survives a re-scrape) and validating the result through
`preprocess_results` before it is written.

### It finds the table by its headers, not by CSS classes

`find_results_table` reads every table's header row, maps the headers to fields
through `HEADER_ALIASES`, and keeps the table with the most recognised columns
(breaking ties by length). Class names on a results site churn constantly, and
a class-based selector that misses returns **zero rows** — the failure mode
this whole module is shaped against. A header row that no longer says `Rnk` is
worth stopping on, and `HEADER_ALIASES` is the single place to teach it a new
column name.

Counting rows alone would not do: a page published mid-race can have a sidebar
table longer than the classification itself.

### Missing pages vs. a broken parser

The same distinction the [lottery scraper](data-pipeline.md#33-empty-years-vs-a-broken-parser)
draws:

- A stage with **no page** (404) or **no rows** is skipped — `--stages 1-21`
  has to work on a race in progress. If *no* stage yields anything, that is the
  parser-is-broken case and it raises.
- A **structural** problem raises immediately, from any stage: an unknown rank
  marker, an unreadable time, a table with no rider column, a page of nothing
  but abandons. Those mean the markup changed, and skipping the page would hide
  it.

### Verification, and its limit

The sandbox this was written in blocks procyclingstats.com at the network
policy, so **the parser has never seen a real page.** It is written against the
site's documented table layout and tested against HTML fixtures built from that
layout, which is not the same as being validated against reality.

Before trusting a scrape: run `--dry-run`, then check three things against the
page in a browser — the winner's time is their real elapsed time, each rider
below them has that time *plus* their gap, and the abandons are present with no
rank rather than missing.

## 5. Synthetic races, with a known answer

`cycling/sample_data.py`, mirroring `football/sample_data.py`: seeded,
deterministic, no scrape needed. It generates from a real model — a latent
per-rider `ability_true`, worth minutes on a mountain stage and seconds in a
sprint — and hands the truth back alongside the results, which is what lets a
test check that a forecast which *is* the truth orders the classification
correctly.

Three things are modelled because leaving them out would make every downstream
test easier than reality:

- **Bunch finishes.** On a flat stage most of the field is credited with the
  winner's exact time, so the data contains large groups of identical times —
  what `,,` means on a results page.
- **Abandons.** The stage a rider leaves on carries a `DNF` row with no rank and
  no time; later stages simply do not list them, which is how a scraped file
  reads.
- **Terrain.** Ability is worth `CLIMBING_SECONDS_PER_ABILITY` on a hard day and
  `SPRINT_SECONDS_PER_ABILITY` on a flat one, so the front of a sprint stage
  arrives on one time while a mountain stage splits it by minutes.

**One known limitation, stated rather than hidden.** Abandons are independent
draws against a per-rider hazard, which is wrong in the way that matters most:
real abandons cluster on one day, in one crash, and there are no teams riding
for each other here at all. It is a fair test bed for a rider-strength model
and not a substitute for real results when the question is about dependence
between riders.

`ability_true` is an answer key. Nothing outside tests may read it.

## 6. The baseline: `cycling/baseline.py`

The thing a model has to beat, and the piece without which nothing else here
could mean anything. It is cycling's `football/market.py`.

### A uniform draw is not a baseline

With ~180 starters, a uniform draw gives everyone 0.55%. Any forecast beats it by
knowing a single name, so a model that beats it has demonstrated only that
cycling has favourites. It is implemented as `uniform_worths` **precisely so the
mistake has a name and a docstring**, and it appears in the dashboard's
evaluation table where it comes out worse than the real baseline — an unnamed
mistake is one that gets made quietly.

The real baseline is the market where a price exists and otherwise the **pre-race
ranking**: `worths_from_points` for UCI/PCS points, `worths_from_rating` for a
strength on any log-odds-like scale, and `form_worths` to build a ranking out of
a rider's earlier results when no points are to hand. `form_worths` takes an
`as_of` and uses only results strictly before it — a ranking that has seen the
race it is ranking for is not a baseline, it is an answer key.

### Why Plackett-Luce

The target is an ordering of ~180 riders, so a vector of win probabilities is not
enough on its own: it says nothing about second place. Luce's rule gives each
rider a positive **worth**, makes their win probability their share of the total,
then removes the winner and repeats. One number per rider generates a
distribution over whole finishing orders, which is the shape the target has.

`top_n_probabilities` samples that distribution with the Gumbel-max trick — add a
standard Gumbel to each `log(worth)`, sort descending, and you have drawn an
exact Plackett-Luce ordering in one vectorised pass. There is no cheap exact form
beyond first place, so the numbers are estimates and the docstring says how
coarse they are.

## 7. Scoring an ordering: `cycling/scoring.py`

`METRICS = ("plackett_luce", "winner_log", "winner_brier")`, and **the
Plackett-Luce log score is the verdict** — the same shape as "the pooled test is
the verdict" on the lottery side and "RPS is the verdict" in football. It is the
only rule here that is proper over the actual target, a whole ordering, rather
than over a summary of it. Lower is better, everywhere.

Rank correlations (`spearman`, `kendall_tau`) and `top_n_accuracy` are here too,
and they decide nothing. A rank correlation rewards getting the middle of the
bunch roughly right, which is the easy and worthless part; a forecast can score
well on it having missed every podium place.

### Non-finishers stay in the denominator, and that is structural

This is the most important line in the module. The likelihood places riders one
at a time, and at each step the denominator is the total worth of everyone **not
yet placed** — which includes every rider who abandoned. A forecast that put its
money on a rider who climbed off is charged for it: they were available to win
each position and took none.

Drop the abandons and the score silently improves, because the field has been
renormalised to the riders who made it. That turns "predict the finishing order"
into the strictly easier "predict the order among those who finished", which is
[the thing §3 already refuses to do to the data](#non-finishers-stay-in-the-frame).
Here the refusal is not a check anyone has to remember — it falls out of the
arithmetic.

## 8. The model: `cycling/plackett_luce.py`

Cycling's Dixon-Coles, and the first thing in the package that estimates rather
than describes. One latent strength per rider, fitted by maximum likelihood over
the Plackett-Luce likelihood of every finishing order in the training data, using
Hunter's minorise-maximise iteration — monotone by construction, one pass over
the data per sweep, and unlike a generic optimiser over 180 parameters it cannot
wander.

**How it differs from the baseline it must beat.** `form_worths` scores a placing
heuristically and sums; that is a reasonable proxy for a ranking, and it is what
a ranking *is*. The model asks which strengths make the orderings actually
observed most likely, which discounts beating a weak field and rewards beating a
strong one.

**The Gamma prior is not a tuning knob.** Without it the fit is not merely noisy
on thin data, it is undefined: a rider nobody ever finished behind has a
maximum-likelihood worth that rises without bound, and one who never finished
ahead of anyone has one that goes to zero, which then takes a logarithm to
negative infinity in every downstream score. `prior_strength` is measured in
placings, so a rider with that much evidence sits halfway between the field
average and what their own results say. On the synthetic Grand Tour it moves the
correlation between fitted strength and the generator's own `ability_true` from
undefined to **0.76**.

An unseen rider gets the field average rather than a refusal, unlike football's
`UnknownTeamError`. The cases are genuinely different: a fixture between two
teams, one unheard of, cannot be predicted at all, while a 180-rider start list
with three neo-pros in it is an ordinary Tuesday, and dropping the race over them
would throw away the 177 riders the model does know.

## 9. Evaluation: `cycling/evaluation.py`

`beats_baseline_test` is the domain half of the contract with
`core/significance.py`, exactly as `beats_market_test` is in football: score the
forecast and the ranking with the same rule per race, take the per-race
difference (baseline minus model, so positive means the model did better), and
test whether its mean is greater than zero. Paired, one-sided, with the naive and
corrected verdicts emitted together as `beats_baseline` and
`beats_baseline_corrected`.

`walk_forward` hands every forecaster only the results strictly before each race;
`compare_forecasters` runs several against one baseline and sets `n_comparisons`
to the number of challengers. There is **no separate backtest module** — in
football one exists because the expensive part is refitting a goals model inside
the window loop, while here the walk is five lines and keeping it beside the test
is what makes it obvious that the two share an `as_of`.

### The hard part is the sample size

A Grand Tour is 21 scored races; a season of one-day classics is a few dozen.
Twenty-odd paired observations resolve only a large difference, so a null result
here is even more a statement about the sample than it is on the lottery side —
which is why `n_races` travels with every verdict. And on a sprint stage the
finishing order is very nearly noise by construction: ability is worth four
seconds against forty-five of race circumstance, so no forecast can or should
beat a ranking there.

### What the controls say

Both hold, and they are the reason to believe the rest:

- An **oracle** handed the generator's own `ability_true` beats the ranking, with
  the corrected verdict, on a race where every stage is a climbing stage.
- A forecaster that **is** the baseline comes out at an effect of exactly zero.
- A **uniform draw** loses to the ranking, which is the claim §6 makes, measured.

On the default synthetic Grand Tour — two stages in three a sprint — the fitted
model edges the ranking and does **not** clear the corrected threshold on 17
races. That is the honest result and the sample-size story above, in one number.

## 10. The dashboard page

`streamlit run dashboard/app.py`, then pick **🚴 Ciclismo** in the sidebar. Five
tabs: **Datos**, **Abandonos** and **Tiempos** put the three invariants of §3 on
screen, since none of them is visible in the shape of a frame; **Pronóstico**
shows the baseline and the model side by side for one race, built only from what
came before its date; **¿Le gana al ranking?** runs the walk-forward comparison.
Loading a stage result together with a general classification renders the
refusal. Details in
[Dashboard §4](dashboard.md#4-ciclismo-the-result-contract-the-ranking-and-the-model).

## 11. What is not built yet

1. **The market baseline.** Where a price exists it, not the ranking, is the bar.
   Nothing here reads odds yet, so every verdict is against the ranking and says
   so.
2. **Course and terrain.** A climber and a sprinter are not one strength number,
   and a model that cannot tell a mountain stage from a bunch sprint is leaving
   most of the available signal on the table.
3. **Teams.** Riders work for each other, which is the largest piece of structure
   the current model ignores entirely.
4. **Correlated abandons.** The synthetic generator draws them independently,
   which is wrong in the way that matters most — real abandons cluster on the
   same day, in the same crash.

---

**Next:** [Football](football.md) · [Architecture](architecture.md) · [Development](development.md)
