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
([§9](#9-evaluation-cyclingevaluationpy)), the **market baseline** that is the
bar wherever a price exists ([§10](#10-the-market-cyclingmarketpy-and-cyclingpricespy)),
and the covariates a single strength number leaves out
([§11](#11-terrain-specialisation-team-and-fatigue-cyclingfeaturespy)). What is
still missing is in [§13](#13-what-is-not-built-yet) — most of all a source for
those prices, since nothing here fetches them.

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

`specialisation` adds a fourth, off by default: above 0, a rider's mountain
strength and flat strength are **different numbers** (`climb_true` and
`sprint_true`, both answer keys), so the world contains real climbers and
sprinters. At 0 they are one number, no extra randomness is drawn, and every
seeded fixture is bit-identical to the race this generator has always made. It
exists so "does conditioning on terrain help?" has a world with a known answer
on both sides — [§11](#11-terrain-specialisation-team-and-fatigue-cyclingfeaturespy)
uses it exactly the way `lottery/analysis/sensitivity.py` plants a bias.

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

## 10. The market: `cycling/market.py` and `cycling/prices.py`

The bar the ranking was standing in for. Every verdict in §9 is against the
pre-race ranking, which is the soft bar — the one a model beats by being a
slightly better reader of recent form. Where an outright price exists it, not
the ranking, is what a forecast has to clear, and these two modules make that
comparison expressible. `market.py` does the arithmetic, `prices.py` owns the
file, and the split is football's: the arithmetic is testable on a vector of
numbers, and the mistakes that actually happen are about which numbers ended up
in the vector.

### An outright book is not a football book with more rows

Football's overround is 1.02-1.08. An outright cycling market runs to 1.4 and
beyond, because a bookmaker pricing 180 mutually exclusive runners takes a
margin on each. Two consequences follow, and both are larger than their football
counterparts.

The three normalisations that "disagree on longshots" in football disagree
*enormously* here, because almost every runner is a longshot. Measured on a
180-runner book at an overround of 1.76:

| method | favourite | top-10 share | tail share | riders zeroed |
| --- | --- | --- | --- | --- |
| multiplicative | 0.142 | 0.485 | 0.133 | 0 |
| additive | 0.214 | 0.705 | 0.000 | 106 |
| power | 0.199 | 0.596 | 0.087 | 0 |

`compare_methods` is not optional reading here the way it nearly is in football.
**`additive` is kept as a named mistake**, the same role `uniform_worths` plays
in §6: subtracting the same absolute excess from every runner drives 106 of 180
negative, and clipping them at zero says 106 riders cannot finish first. The
clip breaks the sum, so the result is renormalised and then *looks* like a
distribution — which is exactly why `n_zeroed` is reported. A method that only
reveals itself in a column nobody printed is one that gets used.

And the favourite-longshot bias is the whole shape of the book. A 200/1 rider is
not priced there because anyone believes 0.5%; that is the shortest price the
book can offer on a runner who will not win. Multiplicative de-margining scales
every price by one factor and leaves that bias fully intact, which on this market
means the baseline is badly wrong about 170 of the 180 riders. `power` is the
default for that reason.

### The quoted field is not the field

A book prices the 40 runners anyone will bet on and leaves 140 unquoted. Those
140 still start, and one of them wins stages. De-margining normalises over the
runners with prices, which implicitly gives the rest probability zero — and a
zero takes the Plackett-Luce log score to minus infinity the first time an
unquoted rider wins.

`field_worths(riders, priced_riders, odds, unpriced="longest")` gives each
unquoted rider the implied probability of the **longest price the book actually
put up** and renormalises the whole field, returning `(worths, n_unpriced)` so
the extrapolation is never silent. That is the book's own statement of its
floor, and it is deliberately unflattering to the market: handing 140 riders a
real probability each takes probability away from the favourites, so the
baseline it builds is *weaker* than the book. **A model that beats it has not
yet beaten the market.** `unpriced="refuse"` raises instead, for a caller who
would rather drop the race than score against an extrapolation.

### Probabilities are not worths

Everything downstream of `cycling/baseline.py` consumes Plackett-Luce worths, and
a win probability is not one. Under Luce's rule a rider's win probability is
`w_i / sum(w)`, so `worths_from_market` inverts that exactly — up to the scale
the model does not identify, fixed at mean 1 the same way `plackett_luce.py`
fixes it. What the inversion does **not** recover is how the market would order
the rest of the field: a book prices who wins, and the parts of a finishing order
below first place are not in the prices at all. A market baseline built this way
is a strong claim about the front of the race and an extrapolation about the back.

### One book per frame, one market per frame

`prices.py` is the contract, and its guards are about *identity* rather than
values — every number in a mixed price frame is a valid decimal price. This is
football's opening/closing trap in a third costume, and it is worse here:

- **Two books in one frame** do not merely differ in margin, they quote
  different fields. Stacked, a race with 176 starters has 240 priced runners and
  an overround that means nothing. `preprocess_prices` refuses a second `book`
  value; `load_books` refuses to concatenate files that resolve to different ones.
- **Two markets in one frame** — a price on the Tour's GC beside a price on its
  seventh stage — normalise against each other and describe a race nobody ran.
  This is §3's one-kind-per-frame rule one layer up.
- A rider priced twice, a field whose prices sum to **under** 1 (which means
  runners are missing, and normalising hands their probability to whoever is
  left), and a column of fractional odds that was never converted are all
  refusals rather than warnings.

`market_slice` cuts one market out of a season's worth of prices with the same
`attrs` a single-market file would carry, because everything downstream
normalises over a field and must be handed exactly one.

### Using it as the baseline

`market_forecaster(prices)` returns the `f(history, riders, as_of) -> worths`
shape `walk_forward` already takes, so the market drops into
`compare_forecasters` as the **baseline** wherever prices exist:

```python
from cycling.evaluation import compare_forecasters
from cycling.prices import load_books, market_forecaster

prices = load_books(["exported_data/cycling/prices/tour-2024-bookA.csv"])
table, scores = compare_forecasters(
    results,
    {"market": market_forecaster(prices),
     "ranking": ranking_forecaster,
     "plackett-luce": model_forecaster,
     "uniform draw": uniform_forecaster},
    baseline="market",
)
```

It looks at nothing in `history`, which is the point: a price quoted before the
race is already a forecast made without the result. Two races priced on the same
day are **not scored** rather than guessed between — returning None drops that
race from both sides of the paired test, which is what `walk_forward` already
does for a forecaster that cannot answer.

The endpoint is pinned by a test, as every endpoint in this project is: a
forecast that *is* the market scores an effect of exactly zero against it. That
zero is what any gain is measured from.

**Nothing in this repository fetches prices.** There is no scraper and no
source, and this page says so rather than implying a pipeline. What exists is
the shape a price file must have for the verdict to be against the market
instead of against the ranking, so that the day prices are to hand the baseline
is already there.

## 11. Terrain, specialisation, team and fatigue: `cycling/features.py`

`plackett_luce.py` gives every rider one worth. That is the right first model and
it is wrong about the thing every cycling fan knows: a sprinter and a climber are
not two points on one scale, they are good at different days. This module builds
the covariates that say so, and `TerrainPlackettLuce` is the first model to use
one.

### Terrain is known before a race; a result's terrain is not

The parcours of tomorrow's stage is published months ahead, so conditioning on it
is not foresight. But nothing in this project's data contract carries it — a
scraped result has a date, a rank and a time, and no roadbook. So terrain here is
**inferred from the finish**, which is fine for a stage that has happened and is
leakage for the stage being predicted.

The leak would be invisible: a label read off tomorrow's result looks exactly
like a label read off a roadbook. So the refusal is structural rather than
advisory. `terrain_of` raises for a race at or after `as_of`, everything that
conditions on terrain takes the target race's terrain as an argument, and
`roadbook(dates, terrains)` is where the caller supplies it — from a published
roadbook, a hand-written list, or, in tests, the generator's own stage plan,
which is fixed before the race is simulated and is therefore honestly exogenous.

### What the inference reads

The share of finishers credited with the winner's **exact** time. A flat stage
ends in a bunch sprint and most of the field shares one time; a mountain stage
strings the race out and almost nobody does. Measured on the synthetic Grand
Tour the two classes sit at 0.80 and 0.01 with nothing between them, so the 0.5
threshold is the middle of a gap rather than a tuned number. It reads the time
*structure* rather than the time *spread*, because a spread in seconds is not
comparable between a four-hour stage and a 45-minute time trial.

### The other three

- **Specialisation** is climbing form minus sprinting form, not the two levels:
  the levels are dominated by how good the rider is overall, and the tilt is the
  part that is about what kind of rider they are.
- **Team strength excludes the rider it describes.** A team mean that includes
  them is their own form wearing a team jersey, and it would enter a model twice.
- **Fatigue is race days, not calendar days.** A rider who has raced eighteen of
  the last twenty-one and one who flew in are in different states and the date
  cannot tell them apart. A rider who abandoned a fortnight ago correctly stops
  accumulating.

Every one of them goes through a single `_history` call whose `<` is the same
refusal `form_worths` makes, and one test covers the lot by rebuilding each
feature from a hand-truncated frame.

### Terrain-conditional worths: `TerrainPlackettLuce`

One strength per rider **per kind of day**: the whole calendar is fitted first,
then each terrain's races on top of it, with the unconditional worths as the
**prior mean** rather than the field average. That is the difference between a
refinement and a noisier copy — seven mountain stages shrunk toward 1 is seven
stages of noise around the field, while seven shrunk toward the rider's own
overall strength is a correction to it. A rider who has ridden no mountain
stages keeps their overall worth rather than being reset to the field, which was
a real defect in the first version of this and an invisible one: the reset
produces a perfectly ordinary number.

A terrain with fewer than `MIN_TERRAIN_RACES` races **falls back to the
unconditional fit**, and `available` says which terrains got their own. A silent
fallback would leave the model looking conditional everywhere while being
unconditional half the time, which is exactly what a results table cannot show.

### Does it help? Measured, in a world with specialists and a world without

`cycling/sample_data.py` gained a `specialisation` dial for this — at 0 the race
is exactly the one it has always generated, and above 0 a rider's mountain
strength and flat strength are different numbers. This is `sensitivity.py`'s move
in a third domain: a null result about terrain means nothing unless the same
machinery finds terrain when it was planted.

Every number below is the terrain-conditional fit against the **unconditional
one**, paired per race through `beats_baseline_test`, with the ranking and the
uniform draw in the same corrected family. Positive means the conditional model
scored lower, which is better.

| calendar | specialists planted | races scored | effect | p (one-sided) | corrected verdict |
| --- | --- | --- | --- | --- | --- |
| 21 stages, seeds 0 / 3 / 7 | none | 15 | +0.043 / +0.051 / +0.058 | 0.063 / 0.076 / 0.056 | no |
| 21 stages, seeds 0 / 3 / 7 | `specialisation=2.0` | 15 | +0.080 / +0.075 / +0.095 | 0.024 / 0.028 / 0.019 | no |
| 60 stages, seed 0 | none | 54 | +0.140 | < 1e-5 | **yes** |
| 60 stages, seed 0 | `specialisation=2.0` | 54 | +0.176 | < 1e-5 | **yes** |

At 15 races nothing clears the corrected threshold even where the edge is real
and planted — which is §9's sample-size point arriving exactly where it was
predicted to, and is why the 60-stage rows exist. In the same 60-stage runs the
ranking (-0.011) and the uniform draw (-0.057) both lose to the plain
Plackett-Luce fit, which is the comparison table doing its usual job.

Two things in that table are worth more than the verdicts.

**The effect grows with the planted specialisation**, which is the control
working: the machinery finds specialists in proportion to how many there are.

**And it is positive even with no specialists at all.** That is not a bug and it
is not luck. On this generator ability is worth three minutes on a mountain stage
and four seconds in a sprint, so a flat stage is very nearly pure noise — and a
conditional fit for the mountains simply excludes it. Terrain conditioning here
buys *weighting the informative days*, not only *finding specialists*, and on a
long enough calendar that alone clears the corrected bar. How much of this
carries to real racing is an open question: real sprints are less purely random
than these, so expect less.

**Nothing on this is wired into a verdict by default.** The dashboard's
walk-forward stays unconditional, because a roadbook column does not exist in
the contract and the alternative — inferring the target race's terrain — is the
leak this section is built around.

## 12. The dashboard page

`streamlit run dashboard/app.py`, then pick **🚴 Ciclismo** in the sidebar. Six
tabs: **Datos**, **Abandonos** and **Tiempos** put the three invariants of §3 on
screen, since none of them is visible in the shape of a frame; **Pronóstico**
shows the baseline and the model side by side for one race, built only from what
came before its date, with a terrain selector the reader fills in *from the
roadbook* — and a caption saying so, plus how many past races of that terrain
were behind the fit or that it fell back to the unconditional one; **Terreno y
forma** puts §11's covariates on screen descriptively, including the warning
that where sprints are near-random every strong rider reads as a climber;
**¿Le gana al ranking?** runs the walk-forward comparison.
Loading a stage result together with a general classification renders the
refusal. Details in
[Dashboard §4](dashboard.md#4-ciclismo-the-result-contract-the-ranking-and-the-model).

## 13. What is not built yet

1. **A price source.** The market baseline exists ([§10](#10-the-market-cyclingmarketpy-and-cyclingpricespy))
   and nothing fetches prices for it, so in practice every verdict is still
   against the ranking and says so. This is now a data problem rather than a
   modelling one, which is the smaller of the two.
2. **A roadbook.** The terrain machinery exists
   ([§11](#11-terrain-specialisation-team-and-fatigue-cyclingfeaturespy)) and
   the one thing it needs — the terrain of the race being predicted, from
   outside its own result — is not in the data contract. Until it is, the
   walk-forward stays unconditional.
3. **Teams riding for each other.** `team_strength` is a covariate about the
   eight riders around someone; the tactics are not modelled at all, and they
   are the largest piece of structure still missing.
4. **Correlated abandons.** The synthetic generator draws them independently,
   which is wrong in the way that matters most — real abandons cluster on the
   same day, in the same crash.

---

**Next:** [Football](football.md) · [Architecture](architecture.md) · [Development](development.md)
