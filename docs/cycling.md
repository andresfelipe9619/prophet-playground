# Cycling

The third domain. Read [Domain and Premise](domain-and-premise.md) and
[Football](football.md) first — this page is about how cycling differs from
both, and the differences are in the shape of the target, not in the discipline.

Currently a **data layer only**: the contract, the scraper, and seeded
synthetic races. No models and no scoring rules yet — see [§7](#7-what-is-not-built-yet).

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

## 6. The dashboard page

`streamlit run dashboard/app.py`, then pick **🚴 Ciclismo** in the sidebar. Three
tabs — **Datos**, **Abandonos**, **Tiempos** — and no predictions of any kind,
because there is neither a baseline nor a model. What it does is put the three
invariants above on screen, since none of them is visible in the shape of a frame:
which kind of result is loaded, how many riders abandoned, and how many are timed
faster than someone placed ahead of them. Loading a stage result together with a
general classification renders the refusal. Details in
[Dashboard §4](dashboard.md#4-ciclismo-the-result-contract).

## 7. What is not built yet

In order:

1. **Scoring rules** for an ordering — Spearman against the finish, top-10 hit
   rate, and a proper score for "wins the stage" as a multiclass forecast.
   Neither the lottery's set-based hit counting nor football's three-way Brier
   applies.
2. **The ranking baseline** — a start-list quality score from PCS or UCI points,
   which is the thing any model has to beat when no price exists.
3. **A rider-strength model** (a Plackett-Luce or a Bradley-Terry fit over
   results), which is the natural first model for an ordering.
4. **Evaluation** through `core/`, with the corrected verdict, exactly as the
   lottery backtest and football's market comparison do.
5. **The evaluation tabs**, once there is something to evaluate. The page exists
   and deliberately predicts nothing.

---

**Next:** [Football](football.md) · [Architecture](architecture.md) · [Development](development.md)
