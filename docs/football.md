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

The "extra league" files for the rest of the world (`new/ARG.csv` and friends)
use a **different contract** — `Home`/`Away`/`HG`/`AG`, several leagues stacked
in one file — and nothing in `football/` reads them, so the downloader refuses
those codes by name rather than fetching something the processor would reject.

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

`streamlit run dashboard/app.py`, then pick **⚽ Fútbol** in the sidebar. Three
tabs — **Datos**, **Mercado**, **Resultados** — and the limit is stated on all of
them: there is no model here and no scoring rule, so **nothing on that page
compares a forecast against the market**. It shows which odds source a file
resolved to, how big the margin is, that the market is calibrated, and how the
three de-margining methods differ on one match. Selecting season files that resolve
to different odds sources renders the refusal instead of merging them. Details in
[Dashboard §3](dashboard.md#3-fútbol-the-data-contract-and-the-market).

## 7. What is not built yet

This is the data layer only. Still to come, in order:

1. **Scoring rules** — Brier / ranked probability score for the three-way
   outcome, log-loss for binaries. Set-based hit counting does not apply here.
2. **Elo**, as a cheap model and a second baseline.
3. **Dixon-Coles** — attack/defence strengths with the low-score correction,
   from which every market (1X2, over/under, both teams to score, correct
   score) falls out of one fitted model.
4. **Evaluation** through `core/`, comparing every model against the market
   with the corrected verdict, exactly as the lottery backtest does.
5. **The evaluation tabs**, once there is something to evaluate. The page exists;
   what it lacks is a model column, and it will not get one before there is a
   scoring rule behind it.

---

**Next:** [Cycling](cycling.md) · [Architecture](architecture.md) · [Evaluation](evaluation.md) · [Development](development.md)
