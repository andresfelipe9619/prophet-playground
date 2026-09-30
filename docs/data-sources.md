# Data sources: populating football and cycling

Baloto's data has one source and one file (see the [Data Pipeline](data-pipeline.md)).
Football and cycling do not work that way. Each has one source this repository
fetches for you, several more you can bring in by hand, and a trap in each
that a successful load will not warn you about. This page covers three things:

1. how to get real data into each domain, step by step, with the commands that
   exist today ([§1](#1-football-step-by-step), [§2](#2-cycling-step-by-step));
2. how to convert data from **any** other source into each domain's contract,
   with recipes that the test suite runs ([§3](#3-recipes-any-source-into-the-contract));
3. a catalogue of **36 remote sources**, each marked as supported, reachable
   through a recipe, or reference only ([§4](#4-source-catalogue)).

> **How far this page has been verified.** None of these sites can be reached
> from the sandbox this project is developed in, because the network policy blocks
> them. The two supported fetchers, `football/downloader.py` and
> `cycling/scraper.py`, are tested against fixtures built from each site's
> documented format, not against live pages. The catalogue was checked against
> the providers' own public descriptions and search results on **2026-09-30**.
> Coverage, prices and terms of use change. Before you depend on a source,
> open it in a browser and read its terms. Where something could not be
> confirmed, this page says so instead of guessing.

## 0. What "populated" means

Each page of the dashboard reads a folder, not a database:

| Domain | Folder | Contract owner | The page reads |
| --- | --- | --- | --- |
| Football, European mode | `exported_data/football/*.csv` | `football/processor.py` | The files ticked under **Temporadas**; by default the newest one |
| Football, Colombia mode | `exported_data/football/COL.csv` | `football/extra_processor.py` | The one file, one `League` at a time |
| Cycling results | `exported_data/cycling/*.csv` | `cycling/processor.py` | The files ticked under **Archivos**; by default the first one |
| Cycling prices | any CSV you point `cycling/prices.py` at | `cycling/prices.py` | Not on the page yet; used from Python |

`exported_data/` is gitignored, so your data never lands in a commit. With no
file present, each page falls back to synthetic data and says so on screen. If
that banner is still showing, the page has not found your files.

Every file goes through its contract **when it is loaded**, so a file that
breaks the contract is refused with a message saying why. The contracts refuse
the mistakes that are invisible in a frame's shape:

- **Football:** one odds source per frame. Closing and pre-closing prices never
  mix ([Football §3](football.md#the-trap-never-mix-opening-and-closing-odds)).
- **Cycling:** one kind of result per frame. A stage result and a general
  classification never share a `rank` column
  ([Cycling §3](cycling.md#3-the-three-traps)).

## 1. Football, step by step

### 1.1 The main leagues (supported)

[football-data.co.uk](https://www.football-data.co.uk/) publishes one CSV per
league per season, and `football/downloader.py` fetches them.

```bash
# 1. Look first: prints what would be fetched and the odds source each file resolves to.
python -m football.downloader --seasons 2019/20..2024/25 --leagues E0 --dry-run

# 2. Fetch. Files land in exported_data/football/<LEAGUE>_<yyyy>.csv, e.g. E0_2324.csv.
python -m football.downloader --seasons 2019/20..2024/25 --leagues E0

# More leagues at once; only seasons that carry closing odds.
python -m football.downloader --seasons 2019/20..2024/25 --leagues E0,SP1,I1,D1,F1 --closing-odds-only
```

**Seasons** can be written `2324`, `2023/24`, `2023-24` or `2023`. Ranges use
`..`, or `-` between two four-digit years, and both bounds are season *start*
years.

**League codes** are football-data's own. The downloader knows these 22:

| Code | League | Code | League |
| --- | --- | --- | --- |
| `E0` | England — Premier League | `I1` | Italy — Serie A |
| `E1` | England — Championship | `I2` | Italy — Serie B |
| `E2` | England — League One | `SP1` | Spain — La Liga |
| `E3` | England — League Two | `SP2` | Spain — Segunda División |
| `EC` | England — National League | `F1` | France — Ligue 1 |
| `SC0` | Scotland — Premiership | `F2` | France — Ligue 2 |
| `SC1` | Scotland — Championship | `N1` | Netherlands — Eredivisie |
| `SC2` | Scotland — League One | `B1` | Belgium — First Division A |
| `SC3` | Scotland — League Two | `P1` | Portugal — Primeira Liga |
| `D1` | Germany — Bundesliga | `T1` | Turkey — Süper Lig |
| `D2` | Germany — 2. Bundesliga | `G1` | Greece — Super League |

**Which seasons are worth having.** Closing odds, the columns ending in `C` such
as `AvgCH` and `PSCH`, exist **only from 2019/20**. Older files carry
pre-closing prices only, so a history mixing both eras cannot be loaded as one
frame. Keep them apart, or use `--closing-odds-only`.

football-data's notes say its non-`C` prices were collected on **Friday afternoons
for weekend games and Tuesday afternoons for midweek games**. They are snapshots
taken a day or more before kick-off. This project calls them "opening" prices.
Strictly, they are not the price at which the market opened, but they are just as
soft relative to the close, and that softness is what the rule depends on.

**Getting them onto the dashboard.** Pick **⚽ Fútbol** in the sidebar, choose
**Europa (football-data)** and tick the seasons under **Temporadas**. The page
refuses a selection that resolves to two different odds sources and says why.
For a market test with any resolution, tick one league's closing-odds seasons
and use **Evaluar todo el historial cargado** in tab 4. That run takes minutes,
so send it to the background ([Dashboard §7.1](dashboard.md#71-background-jobs)).

**Check the first real file.** This takes a minute:

```bash
python -c "
from football.processor import load_and_preprocess
from football.market import market_probabilities
m = market_probabilities(load_and_preprocess('exported_data/football/E0_2324.csv'))
print(m.attrs['odds_source'], m.attrs['odds_are_closing'])
print(len(m), m['ds'].min().date(), m['ds'].max().date(), round((m['outcome'] == 'H').mean(), 2))
"
```

You should see a closing source, 380 matches for a 20-team league, dates
spanning August to May, and a home-win rate near 0.45.

### 1.2 The sixteen "extra" leagues (supported, with one open question)

football-data also publishes one file per country for sixteen leagues outside
the main set. Each file stacks every season. The codes are:

| Code | League | Code | League |
| --- | --- | --- | --- |
| `ARG` | Argentina — Primera División | `MEX` | Mexico — Liga MX |
| `AUT` | Austria — Bundesliga | `NOR` | Norway — Eliteserien |
| `BRA` | Brazil — Serie A | `POL` | Poland — Ekstraklasa |
| `CHN` | China — Super League | `ROU` | Romania — Liga 1 |
| `DNK` | Denmark — Superliga | `RUS` | Russia — Premier League |
| `FIN` | Finland — Veikkausliiga | `SWE` | Sweden — Allsvenskan |
| `IRL` | Ireland — Premier Division | `SWZ` | Switzerland — Super League |
| `JPN` | Japan — J1 League | `USA` | USA — MLS |

```bash
python -m football.downloader --leagues ARG,BRA --extra --dry-run
python -m football.downloader --leagues ARG,BRA --extra        # exported_data/football/ARG.csv, BRA.csv
```

`--seasons` is ignored on this path. These files use their own contract
(`Home`/`Away`/`HG`/`AG`, several seasons per file), read by
`football/extra_processor.py`.

**Colombia is not among them.** An earlier version of this repository offered
`--leagues COL --extra`, and that could only ever return a 404. The downloader
now refuses the code by name. For Colombian data see [§1.3](#13-colombia).

**The open question: are the extra-file prices closing prices?**
`extra_processor.py` recognises `AvgH`/`PH`/`B365H` and always labels them
opening prices, so `odds_are_closing` is always `False`. Part of football-data's
own description says these leagues carry *closing* odds. If that is true, the
real files may use `PSCH`/`AvgCH`/`MaxCH`-style columns that this parser does not
look for. The downloader's validation would then report the file as having
**no odds at all**. The files cannot be opened from here, so this is unresolved.
After your first extra download, run:

```bash
head -1 exported_data/football/ARG.csv
```

- **Header has `PH`/`AvgH` and no `C` columns:** the parser matches the file, and
  the "soft baseline" warning is correct.
- **Header has `PSCH`/`AvgCH`/`MaxCH`:** the parser needs teaching, not the file
  bending. Those prices are closing prices and deserve the hard bar. Until
  `EXTRA_ODDS_SOURCES` learns the closing triples, convert the file with
  [recipe 3.1](#31-football-any-source-into-the-main-contract) into the *main*
  contract, where closing columns are recognised and the market test is the
  real one.

### 1.3 Colombia

No source this repository fetches carries Colombian football. You bring the data
in yourself, and **what you have decides which contract it belongs in**:

| You have | Put it in | What the page can then say |
| --- | --- | --- |
| Results and **closing** prices, from an odds provider that timestamps its snapshots | The **main** contract ([recipe 3.1](#31-football-any-source-into-the-main-contract)), e.g. `exported_data/football/COL1_2425.csv` | The real verdict against the closing line, in **Europa** mode, since the mode is named after the format, not the continent |
| Results and prices of unknown or early timing | The **extra** contract as `COL.csv` ([recipe 3.2](#32-colombian-results-into-the-extra-contract)) | The soft-baseline verdict in **Colombia** mode, which says on screen that no edge measured there is proven |
| Results only | Either contract, with no odds columns | Form, head-to-head and model fits; no market verdict, and the page says there is nothing to measure against |

The sources that carry Colombian Primera A are listed in [§4.1](#41-football):
API-Football, FootyStats, worldfootball.net, RSSSF and the league's own site,
Dimayor.

## 2. Cycling, step by step

### 2.1 Results from procyclingstats (supported)

`cycling/scraper.py` reads [procyclingstats.com](https://www.procyclingstats.com/)
pages at `race/<race>/<year>/stage-<n>`, `race/<race>/<year>/result` for a
one-day race, and `race/<race>/<year>/gc`.

```bash
# 1. Look first. The parser has never seen a live page.
python -m cycling.scraper --race tour-de-france --year 2024 --stages 1-21 --dry-run

# 2. Every stage of a stage race: exported_data/cycling/tour-de-france_2024_stage.csv
python -m cycling.scraper --race tour-de-france --year 2024 --stages 1-21

# A one-day race: exported_data/cycling/milano-sanremo_2024_one_day.csv
python -m cycling.scraper --race milano-sanremo --year 2024 --kind one_day

# The FINAL general classification of a finished race: ..._gc.csv
python -m cycling.scraper --race tour-de-france --year 2024 --kind gc --stage 21
```

- **The race slug is procyclingstats' own**, the path segment in the race's URL
  (`tour-de-france`, `giro-d-italia`, `vuelta-a-espana`, `milano-sanremo`,
  `paris-roubaix`, …). Copy it from the address bar instead of guessing.
- **`--kind gc` always fetches the `/gc` page**, which is the current standing
  (or the final one, once the race is over), and `--stage` only *labels* those
  rows. Run it once a race has finished, with `--stage` set to the last stage.
  Run mid-race with an earlier stage number, it stores today's standing under
  that stage's name, and nothing in the file would show the mismatch.
- **Re-running merges.** Rows are keyed on race, kind, stage and rider, and
  existing rows win, so a hand-corrected file survives a re-scrape. A stage
  that has no page yet is skipped and named. A page whose table has changed
  raises.
- **Use `--date dd/mm/yyyy`** when a page does not publish its date.
- **Be a polite client.** `--delay` defaults to 1.5 seconds between requests.
  Read the site's terms and `robots.txt` before scraping at any volume.

**How much to scrape.** The **¿Le gana al ranking?** tab needs at least five
scorable races, and at around twenty the interval is still wide ([Cycling §9](cycling.md#the-hard-part-is-the-sample-size)).
A whole Grand Tour's stages in **one** file gives 21 races of one kind, which is
the practical minimum. Stage files from several races can be ticked together;
stage files and GC files cannot.

**Check the first scrape** against the page in a browser:

- the winner's time is their real elapsed time;
- each rider below has that time *plus* their gap;
- the abandons are present, with no rank.

### 2.2 Prices (contract only; nothing fetches them)

`cycling/prices.py` defines what a price file must look like. Nothing here
fetches prices, so every cycling verdict is against the ranking until you
supply some. The rules are **one bookmaker per file** and **one market per
file**: a GC price and a stage price describe different events. See
[recipe 3.4](#34-cycling-outright-prices), and the price sources in [§4.2](#42-cycling).

### 2.3 Terrain (not in the contract)

The terrain-conditional model needs the terrain of the race being predicted,
taken from **outside** its result, because inferring it from the result is
leakage ([Cycling §11](cycling.md#11-terrain-specialisation-team-and-fatigue-cyclingfeaturespy)).
On the dashboard you enter it from the roadbook. The organisers' route pages
and the profile sites in [§4.2](#42-cycling) are where a roadbook comes from.

## 3. Recipes: any source into the contract

Each recipe starts from a small frame in the shape an API or a copied table
typically has, maps it onto the contract, writes it where the page looks, and
loads it back through the owning module, which is the check that it worked.
Replace the literal frame with whatever your source gives you.
**`tests/test_data_sources_recipes.py` runs every code block below**, so a recipe
that stops matching its contract fails CI rather than a reader.

### 3.1 Football: any source into the main contract

Required: `Date` (dd/mm/yyyy), `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`. The odds
are optional, and **the column names are the claim you are making about when the
price was taken**:

| Your prices are | Name the triple | Resolves to |
| --- | --- | --- |
| A market average **at the close** | `AvgCH`, `AvgCD`, `AvgCA` | `market_closing_average` (closing) |
| Pinnacle **at the close** | `PSCH`, `PSCD`, `PSCA` | `pinnacle_closing` (closing) |
| Bet365 **at the close** | `B365CH`, `B365CD`, `B365CA` | `bet365_closing` (closing) |
| A market average **before** the close | `AvgH`, `AvgD`, `AvgA` | `market_opening_average` (soft) |
| Pinnacle / Bet365 **before** the close | `PSH`… / `B365H`… | `pinnacle_opening` / `bet365_opening` (soft) |

Nothing in a file can prove a `C` column was taken at the close. Naming a
Friday snapshot `AvgCH` would make every verdict built on it a claim against a
bar that was never there. "At the close" means the last snapshot before
kick-off. If your provider timestamps snapshots, take the last one before the
start; if it does not, the prices are not closing prices.

```python
import os
import pandas as pd

# What a provider handed you, in its own shape: one row per match, closing prices.
raw = pd.DataFrame({
    "kickoff": ["2024-08-16T19:00:00Z", "2024-08-17T11:30:00Z", "2024-08-17T14:00:00Z"],
    "home": ["Manchester United", "Ipswich", "Arsenal"],
    "away": ["Fulham", "Liverpool", "Wolves"],
    "home_score": [1, 0, 2],
    "away_score": [0, 2, 0],
    # The market average at kick-off. Only name these *C* columns if they were
    # taken at the close — the contract cannot tell, and that is the whole trap.
    "close_home": [1.60, 4.70, 1.30], "close_draw": [4.20, 4.00, 5.75], "close_away": [5.50, 1.70, 10.0],
})

season = pd.DataFrame({
    "Date": pd.to_datetime(raw["kickoff"]).dt.strftime("%d/%m/%Y"),
    "HomeTeam": raw["home"], "AwayTeam": raw["away"],
    "FTHG": raw["home_score"], "FTAG": raw["away_score"],
    "AvgCH": raw["close_home"], "AvgCD": raw["close_draw"], "AvgCA": raw["close_away"],
})
os.makedirs("exported_data/football", exist_ok=True)
season.to_csv("exported_data/football/API_2425.csv", index=False)

from football.processor import load_seasons
matches = load_seasons(["exported_data/football/API_2425.csv"])
print(matches.attrs)            # {'odds_source': 'market_closing_average', 'odds_are_closing': True}
```

**Team names must be consistent across every file you load together.**
"Man United" in one season and "Manchester United" in the next are two teams
to every model here. Normalise names before writing, with one mapping table
per source.

### 3.2 Colombian results into the extra contract

Required: `Date`, `Home`, `Away`, `HG`, `AG`, plus `League`, because the Colombia
page loads one league at a time. The odds are optional and are read as
**opening** prices whatever their real timing, so the page will always call the
baseline soft. If you have genuine closing prices, use recipe 3.1 instead.

```python
import os
import pandas as pd

raw = pd.DataFrame({
    "date": ["2024-01-20", "2024-01-21"],
    "competition": ["Colombia Primera A", "Colombia Primera A"],
    "season": [2024, 2024],
    "home": ["Millonarios", "Junior"], "away": ["Atlético Nacional", "América de Cali"],
    "hg": [2, 0], "ag": [1, 0],
    "odds_h": [2.30, 2.05], "odds_d": [3.05, 3.15], "odds_a": [3.10, 3.70],
})

col = pd.DataFrame({
    "Country": "Colombia",
    "League": raw["competition"],
    "Season": raw["season"],
    "Date": pd.to_datetime(raw["date"]).dt.strftime("%d/%m/%Y"),
    "Home": raw["home"], "Away": raw["away"],
    "HG": raw["hg"], "AG": raw["ag"],
    "AvgH": raw["odds_h"], "AvgD": raw["odds_d"], "AvgA": raw["odds_a"],
})
os.makedirs("exported_data/football", exist_ok=True)
col.to_csv("exported_data/football/COL.csv", index=False)

from football.extra_processor import load_extra
matches = load_extra("exported_data/football/COL.csv", league="Colombia Primera A")
print(matches.attrs)            # odds_are_closing is False, whatever the prices really were
```

Colombia plays **Apertura and Finalización** tournaments within one calendar
year, including play-off rounds. Keep them in one `League` value so the
history is continuous. If you split them, loading one tournament at a time
throws away half the data.

### 3.3 Cycling: any source into the result contract

Required: `Date` (dd/mm/yyyy), `Race` (a slug), `Kind` (`stage`, `one_day` or
`gc`), `Stage` (blank for a one-day race), `Rank` (blank for every
non-finisher), `Rider`, `Team`, `Status` (`FIN`, `DNF`, `DNS`, `DSQ`, `OTL`, `NR`)
and `TimeSeconds`. **One kind per file.**

Most sources publish the winner's time and everyone else's *gap*. The contract
holds **totals**, so add each gap to the winner's time. Storing gaps produces
a frame that looks normal and ranks the field backwards, and
`time_order_violations` exists to catch exactly that. Keep the abandons,
because a forecast that backed a rider who climbed off is charged for it.

```python
import os
import pandas as pd

# A stage as a provider publishes it: the winner's elapsed time, everyone else's gap.
raw = pd.DataFrame({
    "pos": ["1", "2", "3", "DNF"],
    "rider": ["POGAČAR Tadej", "VINGEGAARD Jonas", "EVENEPOEL Remco", "ROGLIČ Primož"],
    "team": ["UAE Team Emirates", "Team Visma | Lease a Bike", "Soudal Quick-Step", "Red Bull - BORA"],
    "time_or_gap": ["4:00:52", "+0:21", "+0:21", ""],
})

def seconds(clock):
    total = 0
    for part in clock.lstrip("+").split(":"):
        total = total * 60 + int(part)
    return total

winner = seconds(raw.loc[0, "time_or_gap"])
finished = raw["pos"].str.isdigit()
stage = pd.DataFrame({
    "Date": "13/07/2024", "Race": "tour-de-france", "Kind": "stage", "Stage": 14,
    "Rank": pd.to_numeric(raw["pos"].where(finished), errors="coerce").astype("Int64"),
    "Rider": raw["rider"], "Team": raw["team"],
    "Status": raw["pos"].where(~finished, "FIN"),
    # Totals, never gaps: the winner's time plus each gap.
    "TimeSeconds": [float(winner) if i == 0 else (winner + seconds(g) if f else None)
                    for i, (g, f) in enumerate(zip(raw["time_or_gap"], finished))],
})
os.makedirs("exported_data/cycling", exist_ok=True)
stage.to_csv("exported_data/cycling/tour-de-france_2024_stage.csv", index=False)

from cycling.processor import load_races, time_order_violations
results = load_races(["exported_data/cycling/tour-de-france_2024_stage.csv"])
print(results.attrs, int(time_order_violations(results).sum()))   # {'result_kind': 'stage'} 0
```

**Rider names are identities.** "Pogačar", "POGAČAR Tadej" and "Tadej Pogacar"
are three riders to the model. Use one source's spelling throughout, or map
every other source onto it.

### 3.4 Cycling outright prices

Required: `Date`, `Race`, `Kind`, `Rider`, `Odds` (decimal) and `Book`; `Stage` is
optional. **One book and one market per file.** A second bookmaker is a second
file, and so is a stage market beside a GC market. Record a price file once, at
a fixed time before the start, and never overwrite it: it is the bar, and a bar
moved after the race is not one.

```python
import os
import pandas as pd

# One book, one market, one moment. A second bookmaker is a second file.
prices = pd.DataFrame({
    "Date": "29/06/2024", "Race": "tour-de-france", "Kind": "gc", "Stage": "",
    "Rider": ["POGAČAR Tadej", "VINGEGAARD Jonas", "EVENEPOEL Remco", "ROGLIČ Primož"],
    "Odds": [1.80, 3.50, 8.00, 9.00],
    "Book": "bookmaker-name",
})
os.makedirs("exported_data/cycling/prices", exist_ok=True)
prices.to_csv("exported_data/cycling/prices/tour-de-france_2024_gc_bookmaker-name.csv", index=False)

from cycling.prices import load_prices
book = load_prices("exported_data/cycling/prices/tour-de-france_2024_gc_bookmaker-name.csv")
print(book[["rider", "odds", "book"]].to_string(index=False))
```

A book quotes the riders it chooses, not the whole start list, so the riders it
never priced are filled with the longest price it did quote
([Cycling §10](cycling.md#the-quoted-field-is-not-the-field)). The rider
spellings must match the result files' spellings exactly.

## 4. Source catalogue

**Status** says what this repository does with each source:

- **Supported:** a command here fetches it.
- **Recipe:** you download or export it yourself, then convert it with a recipe
  from §3.
- **Reference:** useful for checking, for features, or as another model's
  baseline, but not a source of the frames the contracts hold.

**Access** summarises what the provider publishes about cost and keys; check it
before you rely on it. Scraping a site that does not publish a download or an API
is governed by that site's terms of use.

### 4.1 Football

| # | Source | What it has | Access | Status | Note |
| --- | --- | --- | --- | --- | --- |
| 1 | [football-data.co.uk](https://www.football-data.co.uk/data.php) — main leagues | Results, match stats and 1X2 odds, one CSV per league per season, 22 leagues, back to 1993/94 | Free CSV | **Supported** (`football.downloader`) | Closing odds only from 2019/20; earlier prices are Friday/Tuesday snapshots |
| 2 | [football-data.co.uk](https://www.football-data.co.uk/all_new_data.php) — 16 extra leagues | Results and odds for ARG, AUT, BRA, CHN, DNK, FIN, IRL, JPN, MEX, NOR, POL, ROU, RUS, SWE, SWZ, USA | Free CSV | **Supported** (`--extra`) | No Colombia; odds columns unverified ([§1.2](#12-the-sixteen-extra-leagues-supported-with-one-open-question)) |
| 3 | [API-Football](https://www.api-football.com/) (api-sports.io) | Fixtures, results, lineups, statistics and pre-match odds for many hundreds of leagues, **including Colombia's Primera A** | Free tier with a daily request quota; paid plans | Recipe 3.1 / 3.2 | Store odds with their snapshot time; only the last one before kick-off is a close |
| 4 | [football-data.org](https://www.football-data.org/) | Fixtures, results and standings over a REST API | Free tier for a set of top competitions (API key); paid for more | Recipe 3.1, results only | No odds, so it trains models but cannot give a market verdict |
| 5 | [The Odds API](https://the-odds-api.com/) | Odds from many bookmakers over a REST API, with historical snapshots | Free tier with a monthly quota; historical data on paid plans | Recipe 3.1 (odds) | Build a closing line from the last snapshot before kick-off; join to results by date and teams |
| 6 | [Betfair Exchange historical data](https://historicdata.betfair.com/) | Exchange price streams for past markets | Betfair account; a free basic tier and paid finer-grained tiers | Recipe 3.1 (odds) | Last traded price before the off is a sharp closing reference; exchange odds carry commission, not an overround |
| 7 | [OddsPortal](https://www.oddsportal.com/) | Historical odds per bookmaker, including closing prices, for many leagues | Website | Reference | No export; automated collection is restricted by its terms. Useful to spot-check a close |
| 8 | [Understat](https://understat.com/) | Shot-level xG for the English, Spanish, German, Italian, French and Russian top flights, from 2014/15 | Website (data embedded in pages) | Reference | Feature source for roadmap item 11 (xG strengths); unofficial access |
| 9 | [FBref](https://fbref.com/) | Team and player statistics, fixtures and results | Website, rate-limited | Reference | Coverage of advanced metrics has changed over time, so check what a league has before planning around it; automated access is restricted |
| 10 | [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Free event data including xG for selected competitions and seasons | Free, attribution required | Reference | Event-level; for features and model research, not a season's match list |
| 11 | [Wyscout public dataset](https://figshare.com/collections/Soccer_match_event_dataset/4415000) (Pappalardo et al., 2019) | Events for 2017/18 in five top leagues, World Cup 2018 and Euro 2016 | Free (research licence) | Reference | One season; good for testing features, not for a verdict |
| 12 | [openfootball](https://github.com/openfootball) | Fixtures and results for many leagues and tournaments in plain text | Free, public domain | Recipe 3.1, results only | No odds |
| 13 | [ClubElo](http://clubelo.com/) | Elo ratings for European clubs, daily, back decades; a CSV API (`api.clubelo.com/<date>`) | Free | Reference | An **external** Elo to sanity-check `football/elo.py` against; a model, not a market |
| 14 | [Transfermarkt datasets](https://github.com/dcaribou/transfermarkt-datasets) | Games, appearances, squads and market values scraped from Transfermarkt, refreshed regularly | Free (community dataset) | Recipe 3.1, results only | Squad value is a candidate feature for promoted-team priors |
| 15 | [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer) (Kaggle) | About 25,000 matches from 11 countries, 2008–2016, with several bookmakers' odds, as SQLite | Free (Kaggle account) | Recipe 3.1 | Odds timing undocumented, so name them as pre-closing (`AvgH`…) |
| 16 | [FiveThirtyEight club soccer predictions](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) | SPI ratings and match probabilities | Free, archived (no longer updated) | Reference | A historical **model** baseline to compare against; not a market |
| 17 | [worldfootball.net](https://www.worldfootball.net/) | Results and tables for leagues worldwide, including Colombia, far back | Website | Recipe 3.2, results only | Good for back-filling Colombian history; check its terms |
| 18 | [RSSSF](https://www.rsssf.org/) | Historical results and tables archive, many countries including Colombia | Free, plain HTML | Reference | For verifying old results, not for bulk loading |
| 19 | [FootyStats](https://footystats.org/download-stats-csv) | CSV downloads of results and statistics for many leagues, **including Colombia's Primera A** | Paid for full downloads | Recipe 3.1 / 3.2 | Check what its odds columns mean before naming them closing |
| 20 | [Dimayor](https://dimayor.com.co/) | The Colombian league's own fixtures and results | Website | Reference | The authority to check Colombian results against |
| 21 | [soccerdata](https://github.com/probberechts/soccerdata) (Python package) | One interface to ClubElo, ESPN, FBref, football-data, Sofascore, Understat, WhoScored and others | Free, open source | Recipe 3.1 | A fetching layer, not a source; each site's terms still apply |
| 22 | [Sofascore](https://www.sofascore.com/) / [FotMob](https://www.fotmob.com/) | Live and historical results, lineups and ratings for a very wide range of leagues | Website/app; no public API | Reference | Unofficial endpoints only; terms restrict automated use |

### 4.2 Cycling

| # | Source | What it has | Access | Status | Note |
| --- | --- | --- | --- | --- | --- |
| 23 | [ProCyclingStats](https://www.procyclingstats.com/) | Stage, one-day and GC results, start lists, rider profiles and PCS points | Website | **Supported** (`cycling.scraper`) | Tables found by their headers; `--kind gc` fetches the current or final standing ([§2.1](#21-results-from-procyclingstats-supported)) |
| 24 | [procyclingstats](https://pypi.org/project/procyclingstats/) (Python package) | An unofficial scraper library for the same site | Free, open source | Recipe 3.3 | A second parser to cross-check this repository's own against |
| 25 | [FirstCycling](https://firstcycling.com/) | Results, start lists and rankings, including smaller races | Website | Recipe 3.3 | A second results source to cross-check a PCS scrape; an unofficial Python wrapper exists on GitHub |
| 26 | [UCI](https://www.uci.org/) (rankings and [DataRide](https://dataride.uci.ch/) results) | Official UCI points rankings and results | Website | Recipe 3.3 / Reference | UCI points are the official input for a pre-race ranking baseline (`worths_from_points`) |
| 27 | [CyclingRanking](https://www.cyclingranking.com/) | Historical rider rankings and results over a long history | Website | Reference | Long-run form for priors |
| 28 | [Tour de France](https://www.letour.fr/) (official) | Official stage results, classifications and **stage profiles** | Website | Recipe 3.3 / roadbook | The organiser's stage type (flat, hilly, mountain, time trial) is the roadbook terrain the model needs |
| 29 | [Giro d'Italia](https://www.giroditalia.it/) (official) | Official results and stage profiles | Website | Recipe 3.3 / roadbook | As above |
| 30 | [La Vuelta](https://www.lavuelta.es/) (official) | Official results and stage profiles | Website | Recipe 3.3 / roadbook | As above |
| 31 | [La Flamme Rouge](https://www.la-flamme-rouge.eu/) | Stage profiles and climb details for many races | Website | Roadbook | Terrain from outside the result, which is what the leakage rule requires |
| 32 | [ClimbFinder](https://climbfinder.com/) | Climb profiles, gradients and lengths | Website | Roadbook | Detail for classifying a stage as a climbing day |
| 33 | [TidyTuesday — Tour de France](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-07) | Historical Tour winners and stage results as CSV | Free | Recipe 3.3 | History only (the 2020 release), for testing on a long record |
| 34 | [Wikipedia](https://en.wikipedia.org/) / [Wikidata](https://www.wikidata.org/) | Stage winners, classifications and route summaries per edition | Free | Reference | Cross-check a scrape; incomplete below the podium |
| 35 | [Oddschecker](https://www.oddschecker.com/cycling) | Current outright odds across many bookmakers, one column per book | Website | Recipe 3.4 | Record once, before the start, **one file per bookmaker**; not a historical archive; terms restrict automated collection |
| 36 | Bookmakers' own outright markets (e.g. Pinnacle, Bet365, Betfair Exchange) | Winner, podium and stage-winner prices for major races | Account/website | Recipe 3.4 | The only route to a cycling **market** baseline; the exchange's historical files may include cycling markets, so check which sports a plan covers |

### What is still missing

- **A historical cycling price archive.** Nothing in the catalogue is a free,
  timestamped history of outright odds. A cycling market verdict therefore starts
  from prices you record yourself from today on, which is also the only kind
  of record the [registry](registry.md) accepts.
- **Colombian closing odds from a free source.** The odds providers above carry
  Colombia, but none of them publishes closing prices for free.
- **A live check of any of this from the development sandbox.** The first real
  download is the verification: run the checks in [§1.1](#11-the-main-leagues-supported)
  and [§2.1](#21-results-from-procyclingstats-supported) before trusting it.

---

**Next:** [Football](football.md) · [Cycling](cycling.md) · [Data Pipeline](data-pipeline.md)
