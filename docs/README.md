# Documentation

Reference documentation for this project. It began as Baloto lottery analysis and
now spans three domains: `lottery/`, where nothing can beat chance, and `football/`
and `cycling/`, where something can. Start with the premise — several design
decisions look wrong until you have read it, and the two sibling domains only
make sense against it.

## Reading order

| # | Document | What it covers |
| --- | --- | --- |
| 1 | **[Domain and Premise](domain-and-premise.md)** | The game's rules, why an i.i.d. process cannot be forecast, what this project does instead and why, and the anti-patterns to avoid. **Read first.** |
| 2 | **[Architecture](architecture.md)** | Layers, data flow, the central data structures, position semantics, the two time axes, cross-cutting invariants. |
| 3 | **[Data Pipeline](data-pipeline.md)** | The data contract, the scraper and its state machine, source resolution, synthetic data, and where the other two domains get their data. |
| 4 | **[Models](models.md)** | Every predictor: Prophet, the statsforecast trio, XGBoost, the frequency baseline. |
| 5 | **[Tickets](tickets.md)** | Generating plays, checking them against real draws, and measuring whether a strategy beats picking at random. |
| 6 | **[Evaluation](evaluation.md)** | Walk-forward backtesting, the hypergeometric chance baseline, randomness tests, expected value, how to read results. |
| 7 | **[Jackpot Splitting](jackpot-splitting.md)** | The only lever that changes anything: unpopular combinations do not win more often, they split less. |
| 8 | **[Power and Sensitivity](power-and-sensitivity.md)** | What an edge would have to look like for this much data to see it, and whether the tests fire on a planted one. The two questions that make a null result mean something. |
| 9 | **[Registry](registry.md)** | Pre-registration: predictions written down before the draw, which is the one thing retrospective analysis can never be. |
| 10 | **[Football](football.md)** | The second domain. The premise inverted, the closing line as the baseline, the odds contract and its opening/closing trap, synthetic seasons with a known answer, Dixon-Coles and Elo, pooling a model with the market, the multi-model backtest, and edge and staking. |
| 11 | **[Cycling](cycling.md)** | The third domain. An ordering rather than an outcome, the result contract and its three traps, the scraper, the ranking baseline, the Plackett-Luce scoring rule and model, and the paired evaluation. |
| 12 | **[Dashboard](dashboard.md)** | One app, three domains: the sidebar selector, Baloto's ten tabs, football's five, cycling's five, and the four layers of explanation that make them readable without a statistics background. |
| 13 | **[Development](development.md)** | Setup, the test suite, the verification workflow, conventions, how to extend, gotchas. |

## Quick answers

| Question | Go to |
| --- | --- |
| Why can't a model win? | [Premise §2](domain-and-premise.md#2-the-premise-this-is-an-iid-uniform-process) |
| What is a ticket actually worth? | [Evaluation §5](evaluation.md#5-expected-value-the-one-exact-answer) |
| Why does my chi-square say "not random"? | [The sorted-data trap](domain-and-premise.md#4-the-sorted-data-trap) |
| How do I read a backtest table? | [Evaluation §7](evaluation.md#7-how-to-read-a-backtest-result) |
| "No model beat chance" — how much does that prove? | [Power §1](power-and-sensitivity.md#1-power-lotteryanalysispowerpy) |
| How do I know the tests aren't just blind? | [Sensitivity §2](power-and-sensitivity.md#2-sensitivity-lotteryanalysissensitivitypy) |
| How many draws would I need to prove an edge? | [Power: how much history](power-and-sensitivity.md#how-much-history-each-edge-would-need) |
| Can I improve anything at all by choosing numbers? | [Jackpot Splitting](jackpot-splitting.md) |
| How do I prove a prediction was made in advance? | [Registry](registry.md) |
| Why isn't there a `freq=` anywhere? | [Architecture §5](architecture.md#5-two-time-axes) |
| How do I add a model? | [Models §7](models.md#7-adding-a-model) |
| How do I generate and check tickets? | [Tickets](tickets.md) |
| My strategy beat chance once — is it real? | [Tickets §5](tickets.md#5-measuring-the-accuracy-system) |
| What does the test suite actually pin? | [Development §3.1](development.md#31-the-test-suite) |
| What runs in CI, and why not the full requirements? | [Development §3.5](development.md#35-continuous-integration) |
| What broke before, so I don't repeat it? | [Evaluation §8](evaluation.md#8-known-failure-modes-we-have-already-hit) |
| What does a football model have to beat? | [Football §2](football.md#2-the-baseline-is-the-closing-line) |
| Why are odds not probabilities? | [Football: odds are not probabilities](football.md#odds-are-not-probabilities) |
| Why won't it load two seasons together? | [Football §3](football.md#the-trap-never-mix-opening-and-closing-odds) |
| How do I download real football seasons? | [Football §5](football.md#5-getting-real-data) |
| How do I scrape cycling results? | [Cycling §4](cycling.md#4-the-scraper) |
| Why can't I mix stage results and a GC? | [Cycling §3](cycling.md#one-kind-of-result-per-frame) |
| Why are the abandons still in my frame? | [Cycling §3](cycling.md#non-finishers-stay-in-the-frame) |
| How do I switch the dashboard to football or cycling? | [Dashboard §0](dashboard.md#0-the-three-domains) |
| How does the football model get scored against the market? | [Football §8](football.md#8-the-model-and-the-two-team-view) · [Evaluation §9](evaluation.md#9-football-dixon-coles-vs-the-market) |
| Why add Elo now, after saying it was pointless? | [Football §9](football.md#9-elo-the-cheap-baseline) |
| Does my model know anything the market doesn't? | [Football §10](football.md#10-pooling-with-the-market) |
| My model says 38% and the market says 35% — is that a bet? | [Football §11](football.md#the-two-bars-which-are-different) |
| Why won't the dashboard show me a stake? | [Football §11](football.md#kelly-is-not-a-safety-feature) |
| What does a cycling forecast have to beat? | [Cycling §6](cycling.md#6-the-baseline-cyclingbaselinepy) |
| Why isn't a uniform draw over the start list a baseline? | [Cycling §6](cycling.md#a-uniform-draw-is-not-a-baseline) |
| How do you score a finishing order? | [Cycling §7](cycling.md#7-scoring-an-ordering-cyclingscoringpy) |
| Why is the sum of the balls a bell curve if all tickets are equal? | [Evaluation §4.1](evaluation.md#41-the-order-agnostic-summaries) |
| I don't know what a p-value is — where do I start? | The **Glosario** in the dashboard sidebar · [Dashboard §5](dashboard.md#5-the-four-layers-of-explanation) |
| How do I use the Colombian league data? | [Data Pipeline §5.1](data-pipeline.md#51-footballs-extra-files) |

## Conventions

Documentation, code, docstrings and commit messages are in **English**. Dashboard
UI strings are in **Spanish**, since the dashboard is the end-user-facing product.

`CLAUDE.md` in the repository root is a condensed version of these conventions for
AI coding agents; it points here for detail.
