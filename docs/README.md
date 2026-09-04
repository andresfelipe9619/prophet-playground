# Documentation

Reference documentation for the Baloto analytics project. Start with the premise —
several design decisions look wrong until you have read it.

## Reading order

| # | Document | What it covers |
| --- | --- | --- |
| 1 | **[Domain and Premise](domain-and-premise.md)** | The game's rules, why an i.i.d. process cannot be forecast, what this project does instead and why, and the anti-patterns to avoid. **Read first.** |
| 2 | **[Architecture](architecture.md)** | Layers, data flow, the central data structures, position semantics, the two time axes, cross-cutting invariants. |
| 3 | **[Data Pipeline](data-pipeline.md)** | The data contract, the scraper and its state machine, source resolution, synthetic data. |
| 4 | **[Models](models.md)** | Every predictor: Prophet, the statsforecast trio, XGBoost, the frequency baseline. |
| 5 | **[Evaluation](evaluation.md)** | Walk-forward backtesting, the hypergeometric chance baseline, randomness tests, expected value, how to read results. |
| 6 | **[Dashboard](dashboard.md)** | Guide to each of the seven tabs and how to read them. |
| 7 | **[Development](development.md)** | Setup, the verification workflow, conventions, how to extend, gotchas. |

## Quick answers

| Question | Go to |
| --- | --- |
| Why can't a model win? | [Premise §2](domain-and-premise.md#2-the-premise-this-is-an-iid-uniform-process) |
| What is a ticket actually worth? | [Evaluation §5](evaluation.md#5-expected-value-the-one-exact-answer) |
| Why does my chi-square say "not random"? | [The sorted-data trap](domain-and-premise.md#4-the-sorted-data-trap) |
| How do I read a backtest table? | [Evaluation §7](evaluation.md#7-how-to-read-a-backtest-result) |
| Why isn't there a `freq=` anywhere? | [Architecture §5](architecture.md#5-two-time-axes) |
| How do I add a model? | [Models §7](models.md#7-adding-a-model) |
| How is anything verified without tests? | [Development §3](development.md#3-verification) |
| What broke before, so I don't repeat it? | [Evaluation §8](evaluation.md#8-known-failure-modes-we-have-already-hit) |

## Conventions

Documentation, code, docstrings and commit messages are in **English**. Dashboard
UI strings are in **Spanish**, since the dashboard is the end-user-facing product.

`CLAUDE.md` in the repository root is a condensed version of these conventions for
AI coding agents; it points here for detail.
