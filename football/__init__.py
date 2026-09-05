"""Match-outcome forecasting for football, evaluated against the betting market.

The premise here is the opposite of the lottery's. Draws are i.i.d. by design
and nothing can beat chance; football has real, persistent signal and a model
genuinely can predict better than a coin. What does not change is the
discipline: a prediction is worth nothing until it beats the baseline it must
beat, and that baseline is stated up front rather than chosen after the fact.

The baseline is not "50/50" or "always pick the home team". It is the
**closing betting line** — the market's own probability, after the money has
moved. That bar is brutally high: it aggregates every model, every injury
report and every insider, and beating it consistently is what "having an
edge" means. A model that beats an Elo rating but loses to the closing line
has found nothing anyone will pay for.

Which is why `football/processor.py` refuses to silently mix opening and
closing odds. Opening lines are soft and beatable; closing lines are not.
Mixing them produces a baseline that looks passable and is not real — the
football counterpart of mixing the two eras of Baloto.
"""
