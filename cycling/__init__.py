"""Race-result forecasting for road cycling, from scraped results.

The third domain, and the one whose premise sits between the other two.
Baloto is i.i.d. by design, so nothing can beat chance. Football has real
signal and a market that prices it, so the bar is the closing line. Cycling
has real signal too — riders differ enormously and persistently — but the
prices are thin and the *shape* of the target is different: not three
outcomes, but an ordering of 150 riders, most of whom cannot win and a
handful of whom are missing from the finish altogether.

That makes it the domain where it is easiest to fool yourself, so the same
rule carries over: a prediction is worth nothing until it beats a baseline
named in advance. Here the honest ones, hardest first, are the **betting
market** where a price exists, and otherwise the **pre-race ranking** (UCI or
PCS points, or a start-list quality score) — a baseline that is already very
hard to beat, because "the best riders finish near the front" explains most
of what happens in a bike race.

What is emphatically *not* a baseline: a uniform draw over the start list. It
is trivially beaten, it makes any model look brilliant, and a result reported
against it says nothing at all.

This package is currently a data layer only: the contract, the scraper and
seeded synthetic races. No models, no scoring rules yet — see docs/cycling.md.
"""
