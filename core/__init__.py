"""Domain-agnostic evaluation machinery.

Nothing in here knows what a lottery is, what a ball is, or what a match is.
It holds the parts of the project that are about *measuring a predictor
honestly* — walk-forward splits, testing an observed score against a null,
and correcting for multiple comparisons — so that a second domain gets the
same discipline without copying it.

The domain supplies two things: the null distribution to compare against
(a mean and variance per observation) and the scoring rule. Everything else
lives here.
"""
