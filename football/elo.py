"""Elo ratings, and an ordered-logit map from a rating gap to H-D-A probabilities.

The baseline this package was missing. Elo is one number per team, nudged up or
down after every match by how surprising the result was, and it has one job
here: to be the cheap, well-understood model a serious one has to beat. Dixon-
Coles beating the market is a claim; Dixon-Coles beating the market while also
beating *this* is a much more interesting one.

**The draw is the whole difficulty.** Classic Elo answers "who wins", which is a
two-way question, and football's target is three-way. Splitting the win
probability by some fixed draw share would be a made-up number wearing a rating
system's credibility. Instead the rating gap is fed through an **ordered
logit** whose two cut points and scale are fitted by maximum likelihood on the
training matches, which is the standard treatment (Hvattum & Arntzen) and has
the property the outcome actually needs: H, D and A are ordered, so a draw sits
between the two wins by construction rather than by assumption.

**What it is not.** Elo produces a probability vector and nothing else — no
scoreline, no over/under, no both-teams-to-score, because a single strength
number cannot know how a 2-1 differs from a 1-0. That is exactly why this
package fitted a goals model first and is only adding Elo now, as a baseline,
rather than the other way round.

Ratings are walked forward **one matchday at a time**: every fixture on a date
is predicted from the ratings as they stood before that date, and the date's
updates are applied together afterwards. Updating match by match would make the
model depend on the order rows happen to sit in within a matchday — an ordering
that does not exist, since the fixtures are played simultaneously — and would
let one 3pm result inform another 3pm forecast. Nothing here can see its own
result, or its neighbour's.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from football.common import (
    MATCH_COLUMNS,  # noqa: F401  (the tidy shape this model consumes)
    OUTCOMES,
    UnknownTeamError,
    outcome_index,
)

DEFAULT_K = 20.0
DEFAULT_HOME_ADVANTAGE = 60.0  # Elo points; ~0.55 expected score for evenly matched sides
DEFAULT_RATING = 1500.0

# Below this many results the ordered logit has nothing to learn and the fit
# wanders, so the starting values stand instead. A caller at this size is not
# doing inference anyway.
_MIN_FIT_MATCHES = 20

# Starting values for (cut, log gap, log scale). The scale is in Elo points, so
# ~200 puts a 400-point gap two logits apart, which is the right order.
_INITIAL_THETA = np.array([0.0, np.log(0.75), np.log(200.0)])
_THETA_BOUNDS = ((-5.0, 5.0), (np.log(1e-3), np.log(10.0)), (np.log(20.0), np.log(2000.0)))


def _unpack(theta):
    """(cut_low, cut_high, scale), parameterised so the cuts cannot cross.

    An ordered logit is only a distribution when cut_low < cut_high; optimising
    the gap in logs makes that impossible to violate rather than something to
    check for afterwards.
    """
    cut_low = float(theta[0])
    cut_high = cut_low + float(np.exp(theta[1]))
    scale = float(np.exp(theta[2]))
    return cut_low, cut_high, scale


def _probabilities(gaps, cut_low, cut_high, scale):
    """H-D-A probabilities for an array of pre-match rating gaps, in OUTCOMES order."""
    z = np.asarray(gaps, dtype=float) / scale
    p_away = expit(cut_low - z)
    p_not_home = expit(cut_high - z)
    return np.column_stack([1.0 - p_not_home, p_not_home - p_away, p_away])


def expected_score(rating_home, rating_away, home_advantage=DEFAULT_HOME_ADVANTAGE):
    """The classic Elo expectation: the home side's share of the point, 0 to 1."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_home + home_advantage - rating_away) / 400.0))


class Elo:
    def __init__(self, ratings, k, home_advantage, cut_low, cut_high, scale):
        self.ratings = dict(ratings)
        self.k = float(k)
        self.home_advantage = float(home_advantage)
        self.cut_low = float(cut_low)
        self.cut_high = float(cut_high)
        self.scale = float(scale)

    @property
    def teams(self):
        return tuple(sorted(self.ratings))

    @classmethod
    def fit(cls, matches, k=DEFAULT_K, home_advantage=DEFAULT_HOME_ADVANTAGE,
            initial=DEFAULT_RATING):
        """Walk the ratings forward a matchday at a time, then fit the outcome map.

        Two jobs over the same matches. The rating engine is not fitted at all —
        K and the home advantage are hyperparameters, as they are everywhere Elo
        is used. What *is* fitted is the ordered logit turning a rating gap into
        three probabilities, and it is fitted on the gaps as they stood before
        each matchday, so the map is learned from genuine one-step-ahead
        forecasts rather than from hindsight.
        """
        ratings = {}
        gaps, observed = [], []

        for _, matchday in matches.groupby("ds", sort=True):
            # Every match on a date is predicted from the ratings as they stood
            # before that date, and all of the date's updates land together
            # afterwards. Updating match by match would make the model depend on
            # the order rows happen to sit in within a matchday — an ordering
            # that does not exist, since the fixtures are played at once — and
            # would let a 3pm result inform a 3pm forecast.
            adjustments = {}
            for row in matchday.itertuples():
                rating_home = ratings.setdefault(row.home_team, float(initial))
                rating_away = ratings.setdefault(row.away_team, float(initial))

                gaps.append(rating_home + home_advantage - rating_away)
                observed.append(outcome_index(row.outcome))

                expected = expected_score(rating_home, rating_away, home_advantage)
                if row.home_goals > row.away_goals:
                    actual = 1.0
                elif row.home_goals < row.away_goals:
                    actual = 0.0
                else:
                    actual = 0.5
                adjustment = k * (actual - expected)
                adjustments[row.home_team] = adjustments.get(row.home_team, 0.0) + adjustment
                adjustments[row.away_team] = adjustments.get(row.away_team, 0.0) - adjustment

            for team, adjustment in adjustments.items():
                ratings[team] += adjustment

        cut_low, cut_high, scale = cls._fit_outcome_map(np.array(gaps), np.array(observed, dtype=int))
        return cls(ratings, k, home_advantage, cut_low, cut_high, scale)

    @staticmethod
    def _fit_outcome_map(gaps, observed):
        """Maximum-likelihood cut points and scale for the ordered logit."""
        if len(gaps) < _MIN_FIT_MATCHES:
            return _unpack(_INITIAL_THETA)

        def negative_log_likelihood(theta):
            probabilities = _probabilities(gaps, *_unpack(theta))
            chosen = probabilities[np.arange(len(observed)), observed]
            # The floor keeps a single impossible-looking result from sending the
            # objective to infinity and stalling the optimiser on its first step.
            return -float(np.log(np.clip(chosen, 1e-12, None)).sum())

        best = minimize(negative_log_likelihood, _INITIAL_THETA,
                        method="L-BFGS-B", bounds=_THETA_BOUNDS)
        return _unpack(best.x if best.success else _INITIAL_THETA)

    def rating(self, team):
        if team not in self.ratings:
            raise UnknownTeamError(
                f"{team!r} does not appear in the matches this model was fitted on.")
        return self.ratings[team]

    def rating_gap(self, home_team, away_team):
        """Home rating plus home advantage, minus away rating."""
        return self.rating(home_team) + self.home_advantage - self.rating(away_team)

    def predict_outcome(self, home_team, away_team):
        """Probabilities in OUTCOMES order: (home, draw, away)."""
        gap = self.rating_gap(home_team, away_team)
        return _probabilities([gap], self.cut_low, self.cut_high, self.scale)[0]

    def predict_matches(self, matches):
        """An `(n, 3)` array of probabilities, one row per match, in OUTCOMES order."""
        gaps = [self.rating_gap(row.home_team, row.away_team) for row in matches.itertuples()]
        if not gaps:
            return np.empty((0, len(OUTCOMES)))
        return _probabilities(gaps, self.cut_low, self.cut_high, self.scale)

    def ranking(self):
        """Teams from strongest to weakest, as (team, rating) pairs."""
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)
