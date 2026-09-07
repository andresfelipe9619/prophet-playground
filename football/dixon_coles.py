"""Dixon-Coles: two Poisson goal counts with a low-score dependence correction.

The plain independent-Poisson model is the standard first pass at football
scores and is known to under-produce draws and low-scoring correlated
scorelines. Dixon & Coles (1997) fix that with one extra parameter, rho, that
reweights the four cells 0-0, 1-0, 0-1, 1-1. From one fitted model every
market falls out: 1X2, correct score, over/under, both teams to score.

Parameters, fitted by maximum likelihood over past matches:

    lambda_home = exp(mu + home_advantage + attack[home] + defence[away])
    lambda_away = exp(mu +                  attack[away] + defence[home])

`attack` and `defence` are each constrained to sum to zero for identifiability
against the intercept `mu`. `half_life` (in days) applies an exponential
time-decay weight to older matches so recent form counts for more; None means
every match counts equally.

This model does NOT know about the market. It produces a forecast;
`football/evaluation.py` is what decides whether that forecast is worth
anything, by scoring it against the closing line.
"""

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from football.common import OUTCOMES  # noqa: F401  (kept for the order contract)

# rho must keep every tau cell positive over realistic lambdas; this bound is
# wide enough for real data and narrow enough that the optimiser stays sane.
_RHO_BOUNDS = (-0.4, 0.4)


class UnknownTeamError(ValueError):
    """A prediction was asked for a team the model was never fitted on."""


def _tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Dixon-Coles low-score correction, broadcast over goal-count arrays."""
    h = np.asarray(home_goals)
    a = np.asarray(away_goals)
    out = np.ones(np.broadcast(h, a).shape, dtype=float)
    out = np.where((h == 0) & (a == 0), 1.0 - lambda_home * lambda_away * rho, out)
    out = np.where((h == 0) & (a == 1), 1.0 + lambda_home * rho, out)
    out = np.where((h == 1) & (a == 0), 1.0 + lambda_away * rho, out)
    out = np.where((h == 1) & (a == 1), 1.0 - rho, out)
    return out


def independent_poisson_matrix(lambda_home, lambda_away, max_goals=10):
    """The rho = 0 scoreline grid: the plain independent-Poisson baseline."""
    goals = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))


class DixonColes:
    def __init__(self, teams, mu, home_advantage, attack, defence, rho):
        self.teams = tuple(teams)
        self._index = {team: i for i, team in enumerate(self.teams)}
        self.params = {
            "mu": float(mu),
            "home_advantage": float(home_advantage),
            "rho": float(rho),
            "attack": dict(attack),
            "defence": dict(defence),
        }

    # -- construction -----------------------------------------------------
    @classmethod
    def _from_params(cls, teams, mu, home_advantage, attack, defence, rho):
        return cls(teams, mu, home_advantage, attack, defence, rho)

    @classmethod
    def fit(cls, matches, half_life=None, max_iter=200):
        """Maximum-likelihood fit on a tidy match frame."""
        teams = tuple(sorted(set(matches["home_team"]) | set(matches["away_team"])))
        idx = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        home_idx = matches["home_team"].map(idx).to_numpy()
        away_idx = matches["away_team"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy(dtype=float)
        ag = matches["away_goals"].to_numpy(dtype=float)

        if half_life:
            age_days = (matches["ds"].max() - matches["ds"]).dt.days.to_numpy(dtype=float)
            weights = 0.5 ** (age_days / float(half_life))
        else:
            weights = np.ones(len(matches))

        # Parameter vector: [mu, home_adv, attack[0..n-2], defence[0..n-2], rho].
        # The last attack/defence value is fixed at minus the sum of the rest.
        def unpack(theta):
            mu, home_adv = theta[0], theta[1]
            a_free = theta[2:2 + n - 1]
            d_free = theta[2 + n - 1:2 + 2 * (n - 1)]
            rho = theta[-1]
            attack = np.append(a_free, -a_free.sum())
            defence = np.append(d_free, -d_free.sum())
            return mu, home_adv, attack, defence, rho

        def neg_log_likelihood(theta):
            mu, home_adv, attack, defence, rho = unpack(theta)
            lam_h = np.exp(mu + home_adv + attack[home_idx] + defence[away_idx])
            lam_a = np.exp(mu + attack[away_idx] + defence[home_idx])
            ll = (poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a)
                  + np.log(np.clip(_tau(hg, ag, lam_h, lam_a, rho), 1e-12, None)))
            return -float(np.sum(weights * ll))

        theta0 = np.concatenate([[np.log(max(hg.mean(), 0.5)), 0.25],
                                 np.zeros(n - 1), np.zeros(n - 1), [0.0]])
        bounds = [(None, None), (None, None)] + [(None, None)] * (2 * (n - 1)) + [_RHO_BOUNDS]
        result = minimize(neg_log_likelihood, theta0, method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": max_iter})
        if not result.success:
            warnings.warn(f"Dixon-Coles fit did not converge: {result.message}", stacklevel=2)

        mu, home_adv, attack, defence, rho = unpack(result.x)
        if abs(rho - _RHO_BOUNDS[0]) < 1e-4 or abs(rho - _RHO_BOUNDS[1]) < 1e-4:
            warnings.warn(
                f"rho hit its bound ({rho:.3f}); the low-score correction is at its limit.",
                stacklevel=2,
            )
        return cls(
            teams=teams, mu=mu, home_advantage=home_adv, rho=rho,
            attack={t: float(attack[i]) for t, i in idx.items()},
            defence={t: float(defence[i]) for t, i in idx.items()},
        )

    # -- prediction -----------------------------------------------------
    def _team(self, name):
        if name not in self._index:
            raise UnknownTeamError(
                f"{name!r} is not in this model. Fitted teams: {', '.join(self.teams)}."
            )
        return name

    def _lambdas(self, home_team, away_team):
        p = self.params
        lam_h = np.exp(p["mu"] + p["home_advantage"]
                       + p["attack"][home_team] + p["defence"][away_team])
        lam_a = np.exp(p["mu"] + p["attack"][away_team] + p["defence"][home_team])
        return float(lam_h), float(lam_a)

    def _grid(self, lambda_home, lambda_away, max_goals=10):
        grid = independent_poisson_matrix(lambda_home, lambda_away, max_goals)
        goals = np.arange(max_goals + 1)
        h, a = np.meshgrid(goals, goals, indexing="ij")
        grid = grid * _tau(h, a, lambda_home, lambda_away, self.params["rho"])
        return np.clip(grid, 0.0, None)

    def scoreline_matrix(self, home_team, away_team, max_goals=10):
        lam_h, lam_a = self._lambdas(self._team(home_team), self._team(away_team))
        grid = self._grid(lam_h, lam_a, max_goals)
        return grid / grid.sum()

    def predict_outcome(self, home_team, away_team):
        grid = self.scoreline_matrix(home_team, away_team)
        return np.array([
            float(np.tril(grid, -1).sum()),   # home win
            float(np.trace(grid)),            # draw
            float(np.triu(grid, 1).sum()),    # away win
        ])

    def most_likely_scores(self, home_team, away_team, n=5):
        grid = self.scoreline_matrix(home_team, away_team)
        flat = np.dstack(np.unravel_index(np.argsort(grid, axis=None)[::-1], grid.shape))[0]
        return [((int(h), int(a)), float(grid[h, a])) for h, a in flat[:n]]

    def over_under(self, home_team, away_team, line=2.5):
        grid = self.scoreline_matrix(home_team, away_team)
        goals = np.arange(grid.shape[0])
        totals = goals[:, None] + goals[None, :]
        p_over = float(grid[totals > line].sum())
        return p_over, 1.0 - p_over

    def both_teams_to_score(self, home_team, away_team):
        grid = self.scoreline_matrix(home_team, away_team)
        return float(grid[1:, 1:].sum())

    def predict_matches(self, matches):
        return np.array([
            self.predict_outcome(row.home_team, row.away_team)
            for row in matches.itertuples()
        ])
