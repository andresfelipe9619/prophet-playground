"""Synthetic football seasons with a known truth, so the package is runnable and testable
without anyone's private CSV — and, unlike real data, checkable against the right answer.

The lottery's synthetic data generates draws from the null hypothesis itself:
i.i.d. uniform, so the randomness tests can be checked for crying wolf. The
football equivalent has to do the opposite, because here there *is* signal.
Matches are generated from a real generative model — per-team attack and
defence strengths, a home advantage, independent Poisson goals — and the
model's own outcome probabilities are computed exactly and carried alongside
as `p_true_home` / `p_true_draw` / `p_true_away`.

That truth is what makes this data worth more than a placeholder. On real
data no one knows the right answer, so a model can only be compared to
another model. Here the right answer is known, which allows the two checks
that actually matter:

- a forecast that *is* the truth must score better than the market, and
- the market, when generated as a noisy read of the truth, must be calibrated
  but beatable — exactly the situation a real model faces.

`market_noise` is the knob. At 0 the simulated bookmaker knows the truth
exactly and is unbeatable except for its margin, which is the realistic
pessimistic case. Turning it up produces a soft market, which is what an
opening line looks like. Nothing here claims a real bookmaker is beatable at
any setting; the parameter exists so a test can distinguish "the code found
an edge" from "the code cannot find an edge that was planted".

**One known limitation, stated rather than hidden.** Goals are drawn as two
independent Poisson variables, which is the standard first model and is known
to under-produce draws and low-scoring correlated scorelines. Dixon-Coles
exists precisely to correct that, so this generator is a fair test bed for it
— but it means the data is not a substitute for real results when the
question is about score dependence itself.
"""

import numpy as np
import pandas as pd

from football.common import OUTCOMES
from football.processor import preprocess_matches

# Goals beyond this are numerically irrelevant (P < 1e-12 at these rates) and
# truncating lets the outcome probabilities be summed exactly rather than simulated.
MAX_GOALS = 15

# Calibrated against the Premier League's long-run figures rather than picked
# for convenience: these settings produce roughly 45% home wins, 25% draws,
# 30% away wins, and 1.54 / 1.19 goals per side (2.73 a game). Data that does
# not look like football would make every downstream test easier to pass and
# less informative.
DEFAULT_TEAMS = 20
BASE_GOAL_RATE = 0.10     # log-scale intercept
HOME_ADVANTAGE = 0.26     # log-scale, ~1.3x the away scoring rate
STRENGTH_SPREAD = 0.35    # sd of the per-team attack and defence effects
DEFAULT_MARGIN = 0.05     # bookmaker overround baked into the generated odds


def team_strengths(n_teams=DEFAULT_TEAMS, seed=0, spread=STRENGTH_SPREAD):
    """Latent attack and defence effects per team, on the log-goal-rate scale.

    Centred so the league average team is exactly neutral, which keeps the
    overall goal rate at `BASE_GOAL_RATE` whatever the spread is set to.
    """
    rng = np.random.default_rng(seed)
    names = [f"Team {i + 1:02d}" for i in range(n_teams)]
    attack = rng.normal(0.0, spread, n_teams)
    defence = rng.normal(0.0, spread, n_teams)
    return pd.DataFrame({
        "team": names,
        "attack": attack - attack.mean(),
        "defence": defence - defence.mean(),
    }).set_index("team")


def expected_goals(strengths, home_team, away_team, home_advantage=HOME_ADVANTAGE,
                   base_rate=BASE_GOAL_RATE):
    """(home rate, away rate) for one fixture, before any goals are drawn."""
    home, away = strengths.loc[home_team], strengths.loc[away_team]
    return (
        float(np.exp(base_rate + home_advantage + home["attack"] - away["defence"])),
        float(np.exp(base_rate + away["attack"] - home["defence"])),
    )


def outcome_probabilities(home_rate, away_rate, max_goals=MAX_GOALS):
    """Exact P(home win), P(draw), P(away win) for independent Poisson scorelines.

    Summed over the scoreline grid rather than simulated, so the "truth" this
    module hands out is the model's actual answer and carries no Monte Carlo
    error of its own — a test comparing a forecast against it is measuring the
    forecast, not the sampler.
    """
    goals = np.arange(max_goals + 1)
    home_pmf = np.exp(-home_rate) * home_rate ** goals / _factorials(goals)
    away_pmf = np.exp(-away_rate) * away_rate ** goals / _factorials(goals)
    joint = np.outer(home_pmf, away_pmf)

    draw = float(np.trace(joint))
    home_win = float(np.tril(joint, -1).sum())
    away_win = float(np.triu(joint, 1).sum())

    total = home_win + draw + away_win  # < 1 by the truncated tail; renormalise
    return np.array([home_win, draw, away_win]) / total


def _factorials(values):
    from scipy.special import factorial
    return factorial(values)


def round_robin(teams, rounds=2):
    """Every team plays every other, `rounds` times, alternating home and away."""
    fixtures = []
    for home in teams:
        for away in teams:
            if home != away:
                fixtures.append((home, away))
    return fixtures * (rounds // 2) if rounds > 2 else fixtures


def generate_matches(n_teams=DEFAULT_TEAMS, seed=0, start_date="2019-08-09",
                     home_advantage=HOME_ADVANTAGE, market_noise=0.0, margin=DEFAULT_MARGIN,
                     include_odds=True, closing_odds=True):
    """A season of results in football-data.co.uk's own column shape.

    Output goes through the real parser in `football/processor.py`, so the
    contract is exercised by every test that uses this data rather than being
    bypassed by a convenient in-memory shortcut.

    `closing_odds=False` writes the odds into the opening columns instead, so
    the opening-versus-closing guard can be tested on data that actually looks
    like a pre-2019 file.
    """
    rng = np.random.default_rng(seed)
    strengths = team_strengths(n_teams, seed=seed)
    fixtures = round_robin(list(strengths.index))
    rng.shuffle(fixtures)

    # One match day every few days, which is close enough to a real fixture
    # list for anything here and needs no calendar of its own.
    dates = pd.to_datetime(start_date) + pd.to_timedelta(
        np.repeat(np.arange(len(fixtures) // (n_teams // 2) + 1), n_teams // 2)[:len(fixtures)] * 4,
        unit="D",
    )

    rows = []
    for (home, away), date in zip(fixtures, dates):
        home_rate, away_rate = expected_goals(strengths, home, away, home_advantage)
        truth = outcome_probabilities(home_rate, away_rate)

        row = {
            "Div": "SYN",
            "Date": date.strftime("%d/%m/%Y"),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": int(rng.poisson(home_rate)),
            "FTAG": int(rng.poisson(away_rate)),
            "TrueH": truth[0], "TrueD": truth[1], "TrueA": truth[2],
        }
        row["FTR"] = OUTCOMES[int(np.argmax([row["FTHG"] > row["FTAG"],
                                             row["FTHG"] == row["FTAG"],
                                             row["FTHG"] < row["FTAG"]]))]
        if include_odds:
            prefix = ("AvgCH", "AvgCD", "AvgCA") if closing_odds else ("AvgH", "AvgD", "AvgA")
            for column, price in zip(prefix, _market_odds(truth, rng, market_noise, margin)):
                row[column] = round(price, 2)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Date", key=lambda s: pd.to_datetime(s, dayfirst=True))


def _market_odds(truth, rng, noise, margin):
    """Prices a bookmaker would post: the truth, blurred by `noise`, plus the margin.

    The blur is applied in log space and renormalised, so the result stays a
    valid probability vector however large the noise gets. The margin is then
    applied multiplicatively, which is the inverse of the default
    normalisation in `football/market.py` — so at `noise = 0` the market's
    de-margined probabilities recover the truth exactly, and a test can tell a
    broken de-margining apart from a genuinely wrong forecast.
    """
    perturbed = np.log(truth) + rng.normal(0.0, noise, len(truth)) if noise else np.log(truth)
    probabilities = np.exp(perturbed) / np.exp(perturbed).sum()
    return 1.0 / (probabilities * (1.0 + margin))


def load_sample_and_preprocess(validate=False, **kwargs):
    """Tidy matches plus the generative truth, aligned row for row.

    Returns the same shape `football.processor.load_and_preprocess` returns,
    with `p_true_home` / `p_true_draw` / `p_true_away` added. Those columns do
    not exist on real data and nothing outside tests may depend on them —
    they are the answer key, and a model that reads its own answer key is not
    being measured.
    """
    raw = generate_matches(**kwargs)
    matches = preprocess_matches(raw.drop(columns=["TrueH", "TrueD", "TrueA"]), validate=validate)

    truth = raw.sort_values("Date", key=lambda s: pd.to_datetime(s, dayfirst=True))
    for column, source in zip(("p_true_home", "p_true_draw", "p_true_away"),
                              ("TrueH", "TrueD", "TrueA")):
        matches[column] = truth[source].to_numpy()
    return matches
