"""Outcome semantics and the data locations. Everything else imports from here.

A football result, for prediction purposes, is one of three mutually
exclusive outcomes: home win, draw, away win. That is the whole target
space, and pinning it in one place is what stops a `["H", "A", "D"]` here and
a `["H", "D", "A"]` there from quietly transposing two probability columns —
a mistake no range check can catch, since all three are valid probabilities.
"""

HOME, DRAW, AWAY = "H", "D", "A"

# Order is load-bearing: every probability vector, every score and every
# column triple in this package is (home, draw, away) in this order.
OUTCOMES = (HOME, DRAW, AWAY)
N_OUTCOMES = len(OUTCOMES)

OUTCOME_LABELS = {HOME: "Home win", DRAW: "Draw", AWAY: "Away win"}

# Probability columns, in OUTCOMES order.
PROBABILITY_COLUMNS = ("p_home", "p_draw", "p_away")
ODDS_COLUMNS = ("odds_home", "odds_draw", "odds_away")

DEFAULT_DATA_DIR = "exported_data/football"

# The tidy shape every loader produces, whatever the source file looked like.
MATCH_COLUMNS = ["ds", "home_team", "away_team", "home_goals", "away_goals", "outcome"]


def outcome_from_goals(home_goals, away_goals):
    """The three-way result of a finished match."""
    if home_goals > away_goals:
        return HOME
    if home_goals < away_goals:
        return AWAY
    return DRAW


def outcome_index(outcome):
    """Position of an outcome in OUTCOMES — the index into any probability vector."""
    try:
        return OUTCOMES.index(outcome)
    except ValueError:
        raise ValueError(f"Unknown outcome {outcome!r}. Expected one of {OUTCOMES}.") from None


def outcome_label(outcome):
    return OUTCOME_LABELS[outcome]
