"""Viterbi decoding in the log domain.

Maximizes the joint probability of the whole state path given the whole
observation sequence; the path is recovered by backtracing. Ties are broken
by a fixed, documented rule so the result is deterministic.
"""

from .logdomain import NEG_INF
from .errors import ZeroProbabilityError

#: Tie-breaking rule, fixed for the whole service and reported in every
#: decode response: whenever two predecessors (or two final states) have
#: exactly equal log scores, the one with the lowest state index wins.
#: Implemented by scanning states in index order and only replacing the
#: incumbent on a strictly greater score.
TIE_BREAK_RULE = "lowest_state_index"


def viterbi(log_initial, log_transition, log_emission, obs_indices):
    """Return (path, log_joint_probability).

    path: list of state indices, one per observation.
    log_joint_probability: log P(path, observations) for the best path.
    Raises ZeroProbabilityError if no path has positive probability.
    """
    n_obs = len(obs_indices)
    n_states = len(log_initial)

    delta = [[NEG_INF] * n_states for _ in range(n_obs)]
    backptr = [[0] * n_states for _ in range(n_obs)]

    first = obs_indices[0]
    for i in range(n_states):
        delta[0][i] = log_initial[i] + log_emission[i][first]

    for t in range(1, n_obs):
        symbol = obs_indices[t]
        for j in range(n_states):
            best_score = NEG_INF
            best_prev = 0
            for i in range(n_states):
                candidate = delta[t - 1][i] + log_transition[i][j]
                if candidate > best_score:  # strict: earliest index wins ties
                    best_score = candidate
                    best_prev = i
            delta[t][j] = best_score + log_emission[j][symbol]
            backptr[t][j] = best_prev

    best_score = NEG_INF
    last_state = 0
    for i in range(n_states):
        if delta[n_obs - 1][i] > best_score:  # strict: earliest index wins
            best_score = delta[n_obs - 1][i]
            last_state = i

    if best_score == NEG_INF:
        raise ZeroProbabilityError(
            "no state path has positive probability for this observation sequence"
        )

    path = [0] * n_obs
    path[n_obs - 1] = last_state
    for t in range(n_obs - 1, 0, -1):
        path[t - 1] = backptr[t][path[t]]
    return path, best_score
