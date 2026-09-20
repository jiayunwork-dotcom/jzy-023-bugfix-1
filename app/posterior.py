"""Forward-backward algorithm and per-time-step posterior marginals.

Everything is accumulated in the log domain (logsumexp), so arbitrarily
long sequences cannot underflow to all-zero intermediate tables.
"""

import math

from .logdomain import NEG_INF, logsumexp
from .errors import ZeroProbabilityError


def forward_backward(log_initial, log_transition, log_emission, obs_indices):
    """Return (posteriors, log_likelihood).

    posteriors[t][i] = P(state_t = i | observations); each row sums to 1.
    log_likelihood = log P(observations) under the model.
    Raises ZeroProbabilityError if the sequence has probability zero.
    """
    n_obs = len(obs_indices)
    n_states = len(log_initial)

    # Forward: alpha[t][i] = P(o_0..o_t, state_t = i), in log space.
    # At t = 0 this is initial * emission, as specified.
    log_alpha = [[NEG_INF] * n_states for _ in range(n_obs)]
    first = obs_indices[0]
    for i in range(n_states):
        log_alpha[0][i] = log_initial[i] + log_emission[i][first]
    for t in range(1, n_obs):
        symbol = obs_indices[t]
        for j in range(n_states):
            log_alpha[t][j] = logsumexp(
                log_alpha[t - 1][i] + log_transition[i][j] for i in range(n_states)
            ) + log_emission[j][symbol]

    # Backward: beta[t][i] = P(o_{t+1}..o_{T-1} | state_t = i), in log space.
    # The last row is the log-domain unit value, log(1) = 0.
    # beta[t][i] = sum_j T[i->j] * emission[j][o_{t+1}] * beta[t+1][j];
    # note the transition is indexed [i][j] (from current state i to
    # successor j) -- transposing it to [j][i] silently corrupts every row
    # except the last whenever the transition matrix is asymmetric.
    log_beta = [[0.0] * n_states for _ in range(n_obs)]
    for t in range(n_obs - 2, -1, -1):
        nxt = obs_indices[t + 1]
        for i in range(n_states):
            log_beta[t][i] = logsumexp(
                log_transition[i][j] + log_emission[j][nxt] + log_beta[t + 1][j]
                for j in range(n_states)
            )

    # Posterior at time t: alpha * beta, normalized (all in log space).
    posteriors = []
    for t in range(n_obs):
        log_joint = [log_alpha[t][i] + log_beta[t][i] for i in range(n_states)]
        norm = logsumexp(log_joint)
        if norm == NEG_INF:
            raise ZeroProbabilityError(
                "observation sequence has zero probability under this model"
            )
        posteriors.append([math.exp(v - norm) for v in log_joint])

    log_likelihood = logsumexp(log_alpha[n_obs - 1])
    return posteriors, log_likelihood
