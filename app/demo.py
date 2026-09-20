"""Built-in demo model: a three-state regime-switching HMM.

Two "regime" states each favour their own observation symbol and are very
sticky; switching directly between them is rare (0.002). The third "transit"
state is the plausible gateway between regimes, but it is itself unstable
(self-transition 0.1), so any single best path passes through it in one
step at most.

On the embedded observation sequence below (a long run of 'a', then a long
run of 'b' -- a human-embedded regime switch), the Viterbi path follows the
switch and routes through the transit state for exactly one step at the
boundary. The per-time-step posterior marginals instead spread the
switch-point uncertainty over the two boundary time slices (transit
posterior ~0.42 on each), so the pointwise posterior argmax never selects
the transit state. Hence, on this very sequence, the posterior-argmax
string and the Viterbi path are genuinely different -- which is exactly
why a decoder must not pass off per-time posterior maxima as the Viterbi
path.
"""

DEMO_MODEL = {
    "name": "demo-regime-switch",
    "states": ["regime_a", "regime_b", "transit"],
    "alphabet": ["a", "b", "c"],
    "initial": [0.495, 0.495, 0.01],
    "transition": [
        [0.950, 0.002, 0.048],
        [0.002, 0.950, 0.048],
        [0.450, 0.450, 0.100],
    ],
    "emission": [
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.45, 0.45, 0.10],
    ],
}

#: Observation sequence with a human-embedded regime switch in the middle.
DEMO_OBSERVATIONS = list("aaaaaaaaaaaabbbbbbbbbbbb")
