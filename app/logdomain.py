"""Log-domain primitives.

All decoding math is done in the log domain so that long observation
sequences never underflow to zero in the linear domain.
"""

import math
from typing import Iterable

NEG_INF = float("-inf")


def to_log(p: float) -> float:
    """Map a probability to its log; zero maps to -inf."""
    if p < 0.0:
        raise ValueError(f"negative probability: {p!r}")
    if p == 0.0:
        return NEG_INF
    return math.log(p)


def logsumexp(values: Iterable[float]) -> float:
    """Stable log(sum(exp(values))); log(0) if every term is -inf."""
    vals = list(values)
    if not vals:
        return NEG_INF
    m = max(vals)
    if m == NEG_INF:
        return NEG_INF
    total = sum(math.exp(v - m) for v in vals if v != NEG_INF)
    return m + math.log(total)
