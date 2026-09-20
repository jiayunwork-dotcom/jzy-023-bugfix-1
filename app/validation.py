"""Validation of model specs and observation sequences.

Registration-time checks (dimensions, normalization, alphabet) live here so
that a malformed model can never reach the decoder. The normalization
tolerance is pinned to a single constant.
"""

import math

from .errors import ApiError

#: Pinned tolerance for all "sums to 1" checks.
TOLERANCE = 1e-6

REQUIRED_FIELDS = ("name", "alphabet", "initial", "transition", "emission")


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_probability_vector(values, what: str, expect_sum_one: bool) -> list:
    if not isinstance(values, list) or not values:
        raise ApiError("invalid_model", f"{what} must be a non-empty list of numbers")
    out = []
    for v in values:
        if not _is_number(v):
            raise ApiError("invalid_model", f"{what} contains a non-numeric value: {v!r}")
        if v < 0.0 or v > 1.0:
            raise ApiError("invalid_model", f"{what} contains a value outside [0, 1]: {v!r}")
        out.append(float(v))
    if expect_sum_one and abs(sum(out) - 1.0) > TOLERANCE:
        raise ApiError(
            "invalid_model",
            f"{what} must sum to 1 (got {sum(out)!r}, tolerance {TOLERANCE})",
        )
    return out


def _check_probability_matrix(rows, what: str, n_rows: int, n_cols: int) -> list:
    if not isinstance(rows, list) or len(rows) != n_rows:
        raise ApiError(
            "invalid_model",
            f"{what} must have exactly {n_rows} rows (got "
            f"{len(rows) if isinstance(rows, list) else type(rows).__name__})",
        )
    matrix = []
    for r, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != n_cols:
            raise ApiError(
                "invalid_model",
                f"{what} row {r} must have exactly {n_cols} entries",
            )
        matrix.append(_check_probability_vector(row, f"{what} row {r}", expect_sum_one=True))
    return matrix


def validate_model_spec(body) -> dict:
    """Validate a raw JSON body; return a normalized model spec.

    Raises ApiError (type "missing_field" or "invalid_model") on any problem.
    """
    if not isinstance(body, dict):
        raise ApiError("invalid_model", "model spec must be a JSON object")
    for field in REQUIRED_FIELDS:
        if field not in body:
            raise ApiError("missing_field", f"missing required field: {field!r}")

    name = body["name"]
    if not isinstance(name, str) or not name.strip():
        raise ApiError("invalid_model", "name must be a non-empty string")
    name = name.strip()
    if len(name) > 128:
        raise ApiError("invalid_model", "name is too long (max 128 characters)")

    alphabet = body["alphabet"]
    if not isinstance(alphabet, list) or not alphabet:
        raise ApiError("invalid_model", "alphabet must be a non-empty list of symbols")
    for symbol in alphabet:
        if not isinstance(symbol, str) or symbol == "":
            raise ApiError("invalid_model", f"alphabet symbols must be non-empty strings, got {symbol!r}")
    if len(set(alphabet)) != len(alphabet):
        raise ApiError("invalid_model", "alphabet symbols must be unique")

    initial = _check_probability_vector(body["initial"], "initial", expect_sum_one=True)
    n_states = len(initial)
    if n_states < 2:
        raise ApiError("invalid_model", "model must have at least 2 states")

    states = body.get("states")
    if states is None:
        states = [f"S{i}" for i in range(n_states)]
    else:
        if not isinstance(states, list) or len(states) != n_states:
            raise ApiError("invalid_model", f"states must be a list of {n_states} names")
        for s in states:
            if not isinstance(s, str) or s == "":
                raise ApiError("invalid_model", f"state names must be non-empty strings, got {s!r}")
        if len(set(states)) != n_states:
            raise ApiError("invalid_model", "state names must be unique")

    transition = _check_probability_matrix(body["transition"], "transition", n_states, n_states)
    emission = _check_probability_matrix(body["emission"], "emission", n_states, len(alphabet))

    return {
        "name": name,
        "states": states,
        "alphabet": list(alphabet),
        "initial": initial,
        "transition": transition,
        "emission": emission,
    }


def parse_observations(alphabet: list, observations) -> list:
    """Map an observation sequence (list of symbols, or a string taken one
    character at a time) to alphabet indices. Rejects empty sequences and
    symbols outside the alphabet with typed errors."""
    if isinstance(observations, str):
        symbols = list(observations)
    elif isinstance(observations, list):
        symbols = observations
        for s in symbols:
            if not isinstance(s, str):
                raise ApiError(
                    "invalid_observations",
                    f"observation symbols must be strings, got {s!r}",
                )
    else:
        raise ApiError(
            "invalid_observations",
            "observations must be a list of symbols or a string",
        )

    if len(symbols) == 0:
        raise ApiError("empty_observations", "observation sequence must contain at least 1 symbol")

    index = {symbol: i for i, symbol in enumerate(alphabet)}
    indices = []
    for s in symbols:
        if s not in index:
            raise ApiError(
                "unknown_symbol",
                f"observation symbol {s!r} is not in the model alphabet {alphabet!r}",
            )
        indices.append(index[s])
    return indices
