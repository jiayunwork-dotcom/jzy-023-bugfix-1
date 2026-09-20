"""End-to-end tests for the HMM decoding service (via the HTTP layer)."""

import copy
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.demo import DEMO_MODEL, DEMO_OBSERVATIONS
from app.service import create_app
from app.viterbi import TIE_BREAK_RULE


@pytest.fixture()
def app(tmp_path):
    application = create_app(db_path=str(tmp_path / "models.sqlite3"))
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


def register(client, spec):
    return client.post("/models", json=spec)


def decode(client, model, observations):
    return client.post("/decode", json={"model": model, "observations": observations})


def two_state_model(name="two-state"):
    return {
        "name": name,
        "states": ["on", "off"],
        "alphabet": ["x", "y"],
        "initial": [0.6, 0.4],
        "transition": [[0.9, 0.1], [0.1, 0.9]],
        "emission": [[0.9, 0.1], [0.1, 0.9]],
    }


# ---------------------------------------------------------------- demo model

def test_demo_model_is_builtin_and_listed(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    assert DEMO_MODEL["name"] in resp.get_json()["models"]
    assert len(client.get(f"/models/{DEMO_MODEL['name']}").get_json()["states"]) == 3


def test_demo_viterbi_follows_embedded_regime_switch(client):
    resp = decode(client, DEMO_MODEL["name"], DEMO_OBSERVATIONS)
    assert resp.status_code == 200
    path = resp.get_json()["path"]
    # The embedded switch is a-block then b-block; Viterbi tracks it.
    assert set(path[:8]) == {"regime_a"}
    assert set(path[-8:]) == {"regime_b"}
    # The switch is routed through the transit state for exactly one step.
    assert path.count("transit") == 1
    assert 8 <= path.index("transit") <= len(path) - 8


def test_posterior_rows_sum_to_one(client):
    resp = decode(client, DEMO_MODEL["name"], DEMO_OBSERVATIONS)
    posteriors = resp.get_json()["posteriors"]
    assert len(posteriors) == len(DEMO_OBSERVATIONS)
    for row in posteriors:
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-9)


def test_posterior_argmax_is_not_the_viterbi_path(client):
    """The fork case: pointwise posterior maxima must not be passed off as
    the Viterbi path -- on the demo sequence they genuinely differ."""
    data = decode(client, DEMO_MODEL["name"], DEMO_OBSERVATIONS).get_json()
    posterior_argmax = [
        max(row, key=row.get) for row in data["posteriors"]
    ]
    assert posterior_argmax != data["path"]
    # Specifically: Viterbi commits to one transit step, the posterior
    # argmax never selects the transit state.
    assert "transit" in data["path"]
    assert "transit" not in posterior_argmax


def test_decode_response_shape_and_tie_break_rule(client):
    data = decode(client, DEMO_MODEL["name"], DEMO_OBSERVATIONS).get_json()
    assert data["tie_break"] == TIE_BREAK_RULE == "lowest_state_index"
    assert isinstance(data["log_probability"], float)
    assert data["length"] == len(DEMO_OBSERVATIONS)
    assert len(data["path"]) == data["length"]


def test_log_probability_is_joint_of_returned_path(client):
    obs = list("aabbaabb")
    data = decode(client, DEMO_MODEL["name"], obs).get_json()
    spec = client.get(f"/models/{DEMO_MODEL['name']}").get_json()
    states = spec["states"]
    path = data["path"]
    alpha = spec["alphabet"]
    logp = math.log(spec["initial"][states.index(path[0])])
    for t, symbol in enumerate(obs):
        logp += math.log(spec["emission"][states.index(path[t])][alpha.index(symbol)])
        if t > 0:
            logp += math.log(
                spec["transition"][states.index(path[t - 1])][states.index(path[t])]
            )
    assert data["log_probability"] == pytest.approx(logp, rel=1e-9, abs=1e-9)


# ------------------------------------------------------- short observations

@pytest.mark.parametrize("obs", [["a"], ["a", "b"], "a", "ab"])
def test_extremely_short_sequences_decode_normally(client, obs):
    resp = decode(client, DEMO_MODEL["name"], obs)
    assert resp.status_code == 200
    data = resp.get_json()
    n = len(obs)
    assert len(data["path"]) == n
    assert len(data["posteriors"]) == n
    for row in data["posteriors"]:
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------- registration

def test_long_sequence_does_not_underflow(client):
    """5000 observations: linear-domain products would underflow to exactly
    0.0 long before the end; the log-domain decoder must still return a
    proper path and normalized posteriors."""
    register(client, two_state_model())
    obs = (["x"] * 50 + ["y"] * 50) * 50  # 5000 symbols, 50 embedded switches
    resp = decode(client, "two-state", obs)
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["path"]) == 5000
    assert math.isfinite(data["log_probability"])
    # path follows the block structure
    assert set(data["path"][:40]) == {"on"}
    assert set(data["path"][50:90]) == {"off"}
    # posteriors are genuine probability rows, not collapsed zeros
    for row in (data["posteriors"][0], data["posteriors"][2500], data["posteriors"][-1]):
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(p > 0.0 for p in row.values())


# ------------------------------------------------------------- registration

def test_transition_row_sum_off_is_rejected(client):
    spec = two_state_model()
    spec["transition"] = [[0.9, 0.2], [0.1, 0.9]]  # row 0 sums to 1.1
    resp = register(client, spec)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["type"] == "invalid_model"


def test_emission_row_sum_off_is_rejected(client):
    spec = two_state_model()
    spec["emission"] = [[0.9, 0.1], [0.4, 0.4]]
    resp = register(client, spec)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["type"] == "invalid_model"


def test_initial_sum_off_is_rejected(client):
    spec = two_state_model()
    spec["initial"] = [0.7, 0.4]
    resp = register(client, spec)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["type"] == "invalid_model"


def test_dimension_mismatch_is_rejected_before_decoding(client):
    spec = two_state_model()
    spec["initial"] = [0.5, 0.3, 0.2]  # 3 states but 2x2 transition
    resp = register(client, spec)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["type"] == "invalid_model"

    spec = two_state_model()
    spec["emission"] = [[0.9, 0.1]]  # only one row for two states
    resp = register(client, spec)
    assert resp.status_code == 400


def test_single_state_model_is_rejected(client):
    spec = {
        "name": "one-state",
        "alphabet": ["x"],
        "initial": [1.0],
        "transition": [[1.0]],
        "emission": [[1.0]],
    }
    resp = register(client, spec)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["type"] == "invalid_model"


def test_missing_field_is_rejected(client):
    spec = two_state_model()
    del spec["emission"]
    resp = register(client, spec)
    assert resp.status_code == 400
    assert resp.get_json()["error"]["type"] == "missing_field"


# ------------------------------------------------------------------ decoding

def test_empty_observations_rejected(client):
    register(client, two_state_model())
    for obs in ([], ""):
        resp = decode(client, "two-state", obs)
        assert resp.status_code == 400
        assert resp.get_json()["error"]["type"] == "empty_observations"


def test_unknown_symbol_rejected(client):
    register(client, two_state_model())
    resp = decode(client, "two-state", ["x", "Q"])
    assert resp.status_code == 400
    assert resp.get_json()["error"]["type"] == "unknown_symbol"


def test_unknown_model_rejected(client):
    resp = decode(client, "no-such-model", ["x"])
    assert resp.status_code == 404
    assert resp.get_json()["error"]["type"] == "unknown_model"


def test_missing_decode_fields_rejected(client):
    register(client, two_state_model())
    resp = client.post("/decode", json={"observations": ["x"]})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["type"] == "missing_field"
    resp = client.post("/decode", json={"model": "two-state"})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["type"] == "missing_field"


def test_zero_probability_sequence_refused(client):
    spec = {
        "name": "rigid",
        "alphabet": ["x", "y"],
        "initial": [0.5, 0.5],
        "transition": [[0.9, 0.1], [0.1, 0.9]],
        "emission": [[1.0, 0.0], [1.0, 0.0]],  # 'y' is impossible everywhere
    }
    register(client, spec)
    resp = decode(client, "rigid", ["x", "y"])
    assert resp.status_code == 422
    assert resp.get_json()["error"]["type"] == "zero_probability"


# ------------------------------------------------------------- interventions

def test_blurred_emission_path_follows_transitions_not_observations(client):
    """Blurring one state's emission until it is nearly indistinguishable
    must make the path follow the transition dynamics instead of hard-
    tracking the observations."""
    obs = list("aaaaaaaabbbb")

    sharp = copy.deepcopy(DEMO_MODEL)
    sharp["name"] = "sharp"
    register(client, sharp)
    sharp_path = decode(client, "sharp", obs).get_json()["path"]

    blurred = copy.deepcopy(DEMO_MODEL)
    blurred["name"] = "blurred"
    # regime_b emission becomes almost identical to regime_a's
    blurred["emission"][1] = [0.79, 0.11, 0.10]
    register(client, blurred)
    blurred_path = decode(client, "blurred", obs).get_json()["path"]

    # Sharp emissions: the path tracks the observation blocks and switches.
    assert "regime_a" in sharp_path[:8] and "regime_b" in sharp_path[-4:]
    # Blurred: the path no longer chases the observations; the sticky
    # transition dynamics keep it in a single state for the whole segment.
    assert len(set(blurred_path)) == 1


def test_single_observation_change_flips_that_time_step(client):
    """With weakly-sticky transitions and sharp emissions, changing one
    observation to another state's typical symbol moves the path there."""
    spec = {
        "name": "flippy",
        "states": ["sx", "sy", "sz"],
        "alphabet": ["x", "y", "z"],
        "initial": [1 / 3, 1 / 3, 1 / 3],
        "transition": [
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ],
        "emission": [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ],
    }
    register(client, spec)
    base = decode(client, "flippy", list("xxxxxx")).get_json()["path"]
    assert base == ["sx"] * 6

    obs = list("xxxxxx")
    obs[3] = "y"
    flipped = decode(client, "flippy", obs).get_json()["path"]
    assert flipped[3] == "sy"
    assert flipped[:3] == ["sx"] * 3 and flipped[4:] == ["sx"] * 2


def test_suppressed_entry_probability_prevents_staying(client):
    """Crushing the probability of entering a state to ~0 must keep the
    path from settling there, even when its emissions fit best."""
    base = {
        "name": "entry-control",
        "states": ["s0", "s1", "s2"],
        "alphabet": ["u", "v", "z"],
        "initial": [0.5, 0.5, 0.0],
        "transition": [
            [0.85, 0.10, 0.05],
            [0.10, 0.85, 0.05],
            [0.025, 0.025, 0.95],
        ],
        "emission": [
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ],
    }
    obs = list("uuzzzzzzuu")
    register(client, base)
    control_path = decode(client, "entry-control", obs).get_json()["path"]
    # Control: with normal entry probability the path does settle in s2.
    assert "s2" in control_path

    suppressed = copy.deepcopy(base)
    suppressed["name"] = "entry-suppressed"
    suppressed["transition"] = [
        [0.9, 0.1, 0.000000001],
        [0.1, 0.9, 0.000000001],
        [0.5, 0.5, 0.000000001],
    ]
    register(client, suppressed)
    suppressed_path = decode(client, "entry-suppressed", obs).get_json()["path"]
    assert "s2" not in suppressed_path


# ------------------------------------------------------------------ isolation

def test_parallel_decode_of_two_models_does_not_cross_write(app):
    client = app.test_client()
    register(client, two_state_model("model-a"))
    demo_spec = client.get(f"/models/{DEMO_MODEL['name']}").get_json()
    assert demo_spec["name"] == DEMO_MODEL["name"]

    obs_a = list("xxyxxxyy")
    obs_b = DEMO_OBSERVATIONS

    baseline_a = decode(client, "model-a", obs_a).get_json()
    baseline_b = decode(client, DEMO_MODEL["name"], obs_b).get_json()
    assert baseline_a["path"] != baseline_b["path"]

    def decode_many(name, obs, rounds):
        c = app.test_client()  # independent client per thread
        return [decode(c, name, obs).get_json() for _ in range(rounds)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        fut_a = pool.submit(decode_many, "model-a", obs_a, 16)
        fut_b = pool.submit(decode_many, DEMO_MODEL["name"], obs_b, 16)
        results_a, results_b = fut_a.result(), fut_b.result()

    # Every concurrent result is identical to the solo baseline: each model
    # keeps its own path and posterior table, no cross-talk.
    for r in results_a:
        assert r == baseline_a
        assert r["model"] == "model-a"
    for r in results_b:
        assert r == baseline_b
        assert r["model"] == DEMO_MODEL["name"]
