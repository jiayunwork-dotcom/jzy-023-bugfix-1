"""Flask HTTP layer: model registration, listing, and decoding."""

import os

from flask import Flask, jsonify, request

from .demo import DEMO_MODEL, DEMO_OBSERVATIONS
from .errors import ApiError, ZeroProbabilityError
from .logdomain import to_log
from .posterior import forward_backward
from .store import ModelStore
from .validation import parse_observations, validate_model_spec
from .viterbi import TIE_BREAK_RULE, viterbi

DEFAULT_DB_PATH = "hmm_models.sqlite3"


def _log_matrices(spec):
    log_initial = [to_log(p) for p in spec["initial"]]
    log_transition = [[to_log(p) for p in row] for row in spec["transition"]]
    log_emission = [[to_log(p) for p in row] for row in spec["emission"]]
    return log_initial, log_transition, log_emission


def create_app(db_path: str = None, with_demo: bool = True) -> Flask:
    app = Flask(__name__)
    store = ModelStore(db_path or os.environ.get("HMM_DB_PATH", DEFAULT_DB_PATH))
    app.extensions["model_store"] = store

    if with_demo and store.get(DEMO_MODEL["name"]) is None:
        store.save(validate_model_spec(DEMO_MODEL))

    @app.errorhandler(ApiError)
    def _api_error(err: ApiError):
        return jsonify(err.to_dict()), err.status

    @app.errorhandler(404)
    def _not_found(_err):
        return jsonify({"error": {"type": "not_found", "message": "unknown endpoint"}}), 404

    @app.errorhandler(405)
    def _method_not_allowed(_err):
        return (
            jsonify({"error": {"type": "method_not_allowed", "message": "method not allowed"}}),
            405,
        )

    @app.post("/models")
    def register_model():
        body = request.get_json(silent=True)
        if body is None:
            raise ApiError("invalid_body", "request body must be a JSON object")
        spec = validate_model_spec(body)
        store.save(spec)
        return jsonify({"registered": spec["name"], "model": spec}), 201

    @app.get("/models")
    def list_models():
        return jsonify({"models": store.names()})

    @app.get("/models/<name>")
    def get_model(name: str):
        spec = store.get(name)
        if spec is None:
            raise ApiError("unknown_model", f"no model registered under name {name!r}", 404)
        return jsonify(spec)

    @app.get("/demo")
    def demo():
        return jsonify({"model": DEMO_MODEL["name"], "observations": DEMO_OBSERVATIONS})

    @app.post("/decode")
    def decode():
        body = request.get_json(silent=True)
        if body is None:
            raise ApiError("invalid_body", "request body must be a JSON object")
        if "model" not in body:
            raise ApiError("missing_field", "missing required field: 'model'")
        if "observations" not in body:
            raise ApiError("missing_field", "missing required field: 'observations'")

        name = body["model"]
        if not isinstance(name, str):
            raise ApiError("invalid_request", "'model' must be a string")
        spec = store.get(name)
        if spec is None:
            raise ApiError("unknown_model", f"no model registered under name {name!r}", 404)

        obs_indices = parse_observations(spec["alphabet"], body["observations"])
        log_initial, log_transition, log_emission = _log_matrices(spec)

        try:
            path, log_joint = viterbi(log_initial, log_transition, log_emission, obs_indices)
            posteriors, log_likelihood = forward_backward(
                log_initial, log_transition, log_emission, obs_indices
            )
        except ZeroProbabilityError as exc:
            raise ApiError("zero_probability", str(exc), 422) from exc

        states = spec["states"]
        return jsonify(
            {
                "model": name,
                "length": len(obs_indices),
                "path": [states[i] for i in path],
                "log_probability": log_joint,
                "log_likelihood": log_likelihood,
                "posteriors": [
                    {states[i]: row[i] for i in range(len(states))} for row in posteriors
                ],
                "tie_break": TIE_BREAK_RULE,
            }
        )

    return app
