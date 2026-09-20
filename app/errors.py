"""Typed errors shared across the service."""


class ApiError(Exception):
    """An error that maps directly onto an HTTP JSON error response."""

    def __init__(self, type_: str, message: str, status: int = 400):
        super().__init__(message)
        self.type = type_
        self.message = message
        self.status = status

    def to_dict(self) -> dict:
        return {"error": {"type": self.type, "message": self.message}}


class ZeroProbabilityError(Exception):
    """Raised when the observation sequence has probability zero under the model.

    In the log domain this surfaces as a best score of -inf; we refuse to
    emit a made-up path in that case.
    """
