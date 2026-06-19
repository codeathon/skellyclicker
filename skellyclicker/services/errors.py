"""Service-layer errors raised by SessionStore."""


class SessionError(Exception):
	"""Recoverable session/state error mapped to HTTP 400 by the API."""

	def __init__(self, message: str) -> None:
		self.message = message
		super().__init__(message)


class SessionConflictError(SessionError):
	"""Mutation blocked while a job is active (HTTP 409)."""
