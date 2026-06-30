"""Shared exceptions for native file dialog backends."""


class DialogCancelled(Exception):
	"""User closed the file dialog without selecting."""


class DialogUnavailable(Exception):
	"""No display / dialog tool available."""
