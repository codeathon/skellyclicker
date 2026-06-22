"""Tests for native file dialog availability checks."""

from __future__ import annotations

import sys
from unittest.mock import patch

from skellyclicker.services.native_dialog import (
	check_dialog_availability,
	dialog_startup_warning,
)


def test_dialog_startup_warning_none_when_available():
	with patch(
		"skellyclicker.services.native_dialog.check_dialog_availability",
		return_value=(True, "ok"),
	):
		assert dialog_startup_warning() is None


def test_dialog_startup_warning_when_unavailable():
	with patch(
		"skellyclicker.services.native_dialog.check_dialog_availability",
		return_value=(False, "no tkinter"),
	):
		msg = dialog_startup_warning()
		assert msg is not None
		assert "fall back" in msg
		assert "no tkinter" in msg


def test_linux_without_display_is_unavailable():
	with patch.object(sys, "platform", "linux"):
		with patch.dict("os.environ", {}, clear=True):
			available, detail = check_dialog_availability()
	assert available is False
	assert "DISPLAY" in detail


def test_import_error_reports_ubuntu_hint():
	with patch.object(sys, "platform", "darwin"):
		with patch.dict("os.environ", {"DISPLAY": ":0"}, clear=True):
			with patch.dict("sys.modules", {"tkinter": None}):
				available, detail = check_dialog_availability()
	assert available is False
	assert "python3-tk" in detail
