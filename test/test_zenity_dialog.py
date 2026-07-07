"""Tests for zenity file dialog backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from skellyclicker.services.dialog_errors import DialogUnavailable
from skellyclicker.services.zenity_dialog import zenity_spawn


def test_zenity_spawn_single_file():
	mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="/data/video.mp4\n"))
	with patch("skellyclicker.services.zenity_dialog.subprocess.run", mock_run):
		paths = zenity_spawn("file", "Pick video", ["mp4"])
	assert paths == ["/data/video.mp4"]


def test_zenity_spawn_cancelled():
	mock_run = MagicMock(return_value=MagicMock(returncode=1, stdout=""))
	with patch("skellyclicker.services.zenity_dialog.subprocess.run", mock_run):
		paths = zenity_spawn("files", "Pick videos", ["mp4"])
	assert paths == []


def test_zenity_spawn_save_passes_default_filename():
	mock_run = MagicMock(
		return_value=MagicMock(
			returncode=0,
			stdout="/data/videos/skellyclicker_data/labels.csv\n",
		)
	)
	with patch("skellyclicker.services.zenity_dialog.subprocess.run", mock_run):
		paths = zenity_spawn(
			"save",
			"Save human labels CSV",
			["csv"],
			"/data/videos/skellyclicker_data/2026_labels.csv",
		)
	assert paths == ["/data/videos/skellyclicker_data/labels.csv"]
	args = mock_run.call_args[0][0]
	assert "--filename" in args
	idx = args.index("--filename")
	assert args[idx + 1] == "/data/videos/skellyclicker_data/2026_labels.csv"


def test_zenity_spawn_error():
	mock_run = MagicMock(
		return_value=MagicMock(returncode=255, stdout="", stderr="Gdk: no display"),
	)
	with patch("skellyclicker.services.zenity_dialog.subprocess.run", mock_run):
		try:
			zenity_spawn("directory", "Pick folder", None)
			raised = False
		except DialogUnavailable:
			raised = True
	assert raised
