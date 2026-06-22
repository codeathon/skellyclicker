"""Tests for DLC labeled-data folder naming from video paths."""

from skellyclicker.core.deeplabcut_handler.create_deeplabcut.create_deeplabcut_project_data import (
	get_session_name,
)


def test_get_session_name_prefers_session_folder():
	path = "/data/session_2025-06-18_ferret/clips/cam1.mp4"
	assert get_session_name(path) == "session_2025-06-18_ferret"


def test_get_session_name_falls_back_to_video_folder():
	path = "/Users/me/experiments/recording"
	assert get_session_name(path) == "recording"
