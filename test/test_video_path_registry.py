"""Tests for basename → absolute path registry (cross-folder train/analyze)."""

from pathlib import Path

import pytest

from skellyclicker.services.video_path_registry import (
	build_video_path_registry,
	labeled_data_prefix,
	resolve_video_path,
)


def test_build_registry_maps_basename(tmp_path: Path):
	a = tmp_path / "exp" / "cam.mp4"
	a.parent.mkdir()
	a.write_bytes(b"x")
	reg = build_video_path_registry([str(a)])
	assert reg["cam.mp4"] == str(a.resolve())


def test_duplicate_basename_raises(tmp_path: Path):
	a = tmp_path / "a" / "cam.mp4"
	b = tmp_path / "b" / "cam.mp4"
	a.parent.mkdir()
	b.parent.mkdir()
	a.write_bytes(b"x")
	b.write_bytes(b"x")
	with pytest.raises(ValueError, match="Duplicate video basename"):
		build_video_path_registry([str(a), str(b)])


def test_resolve_video_path(tmp_path: Path):
	a = tmp_path / "folder" / "expA.mp4"
	a.parent.mkdir()
	a.write_bytes(b"x")
	assert resolve_video_path("expA.mp4", [str(a)]) == str(a.resolve())


def test_labeled_data_prefix_uses_session_segment(tmp_path: Path):
	path = tmp_path / "session_001" / "videos" / "cam.mp4"
	assert labeled_data_prefix(str(path)) == "session_001"


def test_labeled_data_prefix_falls_back_to_folder(tmp_path: Path):
	path = tmp_path / "my_exp" / "cam.mp4"
	assert labeled_data_prefix(str(path)) == "my_exp"
