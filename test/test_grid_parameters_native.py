"""Native grid sizing for web labeler display."""

from __future__ import annotations

from unittest.mock import MagicMock

from skellyclicker.core.video_handler.video_models import (
	GridParameters,
	VideoMetadata,
	VideoPlaybackState,
)


def _video(width: int, height: int, name: str = "cam.mp4") -> VideoPlaybackState:
	return VideoPlaybackState(
		metadata=VideoMetadata(
			path=f"/tmp/{name}",
			name=name,
			width=width,
			height=height,
			frame_count=100,
		),
		cap=MagicMock(),
		scaling_params=None,
	)


def test_calculate_native_single_1080p():
	videos = {"/tmp/a.mp4": _video(1920, 1080, "a.mp4")}
	grid = GridParameters.calculate_native(videos)
	assert grid.cell_width == 1920
	assert grid.cell_height == 1080
	assert grid.total_width == 1920
	assert grid.total_height == 1080


def test_calculate_native_two_1080p_cameras():
	videos = {
		"/tmp/a.mp4": _video(1920, 1080, "a.mp4"),
		"/tmp/b.mp4": _video(1920, 1080, "b.mp4"),
	}
	grid = GridParameters.calculate_native(videos)
	assert grid.cell_width == 1920
	assert grid.cell_height == 1080
	assert grid.total_width == 3840
	assert grid.total_height == 1080


def test_calculate_capped_differs_from_native():
	videos = {"/tmp/a.mp4": _video(3840, 2160, "a.mp4")}
	native = GridParameters.calculate_native(videos)
	capped = GridParameters.calculate(videos, (1920, 1080))
	assert native.total_width == 3840
	assert capped.total_width == 1920
