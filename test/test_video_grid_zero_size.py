"""Regression: OpenCV 0x0 metadata must not 500 the labeler open path."""

from skellyclicker.core.video_handler.video_handler import VideoHandler
from skellyclicker.core.video_handler.video_models import (
	GridParameters,
	VideoMetadata,
	VideoPlaybackState,
)


def test_calculate_scaling_handles_zero_dimensions():
	params = VideoHandler._calculate_scaling_parameters(0, 0, (100, 80))
	assert params.scaled_width >= 1
	assert params.scaled_height >= 1
	assert params.original_width == 1
	assert params.original_height == 1


def test_grid_layout_handles_zero_metadata():
	"""ZeroDivisionError in layout used to become Internal Server Error."""
	# Bypass VideoCapture validation — layout only reads metadata sizes.
	videos = {
		"/tmp/a.mp4": VideoPlaybackState.model_construct(
			metadata=VideoMetadata(
				path="/tmp/a.mp4",
				name="a.mp4",
				width=0,
				height=0,
				frame_count=10,
			),
			cap=None,
			scaling_params=None,
		)
	}
	grid = GridParameters.calculate_native(videos)
	assert grid.cell_width >= 1
	assert grid.cell_height >= 1
	assert grid.rows >= 1
	assert grid.columns >= 1
