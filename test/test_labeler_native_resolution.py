"""Labeler opens with native grid and preview grid for scrubbing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from skellyclicker.core.video_handler.video_handler import VideoHandler


def test_from_videos_for_labeler_sets_native_and_preview_grids():
	video_path = "/tmp/cam0.mp4"
	mock_cap = MagicMock()
	mock_cap.isOpened.return_value = True
	mock_cap.get.side_effect = lambda prop: {3: 1920, 4: 1080, 7: 50}.get(prop, 0)

	with patch("cv2.VideoCapture", return_value=mock_cap):
		handler = VideoHandler.from_videos_for_labeler(
			video_paths=[video_path],
			tracked_point_names=["nose"],
		)

	assert handler.grid_parameters.total_width == 1920
	assert handler.grid_parameters.total_height == 1080
	assert handler.preview_grid_parameters is not None
	assert handler.preview_grid_parameters.total_width == 1920
	assert handler.preview_scaling_params is not None
	assert handler.preview_scaling_params[0].scaled_width == 1920
