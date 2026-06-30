"""Scrub preview must not race on shared VideoCapture state."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from skellyclicker.services.labeling_engine import LabelingEngine


def test_preview_render_does_not_change_committed_frame():
	handler = MagicMock()
	handler.machine_labels_path = None
	handler.show_machine_labels = False
	handler.create_grid_image.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

	engine = LabelingEngine(video_handler=handler)
	engine.frame_number = 12

	with patch("cv2.imencode", return_value=(True, np.array([1, 2, 3], dtype=np.uint8))):
		engine.render_frame_jpeg(99, preview=True)

	assert engine.frame_number == 12
	handler.create_grid_image.assert_called_once_with(99, annotate_images=False, preview=True)
