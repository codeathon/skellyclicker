"""Active bodypart persists across frame changes in the web labeler."""

from __future__ import annotations

from unittest.mock import MagicMock

from skellyclicker.core.click_data_handler.data_handler import DataHandler, DataHandlerConfig
from skellyclicker.services.labeling_engine import LabelingEngine


def test_frame_change_keeps_user_selected_active_point():
	config = DataHandlerConfig(
		num_frames=10,
		video_names=["cam.mp4"],
		tracked_point_names=["nose", "tail_base"],
	)
	data_handler = DataHandler.from_config(config)
	data_handler.set_active_point_by_name("tail_base")

	handler = MagicMock()
	handler.data_handler = data_handler
	handler.machine_labels_path = None
	handler.show_machine_labels = False

	engine = LabelingEngine(video_handler=handler)
	engine.frame_number = 7

	assert data_handler.active_point == "tail_base"
