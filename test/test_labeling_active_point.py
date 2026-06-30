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


def test_handle_click_does_not_advance_active_point():
	from unittest.mock import MagicMock

	config = DataHandlerConfig(
		num_frames=10,
		video_names=["cam.mp4"],
		tracked_point_names=["nose", "tail_base"],
	)
	data_handler = DataHandler.from_config(config)
	data_handler.set_active_point_by_name("nose")

	click_handler = MagicMock()
	click_data = MagicMock()
	click_data.video_index = 0
	click_data.frame_number = 0
	click_data.x = 50
	click_data.y = 60
	click_handler.process_click.return_value = click_data

	handler = MagicMock()
	handler.data_handler = data_handler
	handler.click_handler = click_handler

	from skellyclicker.core.video_handler.video_handler import VideoHandler

	VideoHandler.handle_clicks(handler, 50, 60, 0, auto_next_point=False)

	assert data_handler.active_point == "nose"


def test_open_disables_auto_next_when_human_labels_loaded():
	from unittest.mock import MagicMock, patch

	mock_handler = MagicMock()
	mock_handler.image_annotator.config = MagicMock()

	with patch(
		"skellyclicker.services.labeling_engine.VideoHandler.from_videos_for_labeler",
		return_value=mock_handler,
	):
		with_labels = LabelingEngine.open(
			video_paths=["/tmp/cam.mp4"],
			human_labels_path="/tmp/human.csv",
			machine_labels_path=None,
			train_on_machine_labels=False,
			tracked_point_names=["nose"],
		)
		fresh = LabelingEngine.open(
			video_paths=["/tmp/cam.mp4"],
			human_labels_path=None,
			machine_labels_path=None,
			train_on_machine_labels=False,
			tracked_point_names=["nose"],
		)

	assert with_labels.auto_next_point is False
	assert fresh.auto_next_point is True


def test_undo_last_label_removes_fresh_placement():
	from unittest.mock import MagicMock

	from skellyclicker.core.video_handler.video_models import ClickData

	config = DataHandlerConfig(
		num_frames=10,
		video_names=["cam.mp4"],
		tracked_point_names=["nose", "tail_base"],
	)
	data_handler = DataHandler.from_config(config)

	click_handler = MagicMock()
	click = ClickData(
		video_index=0,
		frame_number=2,
		video_x=40,
		video_y=50,
		window_x=10,
		window_y=20,
	)
	click_handler.process_click.return_value = click

	handler = MagicMock()
	handler.data_handler = data_handler
	handler.click_handler = click_handler

	engine = LabelingEngine(video_handler=handler, auto_next_point=False)
	engine.frame_number = 2
	engine.handle_click(10, 20)

	assert data_handler.point_is_labeled(0, 2, "nose")
	assert engine.undo_last_label() is True
	assert not data_handler.point_is_labeled(0, 2, "nose")
	assert data_handler.active_point == "nose"
	assert engine.frame_number == 2


def test_undo_last_label_restores_previous_coords():
	from unittest.mock import MagicMock

	from skellyclicker.core.video_handler.video_models import ClickData

	config = DataHandlerConfig(
		num_frames=10,
		video_names=["cam.mp4"],
		tracked_point_names=["nose"],
	)
	data_handler = DataHandler.from_config(config)
	data_handler.set_point_coords(0, 1, "nose", 12.0, 13.0)

	click_handler = MagicMock()
	click = ClickData(
		video_index=0,
		frame_number=1,
		video_x=99,
		video_y=88,
		window_x=5,
		window_y=6,
	)
	click_handler.process_click.return_value = click

	handler = MagicMock()
	handler.data_handler = data_handler
	handler.click_handler = click_handler

	engine = LabelingEngine(video_handler=handler, auto_next_point=False)
	engine.frame_number = 1
	engine.handle_click(5, 6)

	x, y = data_handler.get_point_coords(0, 1, "nose")
	assert (x, y) == (99.0, 88.0)
	assert engine.undo_last_label() is True
	x, y = data_handler.get_point_coords(0, 1, "nose")
	assert (x, y) == (12.0, 13.0)
