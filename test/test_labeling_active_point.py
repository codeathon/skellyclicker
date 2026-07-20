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


def test_auto_next_disabled_when_frame_has_human_and_machine_labels():
	from unittest.mock import MagicMock, patch

	mock_handler = MagicMock()
	mock_handler.image_annotator.config = MagicMock()
	mock_handler.machine_labels_path = "/tmp/machine.csv"
	mock_handler.machine_labels_handler = None

	human_dh = MagicMock()
	human_dh.config.video_names = ["cam.mp4"]
	human_dh.get_data_by_video_frame.return_value = {"nose": MagicMock()}
	mock_handler.data_handler = human_dh

	machine_dh = MagicMock()
	machine_dh.config.video_names = ["cam.mp4"]
	machine_dh.get_data_by_video_frame.return_value = {"nose": MagicMock()}
	mock_handler.ensure_machine_labels_loaded.side_effect = lambda: setattr(
		mock_handler, "machine_labels_handler", machine_dh
	)

	with patch(
		"skellyclicker.services.labeling_engine.VideoHandler.from_videos_for_labeler",
		return_value=mock_handler,
	):
		engine = LabelingEngine.open(
			video_paths=["/tmp/cam.mp4"],
			human_labels_path="/tmp/human.csv",
			machine_labels_path="/tmp/machine.csv",
			train_on_machine_labels=False,
			tracked_point_names=["nose"],
		)

	assert engine.auto_next_point is False


def test_auto_next_enabled_on_frame_with_machine_but_no_human_labels():
	from unittest.mock import MagicMock, patch

	mock_handler = MagicMock()
	mock_handler.image_annotator.config = MagicMock()
	mock_handler.machine_labels_path = "/tmp/machine.csv"
	mock_handler.machine_labels_handler = MagicMock()
	mock_handler.machine_labels_handler.config.video_names = ["cam.mp4"]
	mock_handler.machine_labels_handler.get_data_by_video_frame.return_value = {}

	human_dh = MagicMock()
	human_dh.config.video_names = ["cam.mp4"]
	human_dh.get_data_by_video_frame.return_value = {}
	mock_handler.data_handler = human_dh
	mock_handler.ensure_machine_labels_loaded.return_value = None

	with patch(
		"skellyclicker.services.labeling_engine.VideoHandler.from_videos_for_labeler",
		return_value=mock_handler,
	):
		engine = LabelingEngine.open(
			video_paths=["/tmp/cam.mp4"],
			human_labels_path="/tmp/human.csv",
			machine_labels_path="/tmp/machine.csv",
			train_on_machine_labels=False,
			tracked_point_names=["nose", "tail_base"],
		)

	assert engine.auto_next_point is True


def test_set_frame_updates_auto_next_for_review_frames():
	from unittest.mock import MagicMock

	human_dh = MagicMock()
	human_dh.config.video_names = ["cam.mp4"]
	# Leaving a frame is allowed when labeling is complete / empty.
	human_dh.incomplete_labeling_message.return_value = None
	human_dh.get_data_by_video_frame.side_effect = [
		{},
		{"nose": MagicMock()},
	]

	machine_dh = MagicMock()
	machine_dh.config.video_names = ["cam.mp4"]
	machine_dh.get_data_by_video_frame.return_value = {"nose": MagicMock()}

	handler = MagicMock()
	handler.data_handler = human_dh
	handler.machine_labels_path = "/tmp/machine.csv"
	handler.machine_labels_handler = machine_dh
	handler.ensure_machine_labels_loaded.return_value = None

	engine = LabelingEngine(video_handler=handler)
	engine.set_frame(0)
	assert engine.auto_next_point is True

	engine.set_frame(5)
	assert engine.auto_next_point is False


def test_auto_next_stays_enabled_while_labeling_machine_only_frame():
	from skellyclicker.core.video_handler.video_models import ClickData

	config = DataHandlerConfig(
		num_frames=10,
		video_names=["cam.mp4"],
		tracked_point_names=["nose", "tail_base", "ear"],
	)
	data_handler = DataHandler.from_config(config)

	click_handler = MagicMock()
	click = ClickData(
		video_index=0,
		frame_number=0,
		video_x=10,
		video_y=20,
		window_x=1,
		window_y=2,
	)
	click_handler.process_click.return_value = click

	machine_dh = MagicMock()
	machine_dh.config.video_names = ["cam.mp4"]
	machine_dh.get_data_by_video_frame.return_value = {"nose": MagicMock()}

	handler = MagicMock()
	handler.data_handler = data_handler
	handler.click_handler = click_handler
	handler.machine_labels_path = "/tmp/machine.csv"
	handler.machine_labels_handler = machine_dh
	handler.ensure_machine_labels_loaded.return_value = None
	handler.create_grid_image.return_value = None

	engine = LabelingEngine(video_handler=handler)
	engine.set_frame(0)
	assert engine.auto_next_point is True
	assert data_handler.active_point == "nose"

	engine.handle_click(1, 2)
	assert data_handler.active_point == "tail_base"
	assert engine.auto_next_point is True

	engine.handle_click(1, 2)
	assert data_handler.active_point == "ear"
	assert engine.auto_next_point is True


def test_committed_render_on_same_frame_keeps_auto_next_enabled():
	import numpy as np
	from unittest.mock import patch

	human_dh = MagicMock()
	human_dh.config.video_names = ["cam.mp4"]
	human_dh.get_data_by_video_frame.side_effect = [
		{},
		{"nose": MagicMock()},
	]

	handler = MagicMock()
	handler.data_handler = human_dh
	handler.machine_labels_path = "/tmp/machine.csv"
	handler.machine_labels_handler = MagicMock()
	handler.machine_labels_handler.config.video_names = ["cam.mp4"]
	handler.machine_labels_handler.get_data_by_video_frame.return_value = {"nose": MagicMock()}
	handler.ensure_machine_labels_loaded.return_value = None
	handler.show_machine_labels = False
	handler.create_grid_image.return_value = np.zeros((8, 8, 3), dtype=np.uint8)

	engine = LabelingEngine(video_handler=handler)
	engine.set_frame(0)
	assert engine.auto_next_point is True

	with patch("cv2.imencode", return_value=(True, np.array([1], dtype=np.uint8))):
		engine.render_frame_jpeg(0, preview=False)

	assert engine.auto_next_point is True


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


def test_toggle_show_names_updates_annotators():
	from unittest.mock import MagicMock

	handler = MagicMock()
	handler.image_annotator.config = MagicMock(show_names=True, show_help=False)
	handler.machine_labels_annotator = MagicMock()
	handler.machine_labels_annotator.config = MagicMock(show_names=True)

	engine = LabelingEngine(video_handler=handler, show_names=True)
	engine.toggle_show_names()

	assert engine.show_names is False
	assert handler.image_annotator.config.show_names is False
	assert handler.machine_labels_annotator.config.show_names is False
