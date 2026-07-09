"""LabelingEngine state_dict includes human-only nav_frame_list."""

from unittest.mock import MagicMock

from skellyclicker.services.labeling_engine import LabelingEngine
from skellyclicker.services.models import LabelingMode


def test_state_dict_includes_nav_frame_list():
	human_dh = MagicMock()
	human_dh.get_nonempty_frames.return_value = [0, 5]
	human_dh.active_point = "nose"
	human_dh.config.tracked_point_names = ["nose"]
	human_dh.config.video_names = ["cam.mp4"]
	human_dh.get_data_by_video_frame.return_value = {}

	handler = MagicMock()
	handler.data_handler = human_dh
	handler.machine_labels_path = "/tmp/machine.csv"
	handler.machine_labels_handler = MagicMock()
	handler.ensure_machine_labels_loaded.return_value = None
	handler.frame_count = 100
	handler.grid_parameters.total_width = 640
	handler.grid_parameters.total_height = 480

	engine = LabelingEngine(video_handler=handler)
	state = engine.state_dict()

	assert state["nav_frame_list"] == [
		{"frame": 0, "kind": "human"},
		{"frame": 5, "kind": "human"},
	]
	# Default mode is single → scope to the open video basename.
	human_dh.get_nonempty_frames.assert_called_with("cam.mp4")


def test_corpus_nav_scopes_to_active_video():
	"""Left panel must only list human frames for the video selected in the right HUD."""
	human_dh = MagicMock()
	human_dh.get_nonempty_frames.return_value = [2]
	human_dh.active_point = "nose"
	human_dh.config.tracked_point_names = ["nose"]
	human_dh.config.video_names = ["expB.mp4"]
	human_dh.get_data_by_video_frame.return_value = {}

	handler = MagicMock()
	handler.data_handler = human_dh
	handler.machine_labels_path = "/tmp/machine.csv"
	handler.machine_labels_handler = None
	handler.ensure_machine_labels_loaded.return_value = None
	handler.frame_count = 300
	handler.grid_parameters.total_width = 640
	handler.grid_parameters.total_height = 480

	engine = LabelingEngine(
		video_handler=handler,
		labeling_mode=LabelingMode.corpus,
		active_video_path="/data/expB.mp4",
		session_video_paths=["/data/expA.mp4", "/data/expB.mp4"],
	)
	state = engine.state_dict()

	human_dh.get_nonempty_frames.assert_called_with("expB.mp4")
	assert state["nav_frame_list"] == [
		{"frame": 2, "kind": "human"},
	]
