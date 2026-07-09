"""LabelingEngine state_dict includes nav_frame_list."""

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

	machine_dh = MagicMock()
	machine_dh.get_nonempty_frames.return_value = [5, 20, 40]

	handler = MagicMock()
	handler.data_handler = human_dh
	handler.machine_labels_path = "/tmp/machine.csv"
	handler.machine_labels_handler = machine_dh
	handler.ensure_machine_labels_loaded.return_value = None
	handler.frame_count = 100
	handler.grid_parameters.total_width = 640
	handler.grid_parameters.total_height = 480

	engine = LabelingEngine(video_handler=handler)
	state = engine.state_dict()

	assert state["nav_frame_list"] == [
		{"frame": 0, "kind": "human"},
		{"frame": 5, "kind": "both"},
		{"frame": 20, "kind": "machine"},
		{"frame": 40, "kind": "machine"},
	]
	# Default mode is single → scope to the open video basename.
	human_dh.get_nonempty_frames.assert_called_with("cam.mp4")


def test_corpus_nav_scopes_to_active_video():
	"""Left panel must only list frames for the video selected in the right HUD."""
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
	# Sidecar has samples for both videos — only expB should appear in nav.
	engine._sample_frames_by_video = {
		"expA.mp4": [10, 20],
		"expB.mp4": [50, 60],
	}
	state = engine.state_dict()

	human_dh.get_nonempty_frames.assert_called_with("expB.mp4")
	assert state["nav_frame_list"] == [
		{"frame": 2, "kind": "human"},
		{"frame": 50, "kind": "machine"},
		{"frame": 60, "kind": "machine"},
	]


def test_corpus_legacy_flat_sidecar_falls_back_to_active_video_machine_csv():
	"""Flat sample sidecars must not paint every video with the same predicted frames."""
	human_dh = MagicMock()
	human_dh.get_nonempty_frames.return_value = [2]
	human_dh.active_point = "nose"
	human_dh.config.tracked_point_names = ["nose"]
	human_dh.config.video_names = ["expB.mp4"]
	human_dh.get_data_by_video_frame.return_value = {}

	machine_dh = MagicMock()
	# Only frames that belong to expB in the machine CSV.
	machine_dh.get_nonempty_frames.return_value = [7, 8]

	handler = MagicMock()
	handler.data_handler = human_dh
	handler.machine_labels_path = "/tmp/machine.csv"
	handler.machine_labels_handler = machine_dh
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
	# Legacy flat union (would wrongly show 10/20 on every video if broadcast).
	engine._sample_frames_by_video = {"": [10, 20]}
	state = engine.state_dict()

	machine_dh.get_nonempty_frames.assert_called_with("expB.mp4")
	assert state["nav_frame_list"] == [
		{"frame": 2, "kind": "human"},
		{"frame": 7, "kind": "machine"},
		{"frame": 8, "kind": "machine"},
	]
	assert all(item["frame"] not in (10, 20) for item in state["nav_frame_list"])
