"""LabelingEngine state_dict includes nav_frame_list."""

from unittest.mock import MagicMock

from skellyclicker.services.labeling_engine import LabelingEngine


def test_state_dict_includes_nav_frame_list():
	human_dh = MagicMock()
	human_dh.get_nonempty_frames.return_value = [0, 5]
	human_dh.active_point = "nose"
	human_dh.config.tracked_point_names = ["nose"]
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
