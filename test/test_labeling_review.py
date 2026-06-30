"""Review mode seeds machine labels into human dataframe."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from skellyclicker.core.click_data_handler.data_handler import DataHandler, DataHandlerConfig
from skellyclicker.core.video_handler.video_handler import VideoHandler


def _machine_csv(path: Path) -> None:
	pd.DataFrame(
		{
			"video": ["cam.mp4"],
			"frame": [5],
			"nose_x": [100.0],
			"nose_y": [200.0],
			"nose_likelihood": [0.2],
		}
	).to_csv(path, index=False)


def test_seed_frame_from_machine_labels_copies_to_human(tmp_path):
	machine = tmp_path / "machine.csv"
	_machine_csv(machine)
	config = DataHandlerConfig(
		num_frames=10,
		video_names=["cam.mp4"],
		tracked_point_names=["nose"],
	)
	human = DataHandler.from_config(config)
	machine_handler = DataHandler.from_csv_overlay(
		machine,
		video_names=["cam.mp4"],
		num_frames=10,
		tracked_point_names=["nose"],
	)

	handler = MagicMock(spec=VideoHandler)
	handler.videos = {"cam.mp4": MagicMock()}
	handler.machine_labels_handler = machine_handler
	handler.data_handler = human
	handler.ensure_machine_labels_loaded = MagicMock()

	def _copy(frame_number: int, video_index: int) -> None:
		data = machine_handler.get_data_by_video_frame(video_index, frame_number)
		for name, click in data.items():
			human.update_dataframe(click_data=click, point_name=name)

	handler.copy_frame_data_from_machine_labels = _copy

	VideoHandler.seed_frame_from_machine_labels(handler, 5)
	click = human.get_data_by_video_frame(0, 5)
	assert "nose" in click
	assert click["nose"].x == 100
	assert click["nose"].y == 200
