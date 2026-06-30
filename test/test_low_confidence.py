"""Tests for low-confidence machine label queue."""

from pathlib import Path

import pandas as pd

from skellyclicker.core.click_data_handler.data_handler import DataHandler
from skellyclicker.services.low_confidence import (
	DEFAULT_LIKELIHOOD_THRESHOLD,
	find_low_confidence_items,
)


def _write_machine_csv(path: Path) -> None:
	df = pd.DataFrame(
		{
			"video": ["cam.mp4", "cam.mp4", "cam.mp4"],
			"frame": [0, 1, 1],
			"nose_x": [10.0, 20.0, 30.0],
			"nose_y": [11.0, 21.0, 31.0],
			"nose_likelihood": [0.9, 0.3, 0.8],
			"tail_x": [5.0, 6.0, 7.0],
			"tail_y": [6.0, 7.0, 8.0],
			"tail_likelihood": [0.95, 0.95, 0.4],
		}
	)
	df.to_csv(path, index=False)


def test_find_low_confidence_items_below_threshold(tmp_path):
	csv_path = tmp_path / "machine.csv"
	_write_machine_csv(csv_path)
	handler = DataHandler.from_csv_overlay(
		csv_path,
		video_names=["cam.mp4"],
		num_frames=3,
		tracked_point_names=["nose", "tail"],
	)
	items = find_low_confidence_items(handler, threshold=DEFAULT_LIKELIHOOD_THRESHOLD)
	assert len(items) == 2
	assert items[0]["frame_number"] == 1
	assert items[0]["bodypart"] == "nose"
	assert items[0]["likelihood"] == 0.3
	assert items[1]["bodypart"] == "tail"


def test_find_low_confidence_items_without_likelihood_columns(tmp_path):
	csv_path = tmp_path / "machine.csv"
	pd.DataFrame(
		{
			"video": ["cam.mp4"],
			"frame": [0],
			"nose_x": [1.0],
			"nose_y": [2.0],
		}
	).to_csv(csv_path, index=False)
	handler = DataHandler.from_csv_overlay(
		csv_path,
		video_names=["cam.mp4"],
		num_frames=1,
		tracked_point_names=["nose"],
	)
	assert find_low_confidence_items(handler) == []
