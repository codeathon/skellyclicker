"""Tests for sparse CSV export from DataHandler."""

from __future__ import annotations

import pandas as pd

from skellyclicker.core.click_data_handler.data_handler import DataHandler, DataHandlerConfig


def test_save_csv_writes_only_labeled_rows(tmp_path):
	config = DataHandlerConfig(
		num_frames=10_000,
		video_names=["cam1.mp4"],
		tracked_point_names=["nose", "tail"],
	)
	handler = DataHandler.from_config(config)
	handler.dataframe.loc[("cam1.mp4", 12), "nose_x"] = 10.0
	handler.dataframe.loc[("cam1.mp4", 12), "nose_y"] = 20.0
	handler.dataframe.loc[("cam1.mp4", 99), "tail_x"] = 1.0
	handler.dataframe.loc[("cam1.mp4", 99), "tail_y"] = 2.0

	out = tmp_path / "labels.csv"
	handler.save_csv(out)

	saved = pd.read_csv(out)
	assert len(saved) == 2
	assert set(saved["frame"].tolist()) == {12, 99}
