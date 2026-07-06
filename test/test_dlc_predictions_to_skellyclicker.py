"""Tests for DLC prediction → skellyclicker CSV conversion."""

import numpy as np
import pandas as pd

from skellyclicker.core.deeplabcut_handler.dlc_csv_io import dlc_predictions_to_skellyclicker


def test_dlc_predictions_to_skellyclicker_single_animal():
	predictions = [
		{"bodyparts": np.array([[[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]]])},
		{"bodyparts": np.array([[[5.0, 6.0, 0.7], [7.0, 8.0, 0.6]]])},
	]
	df = dlc_predictions_to_skellyclicker(
		predictions, [0, 10], "cam0.mp4", ["nose", "tail"]
	)
	assert df.loc[("cam0.mp4", 0), "nose_x"] == 1.0
	assert df.loc[("cam0.mp4", 10), "tail_likelihood"] == 0.6
	assert list(df.index.get_level_values("frame")) == [0, 10]
