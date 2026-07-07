"""Tests for machine-labels CSV patching."""

from pathlib import Path

import pandas as pd
import pytest

from skellyclicker.core.deeplabcut_handler.machine_labels_patch import patch_machine_labels_csv


def test_patch_updates_existing_rows(tmp_path: Path):
	existing = tmp_path / "machine.csv"
	pd.DataFrame(
		[
			{"video": "cam0.mp4", "frame": 0, "nose_x": 1.0, "nose_y": 2.0},
			{"video": "cam0.mp4", "frame": 1, "nose_x": 3.0, "nose_y": 4.0},
		]
	).to_csv(existing, index=False)

	patch = pd.DataFrame(
		[{"video": "cam0.mp4", "frame": 0, "nose_x": 99.0, "nose_y": 88.0, "nose_likelihood": 0.95}]
	).set_index(["video", "frame"])

	out = patch_machine_labels_csv(existing, patch)
	result = pd.read_csv(out)
	row0 = result[(result["video"] == "cam0.mp4") & (result["frame"] == 0)].iloc[0]
	assert row0["nose_x"] == 99.0
	assert row0["nose_likelihood"] == pytest.approx(0.95)
	row1 = result[(result["video"] == "cam0.mp4") & (result["frame"] == 1)].iloc[0]
	assert row1["nose_x"] == 3.0


def test_patch_creates_file_when_missing(tmp_path: Path):
	missing = tmp_path / "new_machine.csv"
	patch = pd.DataFrame(
		[{"video": "cam0.mp4", "frame": 7, "nose_x": 5.0, "nose_y": 6.0}]
	).set_index(["video", "frame"])
	out = patch_machine_labels_csv(missing, patch)
	assert out.is_file()
	result = pd.read_csv(out)
	assert len(result) == 1
	assert int(result.iloc[0]["frame"]) == 7
