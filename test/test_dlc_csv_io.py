"""Tests for DLC analysis CSV parsing."""

from pathlib import Path

import pandas as pd

from skellyclicker.core.deeplabcut_handler.dlc_csv_io import (
	dlc_analysis_csv_to_skellyclicker,
	iter_dlc_video_csvs,
)


def _write_sample_dlc_csv(path: Path, bodyparts: list[str]) -> None:
	header_scorer = ["scorer"] + ["lab"] * (len(bodyparts) * 3)
	header_bp = ["bodyparts"]
	for bp in bodyparts:
		header_bp.extend([bp, bp, bp])
	header_coord = ["coords"]
	for _ in bodyparts:
		header_coord.extend(["x", "y", "likelihood"])
	row0 = [0]
	for i, bp in enumerate(bodyparts):
		row0.extend([100 + i, 200 + i, 0.9])
	rows = [header_scorer, header_bp, header_coord, row0]
	pd.DataFrame(rows).to_csv(path, index=False, header=False)


def test_dlc_csv_to_skellyclicker_columns(tmp_path: Path):
	bodyparts = ["nose", "tail_base"]
	csv_path = tmp_path / "cam0DLC_resnet_50_testNov10shuffle1_500000.csv"
	_write_sample_dlc_csv(csv_path, bodyparts)

	df = dlc_analysis_csv_to_skellyclicker(csv_path, video_name="cam0.mp4")
	assert list(df.columns) == [
		"nose_x",
		"nose_y",
		"nose_likelihood",
		"tail_base_x",
		"tail_base_y",
		"tail_base_likelihood",
	]
	assert df.index.names == ["video", "frame"]
	assert df.loc[("cam0.mp4", 0), "nose_x"] == 100
	assert df.loc[("cam0.mp4", 0), "nose_likelihood"] == 0.9


def test_iter_dlc_video_csvs_excludes_skellyclicker_output(tmp_path: Path):
	_write_sample_dlc_csv(
		tmp_path / "cam0DLC_resnet_50_testNov10shuffle1_500000.csv",
		["nose"],
	)
	(tmp_path / "skellyclicker_machine_labels_iteration_0.csv").write_text("x\n")
	paths = iter_dlc_video_csvs(tmp_path, filtered=False)
	assert len(paths) == 1
	assert "DLC_" in paths[0].name
