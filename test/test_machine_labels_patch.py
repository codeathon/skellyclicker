"""Tests for machine-labels CSV patching and per-video export."""

from pathlib import Path

import pandas as pd
import pytest

from skellyclicker.core.deeplabcut_handler.machine_labels_patch import (
	export_per_video_machine_csvs,
	patch_machine_labels_csv,
)


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
	assert result.iloc[0]["frame"] == 7


def test_export_per_video_machine_csvs_writes_stem_named_files(tmp_path: Path):
	"""Full Analysis also drops eye1.csv beside eye1.avi (rows for that video only)."""
	vid_dir = tmp_path / "session"
	vid_dir.mkdir()
	eye1 = vid_dir / "eye1.avi"
	eye2 = vid_dir / "eye2.avi"
	eye1.write_bytes(b"")
	eye2.write_bytes(b"")

	combined = tmp_path / "skellyclicker_machine_labels_iteration_0.csv"
	pd.DataFrame(
		[
			{"video": "eye1.avi", "frame": 0, "nose_x": 1.0, "nose_y": 2.0},
			{"video": "eye1.avi", "frame": 1, "nose_x": 3.0, "nose_y": 4.0},
			{"video": "eye2.avi", "frame": 0, "nose_x": 5.0, "nose_y": 6.0},
		]
	).to_csv(combined, index=False)

	written = export_per_video_machine_csvs(
		combined, [str(eye1), str(eye2)]
	)
	assert written == [vid_dir / "eye1.csv", vid_dir / "eye2.csv"]
	eye1_df = pd.read_csv(vid_dir / "eye1.csv")
	eye2_df = pd.read_csv(vid_dir / "eye2.csv")
	assert list(eye1_df["frame"]) == [0, 1]
	assert list(eye2_df["frame"]) == [0]
	assert (eye1_df["video"] == "eye1.avi").all()
	# Combined model_outputs file unchanged.
	assert len(pd.read_csv(combined)) == 3


def test_export_per_video_matches_stem_when_csv_extension_differs(tmp_path: Path):
	"""Merge sometimes labels rows as .mp4 even when the source file is .avi."""
	vid_dir = tmp_path / "clips"
	vid_dir.mkdir()
	video = vid_dir / "eye1.avi"
	video.write_bytes(b"")
	combined = tmp_path / "machine.csv"
	pd.DataFrame(
		[{"video": "eye1.mp4", "frame": 2, "nose_x": 9.0, "nose_y": 8.0}]
	).to_csv(combined, index=False)

	written = export_per_video_machine_csvs(combined, [str(video)])
	assert written[0] == vid_dir / "eye1.csv"
	df = pd.read_csv(written[0])
	assert len(df) == 1
	assert df.iloc[0]["frame"] == 2


def test_export_overwrites_same_stem_csv_without_deleting_others(tmp_path: Path):
	"""New iteration overwrites eye1.csv in place; unrelated CSVs stay."""
	vid_dir = tmp_path / "session"
	vid_dir.mkdir()
	video = vid_dir / "eye1.avi"
	video.write_bytes(b"")
	old = vid_dir / "eye1.csv"
	other = vid_dir / "notes.csv"
	pd.DataFrame(
		[{"video": "eye1.avi", "frame": 0, "nose_x": 1.0, "nose_y": 1.0}]
	).to_csv(old, index=False)
	other.write_text("keep me\n")

	combined = tmp_path / "machine_iter1.csv"
	pd.DataFrame(
		[{"video": "eye1.avi", "frame": 0, "nose_x": 42.0, "nose_y": 43.0}]
	).to_csv(combined, index=False)

	export_per_video_machine_csvs(combined, [str(video)])
	assert pd.read_csv(vid_dir / "eye1.csv").iloc[0]["nose_x"] == 42.0
	assert other.read_text() == "keep me\n"
