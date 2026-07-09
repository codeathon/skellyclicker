"""Corpus human-label CSV merge keeps other videos when saving one active video."""

from pathlib import Path

import pandas as pd

from skellyclicker.services.human_labels_merge import merge_human_label_rows


def test_merge_replaces_active_video_keeps_others(tmp_path: Path):
	existing = tmp_path / "labels.csv"
	pd.DataFrame(
		{
			"video": ["expA.mp4", "expA.mp4", "expB.mp4"],
			"frame": [1, 2, 10],
			"nose_x": [1.0, 2.0, 3.0],
			"nose_y": [1.0, 2.0, 3.0],
		}
	).to_csv(existing, index=False)

	new_rows = pd.DataFrame(
		{
			"video": ["expB.mp4"],
			"frame": [99],
			"nose_x": [9.0],
			"nose_y": [9.0],
		}
	)
	out = tmp_path / "out.csv"
	merge_human_label_rows(
		existing, new_rows, active_video="expB.mp4", output_path=out
	)
	merged = pd.read_csv(out)
	assert set(merged["video"]) == {"expA.mp4", "expB.mp4"}
	b_rows = merged[merged["video"] == "expB.mp4"]
	assert list(b_rows["frame"]) == [99]
	a_rows = merged[merged["video"] == "expA.mp4"]
	assert sorted(a_rows["frame"].tolist()) == [1, 2]
