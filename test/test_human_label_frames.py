"""Tests for human label frame extraction."""

from pathlib import Path

import pandas as pd
import pytest

from skellyclicker.services.human_label_frames import human_label_frames_per_video


def test_human_label_frames_per_video_sparse(tmp_path: Path):
	csv = tmp_path / "labels.csv"
	pd.DataFrame(
		[
			{"video": "cam0.mp4", "frame": 0, "nose_x": 1.0, "nose_y": 2.0},
			{"video": "cam0.mp4", "frame": 5, "nose_x": 3.0, "nose_y": 4.0},
			{"video": "cam1.mp4", "frame": 0, "nose_x": 10.0, "nose_y": 11.0},
			{"video": "cam1.mp4", "frame": 2, "nose_x": None, "nose_y": None},
		]
	).to_csv(csv, index=False)

	frames = human_label_frames_per_video(csv)
	assert frames == {"cam0.mp4": [0, 5], "cam1.mp4": [0]}


def test_human_label_frames_empty_raises(tmp_path: Path):
	csv = tmp_path / "empty.csv"
	pd.DataFrame(columns=["video", "frame", "nose_x", "nose_y"]).to_csv(csv, index=False)
	with pytest.raises(ValueError, match="no labeled frames"):
		human_label_frames_per_video(csv)
