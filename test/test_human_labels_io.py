"""Tests for default human label CSV filenames."""

from datetime import datetime

from skellyclicker.core.human_labels_io import human_labels_csv_filename


def test_human_labels_csv_filename_single_video():
	name = human_labels_csv_filename(
		["/data/session/cam_left.mp4"],
		now=datetime(2026, 6, 18, 14, 30, 0),
	)
	assert name == "2026-06-18_14-30-00_cam_left_skellyclicker_labels.csv"


def test_human_labels_csv_filename_multi_video():
	name = human_labels_csv_filename(
		["/data/cam0.mp4", "/data/cam1.mp4"],
		now=datetime(2026, 6, 18, 14, 30, 0),
	)
	assert name == "2026-06-18_14-30-00_cam0_cam1_skellyclicker_labels.csv"
