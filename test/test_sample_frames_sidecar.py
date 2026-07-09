"""Tests for the partial-analyze sample-frames sidecar."""

from skellyclicker.services.sample_frames_sidecar import (
	read_sample_frames,
	read_sample_frames_by_video,
	sample_frames_for_video,
	sample_frames_sidecar_path,
	write_sample_frames,
)


def test_sidecar_path_next_to_csv(tmp_path):
	csv = tmp_path / "skellyclicker_machine_labels_iteration_1.csv"
	path = sample_frames_sidecar_path(csv)
	assert path == tmp_path / "skellyclicker_machine_labels_iteration_1.sample_frames.json"


def test_write_then_read_roundtrip(tmp_path):
	csv = tmp_path / "machine.csv"
	write_sample_frames(csv, [30, 10, 20, 10])
	assert read_sample_frames(csv) == [10, 20, 30]


def test_write_per_video_and_filter(tmp_path):
	csv = tmp_path / "machine.csv"
	write_sample_frames(
		csv,
		{"expA.mp4": [1, 5], "expB.mp4": [100, 200]},
	)
	by_video = read_sample_frames_by_video(csv)
	assert by_video == {"expA.mp4": [1, 5], "expB.mp4": [100, 200]}
	assert read_sample_frames(csv) == [1, 5, 100, 200]
	assert sample_frames_for_video(by_video, "expA.mp4") == [1, 5]
	assert sample_frames_for_video(by_video, "expB.mp4") == [100, 200]
	assert sample_frames_for_video(by_video, "missing.mp4") == []
	# Synced / no active video → union.
	assert sample_frames_for_video(by_video, None) == [1, 5, 100, 200]


def test_legacy_flat_sidecar_does_not_broadcast_to_one_video(tmp_path):
	"""Old flat sample lists must not appear as predicted frames on every corpus video."""
	csv = tmp_path / "machine.csv"
	write_sample_frames(csv, [10, 20])
	by_video = read_sample_frames_by_video(csv)
	# Caller should fall back to that video's machine CSV instead.
	assert sample_frames_for_video(by_video, "expA.mp4") is None
	assert sample_frames_for_video(by_video, None) == [10, 20]


def test_missing_video_in_per_video_sidecar_returns_empty(tmp_path):
	csv = tmp_path / "machine.csv"
	write_sample_frames(csv, {"expA.mp4": [1, 5]})
	by_video = read_sample_frames_by_video(csv)
	assert sample_frames_for_video(by_video, "expB.mp4") == []



def test_read_missing_sidecar_returns_none(tmp_path):
	csv = tmp_path / "machine.csv"
	assert read_sample_frames(csv) is None
	assert read_sample_frames_by_video(csv) is None


def test_read_invalid_sidecar_returns_none(tmp_path):
	csv = tmp_path / "machine.csv"
	sample_frames_sidecar_path(csv).write_text("{ not json")
	assert read_sample_frames(csv) is None


def test_write_empty_sample_frames(tmp_path):
	csv = tmp_path / "machine.csv"
	write_sample_frames(csv, [])
	assert read_sample_frames(csv) == []
