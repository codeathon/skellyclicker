"""Tests for the partial-analyze sample-frames sidecar."""

from skellyclicker.services.sample_frames_sidecar import (
	read_sample_frames,
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


def test_read_missing_sidecar_returns_none(tmp_path):
	csv = tmp_path / "machine.csv"
	assert read_sample_frames(csv) is None


def test_read_invalid_sidecar_returns_none(tmp_path):
	csv = tmp_path / "machine.csv"
	sample_frames_sidecar_path(csv).write_text("{ not json")
	assert read_sample_frames(csv) is None


def test_write_empty_sample_frames(tmp_path):
	csv = tmp_path / "machine.csv"
	write_sample_frames(csv, [])
	assert read_sample_frames(csv) == []
