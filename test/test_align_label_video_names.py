"""Tests for aligning CSV video names with session videos."""

from skellyclicker.core.click_data_handler.data_handler import align_label_video_names


def test_align_label_video_names_exact_match():
	assert align_label_video_names(["cam0.mp4"], ["cam0.mp4"]) == {"cam0.mp4": "cam0.mp4"}


def test_align_label_video_names_single_mismatch():
	mapping = align_label_video_names(["old.mp4"], ["new.avi"])
	assert mapping == {"old.mp4": "new.avi"}


def test_align_label_video_names_stem_match():
	mapping = align_label_video_names(["foo.mp4"], ["foo.avi"])
	assert mapping == {"foo.mp4": "foo.avi"}
