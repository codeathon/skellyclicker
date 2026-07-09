"""Tests for transparent synced vs corpus labeling mode detection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from skellyclicker.services.labeling_mode import LabelingMode, detect_labeling_mode


def test_detect_single_video():
	assert detect_labeling_mode(["/a/cam.mp4"]) == LabelingMode.single
	assert detect_labeling_mode([]) == LabelingMode.single


def test_detect_synced_equal_frame_counts():
	with patch("skellyclicker.services.labeling_mode.probe_video_frame_count", return_value=100):
		assert detect_labeling_mode(["/a/cam0.mp4", "/a/cam1.mp4"]) == LabelingMode.synced


def test_detect_corpus_unequal_frame_counts():
	counts = {"/a/expA.mp4": 100, "/b/expB.mp4": 250}

	def _probe(path: str) -> int:
		return counts[path]

	with patch("skellyclicker.services.labeling_mode.probe_video_frame_count", side_effect=_probe):
		assert detect_labeling_mode(list(counts)) == LabelingMode.corpus


def test_session_store_sets_mode_on_add_videos(tmp_path: Path):
	from skellyclicker.services.session_store import SessionStore

	# Tiny fake mp4s are not needed — mock frame probe via detect path.
	a = tmp_path / "a.mp4"
	b = tmp_path / "b.mp4"
	a.write_bytes(b"x")
	b.write_bytes(b"x")
	store = SessionStore()

	with patch(
		"skellyclicker.services.session_store.detect_labeling_mode",
		return_value=LabelingMode.corpus,
	):
		store.add_videos([str(a), str(b)])
	assert store.session.labeling_mode == LabelingMode.corpus
	assert store.session.active_video_path == str(a.resolve())
