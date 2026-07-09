"""Tests for transparent synced vs corpus labeling mode detection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from skellyclicker.services.labeling_mode import LabelingMode, detect_labeling_mode
from skellyclicker.services.models import LabelingMode as ModelLabelingMode

# Keep a single LabelingMode symbol (models re-exports the same enum).
assert LabelingMode is ModelLabelingMode


def test_detect_single_video():
	assert detect_labeling_mode(["/a/cam.mp4"]) == LabelingMode.single
	assert detect_labeling_mode([]) == LabelingMode.single


def test_detect_synced_equal_frame_counts():
	with patch("skellyclicker.services.labeling_mode.probe_video_frame_count", return_value=100):
		assert detect_labeling_mode(["/a/cam0.mp4", "/a/cam1.mp4"]) == LabelingMode.synced


def test_detect_corpus_when_frame_count_is_zero():
	"""CAP_PROP 0 is unreliable — must not open as synced multi-cam."""
	with patch("skellyclicker.services.labeling_mode.probe_video_frame_count", return_value=0):
		assert detect_labeling_mode(["/a/cam0.mp4", "/a/cam1.mp4"]) == LabelingMode.corpus


def test_detect_corpus_different_parent_folders(tmp_path: Path):
	"""Videos from different experiment folders are corpus even if lengths match."""
	a = tmp_path / "expA" / "cam.mp4"
	b = tmp_path / "expB" / "cam.mp4"
	a.parent.mkdir()
	b.parent.mkdir()
	a.write_bytes(b"x")
	b.write_bytes(b"x")
	with patch("skellyclicker.services.labeling_mode.probe_video_frame_count", return_value=100):
		assert detect_labeling_mode([str(a), str(b)]) == LabelingMode.corpus


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


def test_labeler_paths_downgrade_synced_when_counts_disagree(tmp_path: Path):
	"""Stale synced mode must not open unequal videos as a grid (500)."""
	from skellyclicker.services.session_store import SessionStore

	a = tmp_path / "short.mp4"
	b = tmp_path / "long.mp4"
	a.write_bytes(b"x")
	b.write_bytes(b"x")
	store = SessionStore()
	store.session.videos = [str(a), str(b)]
	store.session.labeling_mode = LabelingMode.synced
	store.session.active_video_path = None

	counts = {str(a): 10, str(b): 99}

	def _probe(path: str) -> int:
		return counts[path]

	with patch(
		"skellyclicker.services.session_store.detect_labeling_mode",
		return_value=LabelingMode.synced,
	), patch(
		"skellyclicker.services.labeling_mode.probe_video_frame_count",
		side_effect=_probe,
	):
		paths = store._labeler_video_paths()

	assert store.session.labeling_mode == LabelingMode.corpus
	assert paths == [str(a)]


def test_open_labeler_falls_back_when_synced_open_rejects_unequal(tmp_path: Path):
	"""ValueError from VideoHandler must become corpus open, not HTTP 500."""
	from skellyclicker.services.session_store import SessionStore

	a = tmp_path / "a.mp4"
	b = tmp_path / "b.mp4"
	a.write_bytes(b"x")
	b.write_bytes(b"x")
	store = SessionStore()
	store.session.videos = [str(a.resolve()), str(b.resolve())]
	store.session.labeling_mode = LabelingMode.synced
	store.session.tracked_point_names = ["nose"]
	store.session.dlc_project_path = str(tmp_path)

	fake_engine = MagicMock()
	fake_engine.session_id = "sid"
	fake_engine.video_handler.frame_count = 10
	fake_engine.video_handler.data_handler.config.tracked_point_names = ["nose"]

	calls: list[list[str]] = []

	def _open(**kwargs):
		calls.append(list(kwargs["video_paths"]))
		if len(kwargs["video_paths"]) > 1:
			raise ValueError("All videos must have the same number of images")
		return fake_engine

	# Soft-verify sees equal counts so synced paths are returned; open then fails.
	with patch.object(store, "_refresh_labeling_mode"), patch.object(
		store, "_ensure_live_inference"
	), patch(
		"skellyclicker.services.labeling_mode.probe_video_frame_count",
		return_value=100,
	), patch(
		"skellyclicker.services.labeling_engine.LabelingEngine.open",
		side_effect=_open,
	):
		store.session.labeling_mode = LabelingMode.synced
		session = store.open_labeler()

	assert session.labeling_mode == LabelingMode.corpus
	assert calls[0] == [str(a.resolve()), str(b.resolve())]
	assert calls[1] == [str(a.resolve())]
	assert store.labeling_engine is fake_engine
