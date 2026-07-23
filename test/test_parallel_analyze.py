"""Tests for parallel Full Analysis: worker-count policy and progress aggregation."""

import pytest

from skellyclicker.core.deeplabcut_handler import parallel_analyze
from skellyclicker.core.deeplabcut_handler.parallel_analyze import (
	ANALYZE_INFERENCE_END,
	ANALYZE_INFERENCE_START,
	_drain_progress,
	_raw_fraction,
	resolve_worker_count,
)


class _FakeQueue:
	"""Minimal queue: get() pops the next preloaded event."""

	def __init__(self, events):
		self._events = list(events)

	def get(self):
		return self._events.pop(0)


def test_single_video_never_parallel():
	# One video can't benefit from parallel workers.
	assert resolve_worker_count(1, None) == 1
	assert resolve_worker_count(1, 4) == 1


def test_explicit_request_capped_by_video_count(monkeypatch):
	monkeypatch.setattr(parallel_analyze, "available_gpu_count", lambda: 8)
	# 3 videos, asked for 4 -> capped to 3.
	assert resolve_worker_count(3, 4) == 3
	# Asked for 2 with 3 videos -> honored.
	assert resolve_worker_count(3, 2) == 2


def test_auto_uses_gpu_count(monkeypatch):
	monkeypatch.setattr(parallel_analyze, "available_gpu_count", lambda: 2)
	assert resolve_worker_count(4, None) == 2  # two GPUs -> two workers
	assert resolve_worker_count(4, 0) == 2  # 0 also means auto


def test_auto_single_gpu_is_sequential(monkeypatch):
	monkeypatch.setattr(parallel_analyze, "available_gpu_count", lambda: 1)
	assert resolve_worker_count(4, None) == 1
	monkeypatch.setattr(parallel_analyze, "available_gpu_count", lambda: 0)
	assert resolve_worker_count(4, None) == 1  # CPU -> sequential


def test_raw_fraction_unmaps_inference_band():
	assert _raw_fraction(ANALYZE_INFERENCE_START) == pytest.approx(0.0)
	assert _raw_fraction(ANALYZE_INFERENCE_END) == pytest.approx(1.0)
	assert _raw_fraction(None) == 0.0


def test_drain_progress_aggregates_to_band_end():
	# Two videos complete; overall fraction should reach the inference band end.
	events = [
		("progress", 0, "a.mp4", 0.5),
		("progress", 1, "b.mp4", 0.5),
		("done", 0, "a.mp4", 1.0),
		("done", 1, "b.mp4", 1.0),
	]
	seen: list[tuple[float, str]] = []
	_drain_progress(
		progress_queue=_FakeQueue(events),
		procs=[],
		total=2,
		progress_callback=lambda frac, msg: seen.append((frac, msg)),
	)
	# Midpoint (both at 0.5) sits halfway through the band.
	mid = ANALYZE_INFERENCE_START + (ANALYZE_INFERENCE_END - ANALYZE_INFERENCE_START) * 0.5
	assert any(frac == pytest.approx(mid) for frac, _ in seen)
	# Final reaches the band end when both videos are done.
	assert seen[-1][0] == pytest.approx(ANALYZE_INFERENCE_END)


def test_drain_progress_raises_on_worker_error():
	events = [
		("progress", 0, "a.mp4", 0.3),
		("error", 1, "b.mp4", "RuntimeError: CUDA out of memory"),
	]
	with pytest.raises(RuntimeError, match="CUDA out of memory"):
		_drain_progress(
			progress_queue=_FakeQueue(events),
			procs=[],
			total=2,
			progress_callback=None,
		)
