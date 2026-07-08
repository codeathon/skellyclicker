"""Tests for diverse performance-sample frame selection."""

from skellyclicker import (
	PERF_SAMPLE_FRACTION,
	PERF_SAMPLE_MAX_FRAMES,
	PERF_SAMPLE_MIN_FRAMES,
)
from skellyclicker.core.deeplabcut_handler.performance_sample import evenly_spread_sample


def test_evenly_spread_clamps_to_max_on_large_video():
	frames = evenly_spread_sample(
		300_000,
		set(),
		fraction=PERF_SAMPLE_FRACTION,
		min_frames=PERF_SAMPLE_MIN_FRAMES,
		max_frames=PERF_SAMPLE_MAX_FRAMES,
	)
	assert len(frames) <= PERF_SAMPLE_MAX_FRAMES
	assert frames[0] == 0
	assert frames[-1] == 299_999


def test_evenly_spread_uses_min_on_small_video():
	frames = evenly_spread_sample(
		500,
		set(),
		fraction=PERF_SAMPLE_FRACTION,
		min_frames=PERF_SAMPLE_MIN_FRAMES,
		max_frames=PERF_SAMPLE_MAX_FRAMES,
	)
	assert len(frames) == PERF_SAMPLE_MIN_FRAMES


def test_evenly_spread_excludes_labeled_frames():
	exclude = {100, 200, 300}
	frames = evenly_spread_sample(
		1000,
		exclude,
		fraction=PERF_SAMPLE_FRACTION,
		min_frames=PERF_SAMPLE_MIN_FRAMES,
		max_frames=PERF_SAMPLE_MAX_FRAMES,
	)
	assert not any(f in exclude for f in frames)


def test_evenly_spread_dedupes_and_sorted():
	frames = evenly_spread_sample(
		50,
		set(),
		fraction=PERF_SAMPLE_FRACTION,
		min_frames=PERF_SAMPLE_MIN_FRAMES,
		max_frames=PERF_SAMPLE_MAX_FRAMES,
	)
	assert frames == sorted(set(frames))


def test_evenly_spread_empty_video():
	assert evenly_spread_sample(
		0,
		set(),
		fraction=PERF_SAMPLE_FRACTION,
		min_frames=PERF_SAMPLE_MIN_FRAMES,
		max_frames=PERF_SAMPLE_MAX_FRAMES,
	) == []
