"""Tests for DLC tqdm progress bridging."""

from skellyclicker.core.deeplabcut_handler.dlc_progress import (
	ANALYZE_INFERENCE_END,
	ANALYZE_INFERENCE_START,
	ReportingTqdm,
)


def test_reporting_tqdm_maps_frames_to_overall_fraction():
	seen: list[float] = []
	meta = {
		"video_index": 0,
		"video_count": 2,
		"video_name": "cam0.mp4",
		"num_passes": 1,
		"bar_slot": 0,
	}
	bar = ReportingTqdm(
		total=100,
		progress_callback=lambda frac, _msg: seen.append(frac),
		progress_meta=meta,
	)
	bar.update(50)
	bar.close()
	assert seen
	mid = seen[-1]
	assert ANALYZE_INFERENCE_START < mid < ANALYZE_INFERENCE_END
