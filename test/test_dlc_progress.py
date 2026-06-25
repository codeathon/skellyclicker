"""Tests for DLC tqdm progress bridging."""

from skellyclicker.core.deeplabcut_handler.dlc_progress import (
	ANALYZE_INFERENCE_END,
	ANALYZE_INFERENCE_START,
	TRAIN_PROGRESS_END,
	TRAIN_PROGRESS_START,
	ReportingTqdm,
	hook_dlc_training_progress,
	train_epoch_fraction,
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


def test_train_epoch_fraction_maps_epochs_into_train_slice():
	assert train_epoch_fraction(0, 200) == TRAIN_PROGRESS_START
	end = train_epoch_fraction(200, 200)
	assert abs(end - TRAIN_PROGRESS_END) < 1e-9
	mid = train_epoch_fraction(100, 200)
	assert TRAIN_PROGRESS_START < mid < TRAIN_PROGRESS_END


def test_hook_dlc_training_progress_forwards_epoch_logs():
	seen: list[tuple[float | None, str]] = []

	def capture(fraction: float | None, message: str) -> None:
		seen.append((fraction, message))

	with hook_dlc_training_progress(capture):
		logging_info = __import__("logging").info
		logging_info("Epoch 10/200 (lr=0.001), train loss 0.54321")

	assert len(seen) == 1
	fraction, message = seen[0]
	assert fraction is not None
	assert 0.15 < fraction < 0.95
	assert message == "Epoch 10/200 · train loss 0.54321"
