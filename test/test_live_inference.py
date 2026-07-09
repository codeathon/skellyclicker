"""Unit tests for LiveInferenceService cache and coalescing (no DLC required)."""

from __future__ import annotations

import time

import numpy as np

from skellyclicker.services.live_inference import LiveInferenceService


def test_cache_roundtrip_and_lru():
	svc = LiveInferenceService(cache_size=2)
	svc._ready = True
	svc._store_cache("a.mp4", 1, {"nose": (1.0, 2.0)})
	svc._store_cache("a.mp4", 2, {"nose": (3.0, 4.0)})
	assert svc.get_cached("a.mp4", 1) == {"nose": (1.0, 2.0)}
	# After touching 1, order is (2 oldest, 1 newest). Insert 3 → evict 2.
	svc._store_cache("a.mp4", 3, {"nose": (5.0, 6.0)})
	assert ("a.mp4", 2) not in svc._cache
	assert svc.get_cached("a.mp4", 1) is not None
	assert svc.get_cached("a.mp4", 3) is not None


def test_prediction_to_points_drops_nan_and_low_likelihood():
	coords = np.array(
		[
			[10.0, 20.0, 0.9],
			[np.nan, np.nan, 0.9],
			[1.0, 2.0, 0.01],
		],
		dtype=float,
	)
	pred = {"bodyparts": coords}
	points = LiveInferenceService._prediction_to_points(
		pred, ["nose", "ear", "tail"]
	)
	assert points == {"nose": (10.0, 20.0)}


def test_request_infer_coalesces_to_latest_frame():
	svc = LiveInferenceService()
	svc._ready = True
	calls: list[tuple[str, int]] = []

	def fake_infer(video_path: str, frame_number: int):
		calls.append((video_path, frame_number))
		time.sleep(0.05)
		return {"nose": (0.0, 0.0)}

	svc.infer_frame = fake_infer  # type: ignore[method-assign]
	svc.request_infer("/v/a.mp4", 1)
	svc.request_infer("/v/a.mp4", 2)
	svc.request_infer("/v/a.mp4", 3)
	# Wait for worker to drain.
	deadline = time.time() + 2.0
	while svc._worker is not None and svc._worker.is_alive() and time.time() < deadline:
		time.sleep(0.02)
	assert calls, "expected at least one infer"
	# Latest pending should be 3; may also have started 1 before coalesce.
	assert calls[-1] == ("/v/a.mp4", 3)
	assert all(f in (1, 2, 3) for _, f in calls)


def test_prepare_infer_image_downscales_and_reports_scale():
	frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
	rgb, scale = LiveInferenceService._prepare_infer_image(frame, max_side=512)
	assert rgb.shape[0] == 288  # 1080 * (512/1920)
	assert rgb.shape[1] == 512
	assert abs(scale - (512 / 1920)) < 1e-6


def test_run_inference_scales_points_back_to_native(monkeypatch):
	svc = LiveInferenceService(max_side=512)
	frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

	class _Runner:
		def inference(self, _batch):
			# Coordinates in downscaled space (max_side=512 → scale 0.512).
			return [{"bodyparts": np.array([[100.0, 50.0, 0.9]], dtype=float)}]

	points = LiveInferenceService._run_inference(
		frame, _Runner(), None, ["nose"], max_side=512
	)
	assert "nose" in points
	x, y = points["nose"]
	assert abs(x - 100.0 / 0.512) < 1e-3
	assert abs(y - 50.0 / 0.512) < 1e-3


def test_live_cache_is_display_only_not_csv_handler():
	"""Live predictions live in the service LRU — never require DataHandler upsert."""
	svc = LiveInferenceService()
	svc._store_cache("cam.mp4", 9, {"nose": (11.0, 22.0)})
	assert svc.get_cached("cam.mp4", 9) == {"nose": (11.0, 22.0)}
	assert svc.get_cached("other.mp4", 9) is None
