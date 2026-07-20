"""Warm DeepLabCut runners for on-the-fly single-frame machine labels in the labeler."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Keep recent scrub predictions without unbounded GPU/CPU memory growth.
DEFAULT_CACHE_SIZE = 256
# Downscale long side before DLC — big win for scrub latency; coords scaled back.
# 384 keeps overlays usable while cutting GPU time vs full-res / 512.
LIVE_INFER_MAX_SIDE = 384
# Prefer sequential reads when scrubbing nearby frames (cheaper than random seek).
_SEQUENTIAL_SEEK_MAX_GAP = 8
# Fast scrub rarely hits the exact frame in cache — reuse a nearby prediction.
STICKY_MAX_FRAME_DISTANCE = 45


ProgressCallback = Callable[[float | None, str], None]


class LiveInferenceService:
	"""Session-scoped pose inference for coalesced scrub overlays.

	Loads runners once, caches (video, frame) predictions, and serializes
	inference so scrub can request the latest frame without blocking every tick.
	"""

	def __init__(
		self,
		*,
		cache_size: int = DEFAULT_CACHE_SIZE,
		max_side: int = LIVE_INFER_MAX_SIDE,
	) -> None:
		self._lock = threading.RLock()
		self._cap_lock = threading.Lock()
		self._cache: OrderedDict[tuple[str, int], dict[str, tuple[float, float]]] = (
			OrderedDict()
		)
		# Newest successful infer per video — keeps crosses visible during fast scrub.
		self._last_by_video: dict[str, tuple[int, dict[str, tuple[float, float]]]] = {}
		self._cache_size = max(int(cache_size), 1)
		self._max_side = max(int(max_side), 64)
		self._pose_runner: Any | None = None
		self._detector_runner: Any | None = None
		self._bodyparts: list[str] = []
		self._config_path: str | None = None
		# Fingerprint of loaded weights — reload when iteration/snapshot changes.
		self._weights_fingerprint: str | None = None
		self._ready = False
		self._load_error: str | None = None
		# Reused capture — opening per frame dominated scrub latency.
		self._cap: cv2.VideoCapture | None = None
		self._cap_path: str | None = None
		self._cap_frame_index: int | None = None
		# Coalesced background work: only the newest pending request runs.
		self._pending: tuple[str, int] | None = None
		self._worker: threading.Thread | None = None
		self._on_result: Callable[[str, int, dict[str, tuple[float, float]]], None] | None = (
			None
		)

	@property
	def ready(self) -> bool:
		return self._ready

	@property
	def load_error(self) -> str | None:
		return self._load_error

	@property
	def bodyparts(self) -> list[str]:
		return list(self._bodyparts)

	@property
	def weights_fingerprint(self) -> str | None:
		return self._weights_fingerprint

	def set_result_callback(
		self,
		callback: Callable[[str, int, dict[str, tuple[float, float]]], None] | None,
	) -> None:
		"""Called on the worker thread after a successful background infer."""
		self._on_result = callback

	@staticmethod
	def _fingerprint_for_config(project_config_path: str) -> str:
		"""Identify which trained snapshot would be loaded for this config."""
		from deeplabcut.core.engine import Engine
		import deeplabcut.pose_estimation_pytorch.apis.utils as utils
		from deeplabcut.utils import auxiliaryfunctions

		from skellyclicker.services.dlc_paths import resolve_analyze_iteration

		config_path = Path(project_config_path).expanduser().resolve()
		project_path = config_path.parent
		cfg = auxiliaryfunctions.read_config(str(config_path))
		analyze_iteration = resolve_analyze_iteration(project_path, cfg)
		cfg = dict(cfg)
		cfg["iteration"] = analyze_iteration
		train_fraction = cfg["TrainingFraction"][0]
		model_folder = project_path / auxiliaryfunctions.get_model_folder(
			train_fraction, 1, cfg, engine=Engine.PYTORCH
		)
		train_folder = model_folder / "train"
		model_cfg_path = train_folder / Engine.PYTORCH.pose_cfg_name
		if not model_cfg_path.is_file():
			return f"{config_path}|iter={analyze_iteration}|missing-cfg"
		model_cfg = auxiliaryfunctions.read_plainconfig(model_cfg_path)
		from deeplabcut.pose_estimation_pytorch.task import Task

		pose_task = Task(model_cfg["method"])
		snapshot_index, _ = utils.parse_snapshot_index_for_analysis(
			cfg, model_cfg, None, None
		)
		snapshots = utils.get_model_snapshots(snapshot_index, train_folder, pose_task)
		if not snapshots:
			return f"{config_path}|iter={analyze_iteration}|no-snapshot"
		snap_path = Path(snapshots[0].path)
		mtime = snap_path.stat().st_mtime if snap_path.is_file() else 0.0
		return f"{config_path}|iter={analyze_iteration}|{snap_path.name}|{mtime}"

	def load(
		self,
		project_config_path: str,
		*,
		batch_size: int = 1,
		force: bool = False,
	) -> None:
		"""Build warm pose (+ detector) runners from a DLC project config.yaml.

		``force=True`` always reloads (e.g. after Train Network). Otherwise reload
		when the resolved iteration/snapshot fingerprint changes.
		"""
		config_resolved = str(Path(project_config_path).expanduser().resolve())
		try:
			fingerprint = self._fingerprint_for_config(config_resolved)
		except Exception:
			# Fall back to path-only compare if fingerprinting fails mid-setup.
			fingerprint = config_resolved

		with self._lock:
			if (
				not force
				and self._ready
				and self._config_path == config_resolved
				and self._weights_fingerprint == fingerprint
			):
				return
			self._ready = False
			self._load_error = None
			self._pose_runner = None
			self._detector_runner = None
			self._bodyparts = []
			self._cache.clear()
			self._last_by_video.clear()
			self._pending = None
			self._weights_fingerprint = None
		self._close_capture()

		try:
			self._load_runners(project_config_path, batch_size=batch_size)
			with self._lock:
				self._weights_fingerprint = fingerprint
		except Exception as exc:
			logger.exception("Live inference failed to load model")
			with self._lock:
				self._load_error = str(exc)
				self._ready = False
			raise

	def _load_runners(self, project_config_path: str, *, batch_size: int) -> None:
		from deeplabcut.compat import _update_device
		from deeplabcut.core.engine import Engine
		from deeplabcut.pose_estimation_pytorch.runners import DynamicCropper
		from deeplabcut.pose_estimation_pytorch.task import Task
		import deeplabcut.pose_estimation_pytorch.apis.utils as utils
		from deeplabcut.utils import auxiliaryfunctions

		from skellyclicker.services.dlc_paths import resolve_analyze_iteration

		_update_device(None, {})
		config_path = Path(project_config_path).expanduser().resolve()
		project_path = config_path.parent
		cfg = auxiliaryfunctions.read_config(str(config_path))
		analyze_iteration = resolve_analyze_iteration(project_path, cfg)
		cfg = dict(cfg)
		cfg["iteration"] = analyze_iteration

		train_fraction = cfg["TrainingFraction"][0]
		model_folder = project_path / auxiliaryfunctions.get_model_folder(
			train_fraction, 1, cfg, engine=Engine.PYTORCH
		)
		train_folder = model_folder / "train"
		model_cfg_path = train_folder / Engine.PYTORCH.pose_cfg_name
		if not model_cfg_path.is_file():
			raise FileNotFoundError(
				f"PyTorch model config not found: {model_cfg_path}. Train the network first."
			)
		# create_training_dataset writes the config before any weights exist.
		if not any(train_folder.glob("*.pt")):
			raise FileNotFoundError(
				f"No trained snapshot (*.pt) in {train_folder}. "
				"Train the network before live machine labels."
			)

		model_cfg = auxiliaryfunctions.read_plainconfig(model_cfg_path)
		pose_task = Task(model_cfg["method"])
		snapshot_index, detector_snapshot_index = utils.parse_snapshot_index_for_analysis(
			cfg, model_cfg, None, None
		)

		dynamic = DynamicCropper.build(False, 0.5, 10)
		if pose_task != Task.BOTTOM_UP:
			dynamic = None

		snapshots = utils.get_model_snapshots(snapshot_index, train_folder, pose_task)
		if not snapshots:
			raise FileNotFoundError(
				f"No pose snapshots in {train_folder}. Train the network first."
			)
		snapshot = snapshots[0]
		# batch_size=1 for interactive single-frame scrub.
		pose_runner = utils.get_pose_inference_runner(
			model_config=model_cfg,
			snapshot_path=snapshot.path,
			max_individuals=len(model_cfg["metadata"]["individuals"]),
			batch_size=batch_size,
			dynamic=dynamic,
		)

		detector_runner = None
		if pose_task == Task.TOP_DOWN:
			detector_snapshot = utils.get_model_snapshots(
				detector_snapshot_index, train_folder, Task.DETECT
			)[0]
			detector_runner = utils.get_detector_inference_runner(
				model_config=model_cfg,
				snapshot_path=detector_snapshot.path,
				max_individuals=len(model_cfg["metadata"]["individuals"]),
				batch_size=1,
			)

		bodyparts = list(model_cfg["metadata"]["bodyparts"])
		with self._lock:
			self._pose_runner = pose_runner
			self._detector_runner = detector_runner
			self._bodyparts = bodyparts
			self._config_path = str(config_path)
			self._ready = True
			self._load_error = None
		logger.info(
			"Live inference ready (%s, %d bodyparts, iteration %s, max_side=%d)",
			pose_task,
			len(bodyparts),
			analyze_iteration,
			self._max_side,
		)

	def _close_capture(self) -> None:
		with self._cap_lock:
			if self._cap is not None:
				self._cap.release()
			self._cap = None
			self._cap_path = None
			self._cap_frame_index = None

	def close(self) -> None:
		with self._lock:
			self._pending = None
			self._pose_runner = None
			self._detector_runner = None
			self._bodyparts = []
			self._cache.clear()
			self._last_by_video.clear()
			self._ready = False
			self._config_path = None
			self._weights_fingerprint = None
			self._on_result = None
		self._close_capture()

	def get_cached(
		self, video_name: str, frame_number: int
	) -> dict[str, tuple[float, float]] | None:
		"""Return cached bodypart → (x, y) for a video frame, or None."""
		key = (Path(video_name).name, int(frame_number))
		with self._lock:
			if key not in self._cache:
				return None
			self._cache.move_to_end(key)
			return dict(self._cache[key])

	def get_overlay_points(
		self,
		video_name: str,
		frame_number: int,
		*,
		sticky: bool = False,
		max_distance: int = STICKY_MAX_FRAME_DISTANCE,
	) -> dict[str, tuple[float, float]] | None:
		"""Points for drawing: exact cache hit, or sticky nearby/last during scrub."""
		name = Path(video_name).name
		frame = int(frame_number)
		exact = self.get_cached(name, frame)
		if exact is not None:
			return exact
		if not sticky:
			return None
		with self._lock:
			# Prefer nearest cached frame within max_distance (better than stale last).
			best: tuple[int, dict[str, tuple[float, float]]] | None = None
			best_dist = max_distance + 1
			for (v, f), pts in self._cache.items():
				if v != name:
					continue
				dist = abs(f - frame)
				if dist < best_dist:
					best_dist = dist
					best = (f, pts)
			if best is not None and best_dist <= max_distance:
				return dict(best[1])
			last = self._last_by_video.get(name)
			if last is None:
				return None
			last_frame, pts = last
			if abs(last_frame - frame) <= max_distance:
				return dict(pts)
			# Still show last known during very fast scrub so crosses don't vanish.
			return dict(pts)

	def _store_cache(
		self, video_name: str, frame_number: int, points: dict[str, tuple[float, float]]
	) -> None:
		key = (Path(video_name).name, int(frame_number))
		with self._lock:
			self._cache[key] = points
			self._cache.move_to_end(key)
			self._last_by_video[key[0]] = (key[1], points)
			while len(self._cache) > self._cache_size:
				self._cache.popitem(last=False)

	def infer_frame(
		self, video_path: str, frame_number: int
	) -> dict[str, tuple[float, float]]:
		"""Seek + infer one frame synchronously (own VideoCapture, not labeler caps)."""
		cached = self.get_cached(Path(video_path).name, frame_number)
		if cached is not None:
			return cached
		with self._lock:
			if not self._ready or self._pose_runner is None:
				raise RuntimeError(
					self._load_error or "Live inference model is not loaded"
				)
			pose_runner = self._pose_runner
			detector_runner = self._detector_runner
			bodyparts = list(self._bodyparts)
			max_side = self._max_side

		frame = self._read_frame(video_path, frame_number)
		points = self._run_inference(
			frame, pose_runner, detector_runner, bodyparts, max_side=max_side
		)
		self._store_cache(Path(video_path).name, frame_number, points)
		return points

	def _read_frame(self, video_path: str, frame_number: int) -> np.ndarray:
		"""Read one frame, reusing an open capture and cheap sequential seeks."""
		path = str(video_path)
		target = int(frame_number)
		with self._cap_lock:
			if self._cap is None or self._cap_path != path:
				if self._cap is not None:
					self._cap.release()
				cap = cv2.VideoCapture(path)
				if not cap.isOpened():
					self._cap = None
					self._cap_path = None
					self._cap_frame_index = None
					raise FileNotFoundError(
						f"Could not open video for live inference: {video_path}"
					)
				self._cap = cap
				self._cap_path = path
				self._cap_frame_index = None

			cap = self._cap
			assert cap is not None
			pos = self._cap_frame_index
			# Next frame after last read → one grab; small forward gap → skip; else seek.
			if pos is not None and target == pos + 1:
				pass
			elif (
				pos is not None
				and target > pos
				and target - pos <= _SEQUENTIAL_SEEK_MAX_GAP
			):
				for _ in range(target - pos - 1):
					if not cap.grab():
						break
			else:
				cap.set(cv2.CAP_PROP_POS_FRAMES, target)

			ok, frame = cap.read()
			if not ok or frame is None:
				# One retry with a hard seek — some codecs fail after grab skips.
				cap.set(cv2.CAP_PROP_POS_FRAMES, target)
				ok, frame = cap.read()
			if not ok or frame is None:
				self._cap_frame_index = None
				raise RuntimeError(
					f"Could not read frame {frame_number} from {video_path}"
				)
			self._cap_frame_index = target
			return frame

	@staticmethod
	def _prepare_infer_image(
		frame_bgr: np.ndarray, max_side: int
	) -> tuple[np.ndarray, float]:
		"""RGB image for DLC plus scale factor to map coords back to full frame."""
		h, w = frame_bgr.shape[:2]
		scale = 1.0
		if max(h, w) > max_side:
			scale = max_side / float(max(h, w))
			new_w = max(1, int(round(w * scale)))
			new_h = max(1, int(round(h * scale)))
			frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		return frame_rgb, scale

	@staticmethod
	def _run_inference(
		frame_bgr: np.ndarray,
		pose_runner: Any,
		detector_runner: Any | None,
		bodyparts: list[str],
		*,
		max_side: int = LIVE_INFER_MAX_SIDE,
	) -> dict[str, tuple[float, float]]:
		frame_rgb, scale = LiveInferenceService._prepare_infer_image(frame_bgr, max_side)
		if detector_runner is not None:
			det = detector_runner.inference([frame_rgb])
			bboxes = det[0].get("bboxes") if det else None
			if bboxes is None:
				return {}
			predictions = pose_runner.inference([(frame_rgb, {"bboxes": bboxes})])
		else:
			predictions = pose_runner.inference([frame_rgb])
		if not predictions:
			return {}
		points = LiveInferenceService._prediction_to_points(predictions[0], bodyparts)
		if scale == 1.0:
			return points
		# Map downscaled predictions back to native video coordinates for overlay.
		inv = 1.0 / scale
		return {bp: (x * inv, y * inv) for bp, (x, y) in points.items()}

	@staticmethod
	def _prediction_to_points(
		pred: dict, bodyparts: list[str]
	) -> dict[str, tuple[float, float]]:
		coords = pred["bodyparts"]
		if coords.ndim == 3:
			coords = coords[0]
		points: dict[str, tuple[float, float]] = {}
		for i, bp in enumerate(bodyparts):
			if i >= coords.shape[0]:
				break
			x, y = float(coords[i, 0]), float(coords[i, 1])
			if np.isnan(x) or np.isnan(y):
				continue
			# Drop very low-confidence points when likelihood is present.
			if coords.shape[1] > 2 and float(coords[i, 2]) < 0.1:
				continue
			points[bp] = (x, y)
		return points

	def request_infer(self, video_path: str, frame_number: int) -> None:
		"""Coalesce background inference to the latest (video, frame) request."""
		if not self._ready:
			return
		name = Path(video_path).name
		if self.get_cached(name, frame_number) is not None:
			return
		with self._lock:
			self._pending = (str(video_path), int(frame_number))
			if self._worker is not None and self._worker.is_alive():
				return
			self._worker = threading.Thread(
				target=self._worker_loop, name="live-infer", daemon=True
			)
			self._worker.start()

	def _worker_loop(self) -> None:
		while True:
			with self._lock:
				pending = self._pending
				self._pending = None
			if pending is None:
				break
			video_path, frame_number = pending
			try:
				points = self.infer_frame(video_path, frame_number)
				callback = self._on_result
				if callback is not None:
					callback(Path(video_path).name, frame_number, points)
			except Exception:
				logger.exception(
					"Live inference failed for %s frame %s", video_path, frame_number
				)
			with self._lock:
				if self._pending is None:
					self._worker = None
					break
