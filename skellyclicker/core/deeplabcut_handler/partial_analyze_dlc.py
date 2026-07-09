"""Run DLC inference on human-labeled frames plus a diverse sample and patch machine-labels CSV."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import pandas as pd
from deeplabcut.compat import _update_device
from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.apis.videos import video_inference
import deeplabcut.pose_estimation_pytorch.apis.utils as utils
from deeplabcut.pose_estimation_pytorch.runners import DynamicCropper
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils import auxiliaryfunctions

from skellyclicker import (
	PERF_SAMPLE_FRACTION,
	PERF_SAMPLE_MAX_FRAMES,
	PERF_SAMPLE_MIN_FRAMES,
)
from skellyclicker.core.deeplabcut_handler.dlc_csv_io import dlc_predictions_to_skellyclicker
from skellyclicker.core.deeplabcut_handler.machine_labels_patch import patch_machine_labels_csv
from skellyclicker.core.deeplabcut_handler.performance_sample import evenly_spread_sample
from skellyclicker.core.deeplabcut_handler.selected_frames_video_iterator import (
	SelectedFramesVideoIterator,
)
from skellyclicker.services.dlc_paths import resolve_analyze_iteration
from skellyclicker.services.human_label_frames import human_label_frames_per_video
from skellyclicker.services.sample_frames_sidecar import write_sample_frames

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float | None, str], None]


def _resolve_video_path(video_name: str, video_paths: list[str]) -> str:
	"""Map CSV video basename to a session video path (cross-folder OK)."""
	from skellyclicker.services.video_path_registry import resolve_video_path

	return resolve_video_path(video_name, video_paths)


def _video_frame_count(video_path: str) -> int:
	"""Total frames in a video file (same source as VideoHandler)."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise FileNotFoundError(f"Could not open video: {video_path}")
	try:
		count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		return max(count, 0)
	finally:
		cap.release()


def partial_analyze_human_labels(
	config: str,
	human_labels_csv: str,
	video_paths: list[str],
	machine_labels_csv: str,
	*,
	batch_size: int = 8,
	progress_callback: ProgressCallback | None = None,
) -> str:
	"""Infer human-labeled frames plus a diverse sample; patch the machine-labels CSV."""
	def report(fraction: float | None, message: str) -> None:
		if progress_callback:
			progress_callback(fraction, message)

	frames_per_video = human_label_frames_per_video(human_labels_csv)
	if not frames_per_video:
		raise ValueError("No labeled frames in human labels CSV")

	# Cross-folder videos are OK — each basename resolves via the session registry.
	_update_device(None, {})
	cfg = auxiliaryfunctions.read_config(config)
	config_path = Path(config).expanduser().resolve()
	project_path = config_path.parent
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

	model_cfg = auxiliaryfunctions.read_plainconfig(model_cfg_path)
	pose_task = Task(model_cfg["method"])
	snapshot_index, detector_snapshot_index = utils.parse_snapshot_index_for_analysis(
		cfg, model_cfg, None, None
	)

	cropping = None
	if cfg.get("cropping", False):
		cropping = [cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"]]

	dynamic = DynamicCropper.build(False, 0.5, 10)
	if pose_task != Task.BOTTOM_UP:
		dynamic = None

	snapshot = utils.get_model_snapshots(snapshot_index, train_folder, pose_task)[0]
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
			batch_size=cfg.get("detector_batch_size", 1),
		)

	bodyparts = model_cfg["metadata"]["bodyparts"]
	patch_parts: list[pd.DataFrame] = []
	# Sampled (unseen) frames per video — recorded so the labeler can show
	# them for the active video even when the machine CSV is dense.
	sample_frames_by_video: dict[str, list[int]] = {}
	video_items = [
		(name, frames)
		for name, frames in frames_per_video.items()
		if name in {Path(p).name for p in video_paths}
	]
	total = max(len(video_items), 1)

	for video_index, (video_name, frame_list) in enumerate(video_items):
		video_path = _resolve_video_path(video_name, video_paths)
		labeled = list(frame_list)
		n_video_frames = _video_frame_count(video_path)
		sample = evenly_spread_sample(
			n_video_frames,
			set(labeled),
			fraction=PERF_SAMPLE_FRACTION,
			min_frames=PERF_SAMPLE_MIN_FRAMES,
			max_frames=PERF_SAMPLE_MAX_FRAMES,
		)
		sample_frames_by_video[str(video_name)] = list(sample)
		combined = sorted(set(labeled) | set(sample))
		n_frames = len(combined)
		report(
			0.05 + 0.85 * (video_index / total),
			f"Partial analyze {video_name}: {len(labeled)} labeled + "
			f"{len(sample)} sample frame(s)…",
		)
		logger.info(
			"Partial analyze %s: %d labeled + %d sample (%d total)",
			video_name,
			len(labeled),
			len(sample),
			n_frames,
		)

		iterator = SelectedFramesVideoIterator(
			video_path, combined, cropping=cropping
		)
		predictions = video_inference(
			video=iterator,
			pose_runner=pose_runner,
			detector_runner=detector_runner,
		)
		if len(predictions) != n_frames:
			logger.warning(
				"Expected %d predictions for %s, got %d",
				n_frames,
				video_name,
				len(predictions),
			)

		patch_parts.append(
			dlc_predictions_to_skellyclicker(
				predictions, combined, video_name, bodyparts
			)
		)

	if not patch_parts:
		raise ValueError("No session videos matched human labels CSV video names")

	report(0.92, "Patching machine labels CSV…")
	combined_patch = pd.concat(patch_parts)
	machine_path = Path(machine_labels_csv)
	machine_path.parent.mkdir(parents=True, exist_ok=True)
	source = machine_path if machine_path.is_file() else machine_path
	patch_machine_labels_csv(source, combined_patch, output_path=machine_path)
	# Per-video sample list so corpus left-panel nav filters to the active video.
	write_sample_frames(machine_path, sample_frames_by_video)
	report(1.0, f"Partial analysis complete: {machine_path}")
	return str(machine_path)
