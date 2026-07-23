"""Run DLC video analysis in parallel across GPUs — one worker process per device.

Why processes (not threads): CUDA contexts and DLC's pose runner are not safe to
share across threads or to pickle into a Pool. Instead we spawn independent
processes, pin each to a distinct ``cuda:N`` device, and let each pull whole
videos from a shared queue. Each worker calls the existing single-device
``analyze_videos_dlc`` for one video at a time, so no DLC setup is duplicated and
the sequential filter/merge/plot stages downstream are unchanged.

Real speedup needs multiple GPUs (one video per GPU). On a single GPU this falls
back to the normal sequential path (see ``resolve_worker_count``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float | None, str], None]

# Overall analyze band reserved for inference progress. Mirrors the sequential
# path in dlc_progress.py; defined here so this supervisor (and its unit tests)
# stay importable without the tqdm-bound progress module.
ANALYZE_INFERENCE_START = 0.05
ANALYZE_INFERENCE_END = 0.75
_INFER_START = ANALYZE_INFERENCE_START
_INFER_END = ANALYZE_INFERENCE_END
_INFER_SPAN = _INFER_END - _INFER_START


def available_gpu_count() -> int:
	"""CUDA device count, or 0 when torch/CUDA is unavailable."""
	try:
		import torch

		if torch.cuda.is_available():
			return int(torch.cuda.device_count())
	except Exception as exc:  # torch missing / driver error — treat as CPU.
		logger.debug("CUDA probe failed, assuming no GPU: %s", exc)
	return 0


def resolve_worker_count(num_videos: int, requested: int | None) -> int:
	"""Pick worker process count.

	``requested`` None/0 means auto: one worker per GPU (min 1). Any explicit
	value is honored but capped by video count. Never exceeds ``num_videos`` —
	extra workers would sit idle. Single GPU / CPU auto-resolves to 1 (sequential).
	"""
	if num_videos <= 1:
		return 1
	if requested and requested > 0:
		return max(1, min(int(requested), num_videos))
	gpus = available_gpu_count()
	auto = gpus if gpus > 0 else 1
	return max(1, min(auto, num_videos))


def _raw_fraction(overall: float | None) -> float:
	"""Convert a worker's banded 0.05–0.75 fraction back to a 0–1 per-video ratio."""
	if overall is None or _INFER_SPAN <= 0:
		return 0.0
	ratio = (overall - _INFER_START) / _INFER_SPAN
	return min(max(ratio, 0.0), 1.0)


def _worker_main(
	task_queue,
	progress_queue,
	device_index: int,
	analyze_kwargs: dict,
) -> None:
	"""Spawned process: pin to one GPU and analyze videos pulled from the queue."""
	# Import inside the process so CUDA initializes on the child, not the parent.
	from skellyclicker.core.deeplabcut_handler.analyze_videos_dlc import (
		analyze_videos_dlc,
	)

	device = f"cuda:{device_index}" if device_index >= 0 else None

	while True:
		task = task_queue.get()
		if task is None:  # sentinel — no more videos.
			break
		video_index, video_path = task
		name = Path(video_path).name

		def on_progress(overall: float | None, _message: str) -> None:
			progress_queue.put(
				("progress", video_index, name, _raw_fraction(overall))
			)

		try:
			analyze_videos_dlc(
				videos=[video_path],
				multiprocess=False,
				device=device,
				progress_callback=on_progress,
				**analyze_kwargs,
			)
			progress_queue.put(("done", video_index, name, 1.0))
		except Exception as exc:  # fail-fast: report and stop this worker.
			progress_queue.put(("error", video_index, name, f"{type(exc).__name__}: {exc}"))
			break


def analyze_videos_parallel(
	*,
	video_paths: list[str],
	analyze_kwargs: dict,
	worker_count: int,
	progress_callback: ProgressCallback | None = None,
) -> None:
	"""Analyze videos concurrently, one worker process per GPU.

	``analyze_kwargs`` are forwarded verbatim to ``analyze_videos_dlc`` (config,
	destfolder, batch_size, save_as_csv, videotype, overwrite, …) minus the
	per-call fields this supervisor owns (``videos``, ``device``, ``multiprocess``,
	``progress_callback``). Raises RuntimeError if any video fails.
	"""
	import torch.multiprocessing as tmp

	ctx = tmp.get_context("spawn")
	task_queue = ctx.Queue()
	progress_queue = ctx.Queue()

	for index, path in enumerate(video_paths):
		task_queue.put((index, path))
	for _ in range(worker_count):
		task_queue.put(None)  # one sentinel per worker.

	gpus = available_gpu_count()
	procs = []
	for slot in range(worker_count):
		# Round-robin GPUs; -1 (CPU) when no CUDA so device stays None.
		device_index = (slot % gpus) if gpus > 0 else -1
		proc = ctx.Process(
			target=_worker_main,
			args=(task_queue, progress_queue, device_index, analyze_kwargs),
			daemon=True,
		)
		proc.start()
		procs.append(proc)

	_drain_progress(
		progress_queue=progress_queue,
		procs=procs,
		total=len(video_paths),
		progress_callback=progress_callback,
	)


def _drain_progress(
	*,
	progress_queue,
	procs: list,
	total: int,
	progress_callback: ProgressCallback | None,
) -> None:
	"""Aggregate per-video fractions into one combined inference progress bar.

	Deliberately shows a single combined figure (overall % + videos-complete
	count), not per-video lines: with N GPUs all N videos run at once, so a
	per-video breakdown is noisy and non-intuitive.
	"""
	fractions: dict[int, float] = {}
	completed = 0
	error: str | None = None

	def report() -> None:
		mean = (sum(fractions.values()) / total) if total else 0.0
		overall = _INFER_START + _INFER_SPAN * min(max(mean, 0.0), 1.0)
		if progress_callback:
			progress_callback(
				overall,
				f"Analyzing {total} videos in parallel · {completed}/{total} complete",
			)

	while completed < total and error is None:
		kind, video_index, _name, payload = progress_queue.get()
		if kind == "progress":
			fractions[video_index] = float(payload)
			report()
		elif kind == "done":
			fractions[video_index] = 1.0
			completed += 1
			report()
		elif kind == "error":
			error = str(payload)

	if error is not None:
		for proc in procs:  # stop stragglers on fail-fast.
			if proc.is_alive():
				proc.terminate()
		raise RuntimeError(f"Parallel analyze failed: {error}")

	for proc in procs:
		proc.join()
