"""Bridge DeepLabCut tqdm inference bars to SkellyClicker job progress callbacks."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from tqdm import tqdm as TqdmBase

ProgressCallback = Callable[[float, str], None]
TrainProgressCallback = Callable[[float | None, str], None]

# Inference phase occupies this slice of overall analyze job progress.
ANALYZE_INFERENCE_START = 0.05
ANALYZE_INFERENCE_END = 0.75

# Training epoch updates map into this slice (after dataset prep reports ~15%).
TRAIN_PROGRESS_START = 0.15
TRAIN_PROGRESS_END = 0.95

_TRAIN_EPOCH_RE = re.compile(
	r"Epoch (\d+)/(\d+).*?train loss ([\d.]+)",
)


def train_epoch_fraction(
	current_epoch: int,
	total_epochs: int,
	*,
	train_start: float = TRAIN_PROGRESS_START,
	train_end: float = TRAIN_PROGRESS_END,
) -> float:
	"""Map DLC epoch counter to overall train-job progress."""
	if total_epochs <= 0:
		return train_start
	ratio = min(max(current_epoch / total_epochs, 0.0), 1.0)
	return train_start + ratio * (train_end - train_start)


@contextmanager
def hook_dlc_training_progress(
	callback: TrainProgressCallback,
	*,
	train_start: float = TRAIN_PROGRESS_START,
	train_end: float = TRAIN_PROGRESS_END,
):
	"""Forward DLC PyTorch `Epoch X/Y` log lines to the job progress callback."""
	original_info = logging.info

	def patched_info(msg, *args, **kwargs):
		if args or kwargs:
			original_info(msg, *args, **kwargs)
			try:
				text = msg % args if args else str(msg)
			except (TypeError, ValueError):
				text = str(msg)
		else:
			original_info(msg)
			text = str(msg)
		match = _TRAIN_EPOCH_RE.search(text)
		if not match:
			return
		current = int(match.group(1))
		total = int(match.group(2))
		loss = match.group(3)
		fraction = train_epoch_fraction(
			current,
			total,
			train_start=train_start,
			train_end=train_end,
		)
		callback(fraction, f"Epoch {current}/{total} · train loss {loss}")

	logging.info = patched_info
	try:
		yield
	finally:
		logging.info = original_info


class ReportingTqdm(TqdmBase):
	"""tqdm that forwards frame progress to a callback (throttled)."""

	def __init__(
		self,
		*args: Any,
		progress_callback: ProgressCallback | None = None,
		progress_meta: dict[str, Any] | None = None,
		**kwargs: Any,
	) -> None:
		self._progress_callback = progress_callback
		self._progress_meta = progress_meta or {}
		self._last_reported = -1.0
		self._last_report_time = 0.0
		super().__init__(*args, **kwargs)

	def _overall_fraction(self) -> float | None:
		if self.total is None or self.total <= 0:
			return None
		meta = self._progress_meta
		video_index = int(meta["video_index"])
		video_count = max(int(meta["video_count"]), 1)
		num_passes = max(int(meta.get("num_passes", 1)), 1)
		bar_slot = int(meta.get("bar_slot", 0))
		frame_frac = min(max(self.n / self.total, 0.0), 1.0)

		video_span = (ANALYZE_INFERENCE_END - ANALYZE_INFERENCE_START) / video_count
		pass_span = video_span / num_passes
		return (
			ANALYZE_INFERENCE_START
			+ video_index * video_span
			+ bar_slot * pass_span
			+ frame_frac * pass_span
		)

	def _maybe_report(self) -> None:
		if not self._progress_callback:
			return
		overall = self._overall_fraction()
		if overall is None:
			return
		now = time.monotonic()
		if overall - self._last_reported < 0.005 and overall < 0.99:
			if now - self._last_report_time < 0.25:
				return
		self._last_reported = overall
		self._last_report_time = now
		meta = self._progress_meta
		name = meta.get("video_name", "video")
		vi = int(meta["video_index"]) + 1
		vc = int(meta["video_count"])
		self._progress_callback(
			overall,
			f"{name}: frame {self.n}/{self.total} (video {vi}/{vc})",
		)

	def update(self, n: float = 1) -> int | None:
		result = super().update(n)
		self._maybe_report()
		return result

	def close(self) -> None:
		self._maybe_report()
		super().close()


def _reporting_gpu_tqdm_factory(original_gpu: type | None):
	"""GpuTqdm subclass that keeps GPU postfix display and reports progress."""

	class ReportingGpuTqdm(ReportingTqdm):
		def __init__(self, *args: Any, **kwargs: Any) -> None:
			super().__init__(*args, **kwargs)
			try:
				import torch

				self._cuda_available = torch.cuda.is_available()
			except ImportError:
				self._cuda_available = False

		def __iter__(self) -> Iterator[Any]:
			for obj in TqdmBase.__iter__(self):
				if self._cuda_available:
					import torch

					used = torch.cuda.memory_reserved() / 1024**2
					total = torch.cuda.get_device_properties(0).total_memory / 1024**2
					self.set_postfix({"GPU": f"{used:.1f}/{total:.1f} MiB"})
				yield obj

	if original_gpu is not None:
		ReportingGpuTqdm.__name__ = "ReportingGpuTqdm"
	return ReportingGpuTqdm


@contextmanager
def hook_dlc_tqdm(
	callback: ProgressCallback,
	video_index: int,
	video_count: int,
	video_name: str,
	*,
	num_passes: int = 1,
):
	"""Patch DLC video_inference tqdm for one video analyze pass."""
	import deeplabcut.pose_estimation_pytorch.apis.videos as dlc_videos

	original_tqdm = dlc_videos.tqdm
	original_gpu = getattr(dlc_videos, "GpuTqdm", None)
	bar_slot = {"n": 0}

	def _meta() -> dict[str, Any]:
		return {
			"video_index": video_index,
			"video_count": video_count,
			"video_name": video_name,
			"num_passes": num_passes,
			"bar_slot": bar_slot["n"],
		}

	def tqdm_factory(*args: Any, **kwargs: Any) -> ReportingTqdm:
		kwargs["progress_callback"] = callback
		kwargs["progress_meta"] = _meta()
		bar_slot["n"] += 1
		return ReportingTqdm(*args, **kwargs)

	ReportingGpuTqdm = _reporting_gpu_tqdm_factory(original_gpu)
	dlc_videos.tqdm = tqdm_factory
	if original_gpu is not None:
		dlc_videos.GpuTqdm = ReportingGpuTqdm
	try:
		yield
	finally:
		dlc_videos.tqdm = original_tqdm
		if original_gpu is not None:
			dlc_videos.GpuTqdm = original_gpu
