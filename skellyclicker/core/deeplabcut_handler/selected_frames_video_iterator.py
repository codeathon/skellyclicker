"""VideoIterator that yields only selected frame indices (for partial DLC analyze)."""

from __future__ import annotations

import numpy as np
from deeplabcut.pose_estimation_pytorch.apis.videos import VideoIterator


class SelectedFramesVideoIterator(VideoIterator):
	"""Iterate a subset of frames by seeking — used instead of full-video inference."""

	def __init__(
		self,
		video_path: str,
		frame_indices: list[int],
		cropping: list[int] | None = None,
	) -> None:
		super().__init__(video_path, cropping=cropping)
		# Sorted unique — inference order must match frame_numbers passed to export.
		self._frame_indices = sorted({int(f) for f in frame_indices})
		self._pos = 0

	def get_n_frames(self, robust: bool = False) -> int:
		return len(self._frame_indices)

	def __next__(self) -> np.ndarray:
		if self._pos >= len(self._frame_indices):
			self._pos = 0
			self.reset()
			raise StopIteration

		frame_num = self._frame_indices[self._pos]
		self.set_to_frame(frame_num)
		frame = self.read_frame(crop=self._crop)
		if frame is None:
			raise StopIteration

		frame = frame.copy()
		self._pos += 1
		return frame
