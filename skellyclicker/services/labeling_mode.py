"""Auto-detect synced multi-cam vs training-corpus labeling from video metadata."""

from __future__ import annotations

from pathlib import Path

import cv2

from skellyclicker.services.models import LabelingMode

# Re-export for callers that import from this module.
__all__ = ["LabelingMode", "detect_labeling_mode", "probe_video_frame_count"]


def probe_video_frame_count(path: str) -> int:
	"""Return frame count for a video file, or raise if it cannot be opened."""
	cap = cv2.VideoCapture(str(Path(path).expanduser()))
	if not cap.isOpened():
		raise ValueError(f"Could not open video: {path}")
	try:
		return max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
	finally:
		cap.release()


def detect_labeling_mode(video_paths: list[str]) -> LabelingMode:
	"""Pick labeler layout from the session video set.

	Rules (v1):
	- 0 videos → single (caller should not open labeler)
	- 1 video → single
	- 2+ equal frame counts → synced grid
	- 2+ unequal frame counts → corpus (one video at a time)
	"""
	if len(video_paths) <= 1:
		return LabelingMode.single
	counts = {probe_video_frame_count(p) for p in video_paths}
	if len(counts) == 1:
		return LabelingMode.synced
	return LabelingMode.corpus
