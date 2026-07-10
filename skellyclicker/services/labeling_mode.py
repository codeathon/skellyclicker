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

	Rules:
	- 0/1 videos → single
	- 2+ videos → corpus (one video at a time)

	Synced multi-cam grid is disabled for now: auto-detecting equal CAP_PROP
	counts was opening both videos and 500'ing for training-corpus sessions
	(different experiments / unequal lengths). Re-enable synced only behind an
	explicit multi-cam opt-in later.
	"""
	if len(video_paths) <= 1:
		return LabelingMode.single
	return LabelingMode.corpus
