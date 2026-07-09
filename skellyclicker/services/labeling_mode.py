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


def _parent_dirs(video_paths: list[str]) -> set[str]:
	"""Resolved parent folders — different parents imply separate experiments."""
	parents: set[str] = set()
	for raw in video_paths:
		parents.add(str(Path(raw).expanduser().resolve().parent))
	return parents


def detect_labeling_mode(video_paths: list[str]) -> LabelingMode:
	"""Pick labeler layout from the session video set.

	Rules:
	- 0/1 videos → single
	- 2+ videos in different parent folders → corpus (training set, not multi-cam)
	- 2+ same folder, equal frame counts → synced grid
	- 2+ same folder, unequal frame counts → corpus
	"""
	if len(video_paths) <= 1:
		return LabelingMode.single
	# Different experiment folders must never open as a multi-cam grid — even when
	# CAP_PROP reports matching lengths (common false positive → Internal Server Error).
	if len(_parent_dirs(video_paths)) > 1:
		return LabelingMode.corpus
	counts = {probe_video_frame_count(p) for p in video_paths}
	# CAP_PROP of 0 is unreliable (common before decode) — never treat as synced.
	if 0 in counts or len(counts) > 1:
		return LabelingMode.corpus
	return LabelingMode.synced
