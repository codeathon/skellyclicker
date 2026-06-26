"""Helpers for human label CSV paths and default filenames."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _sanitize_filename_part(value: str) -> str:
	"""Keep basename segments filesystem-safe."""
	cleaned = re.sub(r"[^\w\-.]+", "_", value.strip())
	return cleaned or "video"


def human_labels_csv_filename(
	video_names: list[str],
	*,
	now: datetime | None = None,
) -> str:
	"""Default human-label CSV name: timestamp, video stem(s), skellyclicker_labels."""
	when = now or datetime.now()
	timestamp = when.strftime("%Y-%m-%d_%H-%M-%S")
	stems = [_sanitize_filename_part(Path(name).stem) for name in video_names if name]
	if not stems:
		video_part = "video"
	elif len(stems) == 1:
		video_part = stems[0]
	else:
		video_part = "_".join(stems)
	return f"{timestamp}_{video_part}_skellyclicker_labels.csv"
