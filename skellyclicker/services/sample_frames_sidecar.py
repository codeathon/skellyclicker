"""Sidecar file recording which frames a partial analyze sampled for review.

The machine-labels CSV can be dense (seeded from a prior full analysis), so
"has a machine label" is not a reliable signal for which frames were the
performance sample. This sidecar records the sampled frames explicitly,
keyed by video basename so the left panel can show only the active video.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".sample_frames.json"


def sample_frames_sidecar_path(machine_labels_csv: str | Path) -> Path:
	"""Deterministic sidecar path next to the machine-labels CSV."""
	csv = Path(machine_labels_csv)
	return csv.with_name(csv.stem + SIDECAR_SUFFIX)


def write_sample_frames(
	machine_labels_csv: str | Path,
	frames: Iterable[int] | Mapping[str, Iterable[int]],
) -> Path:
	"""Persist performance-sample frame indices beside the machine CSV.

	Prefer a per-video mapping so corpus labeler nav can filter by active video.
	A flat iterable is still accepted for single-video / legacy callers.
	"""
	path = sample_frames_sidecar_path(machine_labels_csv)
	if isinstance(frames, Mapping):
		by_video = {
			str(name): sorted({int(f) for f in idxs})
			for name, idxs in frames.items()
		}
		# Flat union kept for older readers that only know sample_frames.
		flat = sorted({f for idxs in by_video.values() for f in idxs})
		payload = {"sample_frames_by_video": by_video, "sample_frames": flat}
	else:
		flat = sorted({int(f) for f in frames})
		payload = {"sample_frames": flat}
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload))
	return path


def read_sample_frames(machine_labels_csv: str | Path) -> list[int] | None:
	"""Read flat sampled frame indices (union), or None when missing / invalid."""
	by_video = read_sample_frames_by_video(machine_labels_csv)
	if by_video is None:
		return None
	return sorted({f for idxs in by_video.values() for f in idxs})


def read_sample_frames_by_video(
	machine_labels_csv: str | Path,
) -> dict[str, list[int]] | None:
	"""Read per-video sample frames, or None when no sidecar exists / is invalid.

	Legacy sidecars with only a flat ``sample_frames`` list are returned under
	the empty-string key so callers can fall back to the union.
	"""
	path = sample_frames_sidecar_path(machine_labels_csv)
	if not path.is_file():
		return None
	try:
		data = json.loads(path.read_text())
	except (json.JSONDecodeError, OSError):
		logger.warning("Could not read sample-frames sidecar: %s", path)
		return None

	by_video_raw = data.get("sample_frames_by_video")
	if isinstance(by_video_raw, dict):
		result: dict[str, list[int]] = {}
		for name, frames in by_video_raw.items():
			if isinstance(frames, list):
				result[str(name)] = sorted({int(f) for f in frames})
		return result

	frames = data.get("sample_frames")
	if not isinstance(frames, list):
		return None
	# Legacy flat list — empty key means "apply to any active video".
	return {"": sorted({int(f) for f in frames})}


def sample_frames_for_video(
	by_video: dict[str, list[int]] | None,
	video_name: str | None,
) -> list[int] | None:
	"""Pick sample frames for one video basename; legacy flat list applies to all."""
	if by_video is None:
		return None
	if video_name and video_name in by_video:
		return by_video[video_name]
	# Legacy sidecar: only the empty-key union exists.
	if "" in by_video and len(by_video) == 1:
		return by_video[""]
	if video_name:
		return []
	# No active video (synced grid): show union of all videos' samples.
	return sorted({f for idxs in by_video.values() for f in idxs})
