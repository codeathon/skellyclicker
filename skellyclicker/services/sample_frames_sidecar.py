"""Sidecar file recording which frames a partial analyze sampled for review.

The machine-labels CSV can be dense (seeded from a prior full analysis), so
"has a machine label" is not a reliable signal for which frames were the
performance sample. This sidecar records the sampled frames explicitly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".sample_frames.json"


def sample_frames_sidecar_path(machine_labels_csv: str | Path) -> Path:
	"""Deterministic sidecar path next to the machine-labels CSV."""
	csv = Path(machine_labels_csv)
	return csv.with_name(csv.stem + SIDECAR_SUFFIX)


def write_sample_frames(
	machine_labels_csv: str | Path,
	frames: Iterable[int],
) -> Path:
	"""Persist the performance-sample frame indices beside the machine CSV."""
	path = sample_frames_sidecar_path(machine_labels_csv)
	payload = {"sample_frames": sorted({int(f) for f in frames})}
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload))
	return path


def read_sample_frames(machine_labels_csv: str | Path) -> list[int] | None:
	"""Read sampled frame indices, or None when no sidecar exists / is invalid."""
	path = sample_frames_sidecar_path(machine_labels_csv)
	if not path.is_file():
		return None
	try:
		data = json.loads(path.read_text())
	except (json.JSONDecodeError, OSError):
		logger.warning("Could not read sample-frames sidecar: %s", path)
		return None
	frames = data.get("sample_frames")
	if not isinstance(frames, list):
		return None
	return sorted({int(f) for f in frames})
