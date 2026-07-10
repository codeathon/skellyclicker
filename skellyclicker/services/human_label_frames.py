"""Extract labeled frame indices from human labels (DLC labeled-data or legacy CSV)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from skellyclicker.core.deeplabcut_handler.labeled_data_io import (
	frames_per_video_from_labeled_data,
	is_legacy_skellyclicker_csv,
	resolve_human_labels_root,
)
from skellyclicker.core.session_validation import bodypart_names_from_csv_columns


def human_label_frames_per_video(
	path: str | Path,
	video_paths: list[str] | None = None,
) -> dict[str, list[int]]:
	"""Return sorted frame indices per video basename for rows with any coordinate.

	Accepts a DLC ``labeled-data`` directory, a ``CollectedData_*.csv``, or a
	legacy flat skellyclicker human-labels CSV. Pass ``video_paths`` when reading
	labeled-data so keys match session video basenames.
	"""
	p = Path(path).expanduser()
	if p.is_file() and is_legacy_skellyclicker_csv(p):
		return _frames_from_legacy_csv(p)
	root = resolve_human_labels_root(p)
	return frames_per_video_from_labeled_data(root, video_paths=video_paths)


def _frames_from_legacy_csv(csv_path: Path) -> dict[str, list[int]]:
	df = pd.read_csv(csv_path)
	if "video" not in df.columns or "frame" not in df.columns:
		raise ValueError("Human labels CSV must have 'video' and 'frame' columns")

	bodyparts = bodypart_names_from_csv_columns(list(df.columns))
	if not bodyparts:
		raise ValueError(f"No bodypart columns found in human labels CSV: {csv_path}")

	coord_cols = [f"{bp}_{axis}" for bp in bodyparts for axis in ("x", "y")]
	coord_cols = [c for c in coord_cols if c in df.columns]
	if not coord_cols:
		raise ValueError(f"No coordinate columns found in human labels CSV: {csv_path}")

	labeled = df[~df[coord_cols].isna().all(axis=1)]
	if labeled.empty:
		raise ValueError("Human labels CSV has no labeled frames")

	result: dict[str, list[int]] = {}
	for video_name, group in labeled.groupby("video"):
		frames = sorted({int(f) for f in group["frame"].unique()})
		result[str(video_name)] = frames
	return result
