"""Extract labeled frame indices from a SkellyClicker human-labels CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from skellyclicker.core.session_validation import bodypart_names_from_csv_columns


def human_label_frames_per_video(csv_path: str | Path) -> dict[str, list[int]]:
	"""Return sorted frame indices per video basename for rows with any coordinate."""
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
