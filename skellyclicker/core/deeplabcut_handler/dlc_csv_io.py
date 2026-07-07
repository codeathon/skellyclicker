"""Parse DeepLabCut analysis CSV files into skellyclicker wide format."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def _dlc_header_layout(raw: pd.DataFrame) -> tuple[int, int, int]:
	"""Return (bodyparts_row, coords_row, data_start_row) for a DLC CSV header."""
	if raw.shape[0] < 4:
		raise ValueError("DLC CSV is too short to contain header rows and data")
	first_col = str(raw.iloc[1, 0]).strip().lower()
	if first_col == "individuals":
		# Multi-animal: scorer, individuals, bodyparts, coords, then data.
		return 2, 3, 4
	return 1, 2, 3


def dlc_analysis_csv_to_skellyclicker(
	csv_path: str | Path,
	video_name: str,
) -> pd.DataFrame:
	"""Convert one DLC analyze output CSV (3- or 4-row header) to skellyclicker format."""
	raw = pd.read_csv(csv_path, header=None)
	bp_row, coord_row, data_start = _dlc_header_layout(raw)
	data = raw.iloc[data_start:].copy()

	rows: dict[str, pd.Series] = {
		"frame": pd.to_numeric(data.iloc[:, 0], errors="coerce"),
	}
	for col in range(1, raw.shape[1]):
		bodypart = str(raw.iloc[bp_row, col]).strip()
		coord = str(raw.iloc[coord_row, col]).strip().lower()
		if coord not in ("x", "y", "likelihood"):
			continue
		rows[f"{bodypart}_{coord}"] = pd.to_numeric(data.iloc[:, col], errors="coerce")

	df = pd.DataFrame(rows)
	df = df.dropna(subset=["frame"])
	df["frame"] = df["frame"].astype(int)
	df["video"] = video_name
	return df.set_index(["video", "frame"])


def iter_dlc_video_csvs(csv_folder: Path, filtered: bool) -> list[Path]:
	"""Per-video DLC CSV paths in a folder, excluding skellyclicker merge outputs."""
	paths: list[Path] = []
	for path in sorted(csv_folder.glob("*.csv")):
		name = path.name
		if name.startswith("skellyclicker_"):
			continue
		if filtered:
			if name.endswith("_filtered.csv"):
				paths.append(path)
		elif name.endswith("_filtered.csv"):
			continue
		elif "DLC_" in name:
			paths.append(path)
	return paths


def dlc_predictions_to_skellyclicker(
	predictions: Sequence[dict],
	frame_numbers: Sequence[int],
	video_name: str,
	bodyparts: list[str],
) -> pd.DataFrame:
	"""Convert in-memory DLC predictions for selected frames to skellyclicker wide format."""
	if len(predictions) != len(frame_numbers):
		raise ValueError("predictions and frame_numbers must have the same length")

	rows: list[dict] = []
	for frame_num, pred in zip(frame_numbers, predictions):
		coords = pred["bodyparts"]
		# Single-animal: (1, n_bodyparts, 3) — x, y, likelihood.
		if coords.ndim == 3:
			coords = coords[0]
		row: dict = {"video": video_name, "frame": int(frame_num)}
		for i, bp in enumerate(bodyparts):
			row[f"{bp}_x"] = float(coords[i, 0])
			row[f"{bp}_y"] = float(coords[i, 1])
			if coords.shape[1] > 2:
				row[f"{bp}_likelihood"] = float(coords[i, 2])
		rows.append(row)

	df = pd.DataFrame(rows)
	return df.set_index(["video", "frame"])
