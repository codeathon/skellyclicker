"""Parse DeepLabCut analysis CSV files into skellyclicker wide format."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Progress: (completed_count, total_count, video_label).
MergeProgressCallback = Callable[[int, int, str], None]


def _dlc_header_layout(raw: pd.DataFrame) -> tuple[int, int, int]:
	"""Return (bodyparts_row, coords_row, data_start_row) for a DLC CSV header."""
	if raw.shape[0] < 4:
		raise ValueError("DLC CSV is too short to contain header rows and data")
	first_col = str(raw.iloc[1, 0]).strip().lower()
	if first_col == "individuals":
		# Multi-animal: scorer, individuals, bodyparts, coords, then data.
		return 2, 3, 4
	return 1, 2, 3


def dlc_csv_video_name(csv_path: str | Path) -> str:
	"""Basename DeepLabCut embeds before ``DLC_`` in analyze CSV filenames."""
	video_name = Path(csv_path).name.split("DLC_")[0]
	if not video_name.endswith((".mp4", ".avi", ".mov", ".mkv")):
		# Older DLC outputs omit the extension; default keeps prior behavior.
		video_name = f"{video_name}.mp4"
	return video_name


def dlc_analysis_csv_to_skellyclicker(
	csv_path: str | Path,
	video_name: str,
) -> pd.DataFrame:
	"""Convert one DLC analyze output CSV (3- or 4-row header) to skellyclicker format."""
	# One video at a time — callers must not concat many large CSVs in memory.
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


def _stem_to_video_path(video_paths: list[str]) -> dict[str, Path]:
	"""Map video stem → resolved path (last wins on duplicate stems)."""
	mapping: dict[str, Path] = {}
	for raw in video_paths:
		path = Path(raw).expanduser().resolve()
		mapping[path.stem] = path
	return mapping


def merge_dlc_csvs_for_skellyclicker(
	csv_folder: str | Path,
	output_path: str | Path,
	*,
	filtered: bool = False,
	video_paths: list[str] | None = None,
	on_video_progress: MergeProgressCallback | None = None,
) -> list[Path]:
	"""Stream DLC per-video CSVs into one skellyclicker CSV; optionally write sidecars.

	Processes **one video at a time** so many large files (e.g. several × ~400k
	frames) never sit in memory together. When ``video_paths`` is set, also writes
	``{stem}.csv`` beside each matching source video during the same pass.
	"""
	folder = Path(csv_folder)
	out = Path(output_path)
	csv_paths = iter_dlc_video_csvs(folder, filtered=filtered)
	if not csv_paths:
		raise FileNotFoundError(
			f"No matching CSV files found in {folder}. Please check the path."
		)

	out.parent.mkdir(parents=True, exist_ok=True)
	stem_paths = _stem_to_video_path(video_paths or [])
	per_video_written: list[Path] = []
	columns: list[str] | None = None
	total = len(csv_paths)

	for index, csv in enumerate(csv_paths):
		video_name = dlc_csv_video_name(csv)
		stem = Path(video_name).stem
		if on_video_progress:
			on_video_progress(index, total, video_name)

		# Peak memory ≈ one DLC CSV; released before the next video.
		indexed = dlc_analysis_csv_to_skellyclicker(csv, video_name=video_name)
		flat = indexed.reset_index()
		del indexed

		if columns is None:
			columns = list(flat.columns)
			flat.to_csv(out, mode="w", header=True, index=False)
		else:
			# Same project bodyparts expected; pad/reorder so append stays valid.
			flat = flat.reindex(columns=columns)
			flat.to_csv(out, mode="a", header=False, index=False)

		dest_video = stem_paths.get(stem)
		if dest_video is not None:
			sidecar = dest_video.parent / f"{stem}.csv"
			# Overwrite same path each iteration; leave unrelated CSVs alone.
			flat.to_csv(sidecar, index=False)
			per_video_written.append(sidecar)

		del flat
		if on_video_progress:
			on_video_progress(index + 1, total, video_name)

	logger.info("Saved skellyclicker compatible CSV to %s", out)
	return per_video_written


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
