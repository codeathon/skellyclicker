"""Parse DeepLabCut analysis CSV files into skellyclicker wide format."""

from pathlib import Path

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
		if coord not in ("x", "y"):
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
