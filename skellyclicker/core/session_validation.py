"""Validation helpers for labeling session consistency."""

from pathlib import Path

import pandas as pd


def bodypart_names_from_csv_columns(columns: list[str]) -> list[str]:
	"""Extract unique bodypart names from skellyclicker CSV column headers."""
	names: list[str] = []
	seen: set[str] = set()
	for column in columns:
		if column in ("video", "frame"):
			continue
		name = column.removesuffix("_x").removesuffix("_y")
		if name not in seen:
			seen.add(name)
			names.append(name)
	return names


def validate_label_csv_against_videos(
	csv_path: str | Path,
	video_files: list[str],
	label_kind: str,
) -> list[str]:
	"""Return human-readable warnings when CSV videos diverge from loaded files."""
	warnings: list[str] = []
	csv_path = Path(csv_path)
	if not csv_path.is_file():
		warnings.append(f"{label_kind} CSV not found: {csv_path}")
		return warnings

	loaded_names = {Path(path).name for path in video_files}
	df = pd.read_csv(csv_path, usecols=["video"])
	csv_names = set(df["video"].astype(str).unique().tolist())

	missing_in_csv = sorted(loaded_names - csv_names)
	extra_in_csv = sorted(csv_names - loaded_names)
	if missing_in_csv:
		warnings.append(
			f"{label_kind}: loaded videos missing from CSV: {', '.join(missing_in_csv)}"
		)
	if extra_in_csv:
		warnings.append(
			f"{label_kind}: CSV videos not currently loaded: {', '.join(extra_in_csv)}"
		)
	return warnings


def validate_bodypart_overlap(
	human_bodyparts: list[str],
	machine_bodyparts: list[str],
) -> list[str]:
	"""Warn when machine-label bodyparts do not overlap human label schema."""
	overlap = set(human_bodyparts) & set(machine_bodyparts)
	if not overlap:
		return [
			"Machine labels bodyparts do not overlap human labels; "
		 f"human={human_bodyparts}, machine={machine_bodyparts}"
		]
	return []
