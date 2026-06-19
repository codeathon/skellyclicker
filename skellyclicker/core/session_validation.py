"""Validate label CSVs against videos and bodypart sets."""

from pathlib import Path


def bodypart_names_from_csv_columns(columns: list[str]) -> list[str]:
	"""Extract unique bodypart names from DLC/SkellyClicker CSV columns."""
	seen: set[str] = set()
	names: list[str] = []
	for col in columns:
		if col.endswith("_x"):
			name = col[:-2]
		elif col.endswith("_y"):
			name = col[:-2]
		elif col in ("video", "frame"):
			continue
		else:
			continue
		if name not in seen:
			seen.add(name)
			names.append(name)
	return names


def validate_label_csv_against_videos(
	csv_path: str, video_paths: list[str]
) -> list[str]:
	"""Return warning strings if CSV video names don't match selected files."""
	warnings: list[str] = []
	import pandas as pd
	df = pd.read_csv(csv_path)
	if "video" not in df.columns:
		return ["CSV has no 'video' column"]
	csv_videos = set(df["video"].astype(str).unique())
	disk_videos = {Path(p).name for p in video_paths}
	missing = csv_videos - disk_videos
	extra = disk_videos - csv_videos
	if missing:
		warnings.append(f"CSV references videos not in selection: {missing}")
	if extra:
		warnings.append(f"Selected videos missing from CSV: {extra}")
	return warnings


def validate_bodypart_overlap(
	human_path: str | None, machine_path: str | None
) -> list[str]:
	"""Warn if human and machine label bodyparts differ."""
	if not human_path or not machine_path:
		return []
	import pandas as pd
	human = set(bodypart_names_from_csv_columns(list(pd.read_csv(human_path).columns)))
	machine = set(bodypart_names_from_csv_columns(list(pd.read_csv(machine_path).columns)))
	if human != machine:
		return [f"Bodypart mismatch: human={human}, machine={machine}"]
	return []
