"""Patch SkellyClicker machine-labels CSV rows without rewriting the full file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def patch_machine_labels_csv(
	existing_path: str | Path,
	patch_df: pd.DataFrame,
	output_path: str | Path | None = None,
) -> Path:
	"""Update or insert (video, frame) rows; preserve all other rows unchanged."""
	out = Path(output_path or existing_path)
	patch = patch_df.copy()
	if patch.index.names != ["video", "frame"]:
		if "video" in patch.columns and "frame" in patch.columns:
			patch = patch.set_index(["video", "frame"])
		else:
			raise ValueError("patch_df must be indexed by (video, frame)")

	if Path(existing_path).is_file():
		existing = pd.read_csv(existing_path)
		existing["video"] = existing["video"].astype(str)
		existing["frame"] = pd.to_numeric(existing["frame"], errors="coerce").astype("Int64")
		existing = existing.set_index(["video", "frame"])
		# Align columns — partial rows may introduce likelihood columns.
		for col in patch.columns:
			if col not in existing.columns:
				existing[col] = pd.NA
		for col in existing.columns:
			if col not in patch.columns:
				patch[col] = pd.NA
		existing.update(patch)
		# Rows in patch but not in existing (new sparse entries).
		missing_idx = patch.index.difference(existing.index)
		if len(missing_idx):
			existing = pd.concat([existing, patch.loc[missing_idx]])
	else:
		existing = patch

	result = existing.reset_index()
	result.to_csv(out, index=False)
	return out


def export_per_video_machine_csvs(
	machine_csv: str | Path,
	video_paths: list[str],
) -> list[Path]:
	"""Copy each video's rows from the combined machine CSV next to that video.

	Writes ``{video_dir}/{stem}.csv`` (e.g. ``eye1.avi`` → ``eye1.csv``).
	A later Full Analysis / iteration overwrites the same path; other CSVs in
	the folder are left alone.
	"""
	src = Path(machine_csv).expanduser().resolve()
	if not src.is_file():
		raise FileNotFoundError(f"Machine labels CSV not found: {src}")
	df = pd.read_csv(src)
	if "video" not in df.columns:
		raise ValueError(f"Machine labels CSV missing 'video' column: {src}")
	df = df.copy()
	df["video"] = df["video"].astype(str)

	written: list[Path] = []
	for raw in video_paths:
		video = Path(raw).expanduser().resolve()
		stem = video.stem
		name = video.name
		# Match basename or stem — merge may store .mp4 even when the file is .avi.
		mask = (df["video"] == name) | (df["video"].map(lambda v: Path(v).stem) == stem)
		rows = df.loc[mask]
		out = video.parent / f"{stem}.csv"
		# Same path each iteration — overwrite in place; do not delete other CSVs.
		rows.to_csv(out, index=False)
		written.append(out)
	return written
