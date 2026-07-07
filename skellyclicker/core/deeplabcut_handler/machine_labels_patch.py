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
