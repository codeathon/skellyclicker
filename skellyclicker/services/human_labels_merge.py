"""Merge sparse human-label CSV rows so corpus single-video saves keep other videos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_human_label_rows(
	existing_csv: str | Path | None,
	new_rows: pd.DataFrame,
	*,
	active_video: str,
	output_path: str | Path,
) -> str:
	"""Write corpus human labels: replace rows for active_video, keep other videos.

	Why: corpus mode opens one video at a time; a naive save would drop labels for
	other experiments still in the session CSV.
	"""
	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	active = Path(active_video).name
	incoming = new_rows.copy()
	if "video" in incoming.columns:
		incoming["video"] = incoming["video"].astype(str)

	if existing_csv and Path(existing_csv).is_file():
		prior = pd.read_csv(existing_csv)
		if "video" in prior.columns:
			prior["video"] = prior["video"].astype(str)
			kept = prior[prior["video"] != active]
			merged = pd.concat([kept, incoming], ignore_index=True)
		else:
			merged = incoming
	else:
		merged = incoming

	# Stable order: video then frame for readable diffs / DLC fill.
	if "video" in merged.columns and "frame" in merged.columns:
		merged = merged.sort_values(["video", "frame"]).reset_index(drop=True)
	merged.to_csv(out, index=False)
	return str(out.resolve())
