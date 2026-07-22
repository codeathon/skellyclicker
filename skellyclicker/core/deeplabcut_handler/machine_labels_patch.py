"""Patch SkellyClicker machine-labels CSV rows without rewriting the full file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Chunk so export never holds a multi-video × hundreds-of-k-frames table in RAM.
_DEFAULT_EXPORT_CHUNKSIZE = 50_000


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


def _video_stem(value: str) -> str:
	return Path(str(value)).stem


def export_per_video_machine_csvs(
	machine_csv: str | Path,
	video_paths: list[str],
	*,
	chunksize: int = _DEFAULT_EXPORT_CHUNKSIZE,
) -> list[Path]:
	"""Copy each video's rows from the combined machine CSV next to that video.

	Writes ``{video_dir}/{stem}.csv`` (e.g. ``eye1.avi`` → ``eye1.csv``).
	A later Full Analysis / iteration overwrites the same path; other CSVs in
	the folder are left alone.

	Reads the combined file in chunks so multiple large videos do not need to
	fit in memory at once (fallback when sidecars were not written during merge).
	"""
	src = Path(machine_csv).expanduser().resolve()
	if not src.is_file():
		raise FileNotFoundError(f"Machine labels CSV not found: {src}")

	# stem → (output path, whether header already written)
	targets: dict[str, tuple[Path, bool]] = {}
	for raw in video_paths:
		video = Path(raw).expanduser().resolve()
		targets[video.stem] = (video.parent / f"{video.stem}.csv", False)

	if not targets:
		return []

	# Truncate outputs up front so a failed mid-export cannot leave a stale mix.
	for out_path, _ in targets.values():
		out_path.write_text("")

	header_done: dict[str, bool] = {stem: False for stem in targets}
	reader = pd.read_csv(src, chunksize=max(int(chunksize), 1))
	saw_video_col = False

	for chunk in reader:
		if "video" not in chunk.columns:
			raise ValueError(f"Machine labels CSV missing 'video' column: {src}")
		saw_video_col = True
		chunk = chunk.copy()
		chunk["video"] = chunk["video"].astype(str)
		chunk["_stem"] = chunk["video"].map(_video_stem)

		for stem, group in chunk.groupby("_stem", sort=False):
			if stem not in targets:
				continue
			out_path, _ = targets[stem]
			rows = group.drop(columns=["_stem"])
			rows.to_csv(
				out_path,
				mode="a",
				header=not header_done[stem],
				index=False,
			)
			header_done[stem] = True

	if not saw_video_col:
		# Empty file / no chunks — still surface the contract.
		peek = pd.read_csv(src, nrows=0)
		if "video" not in peek.columns:
			raise ValueError(f"Machine labels CSV missing 'video' column: {src}")

	# Videos with no rows keep an empty file (header-only if we never wrote).
	# Re-write header-only from the combined schema when possible.
	empty_stems = [stem for stem, done in header_done.items() if not done]
	if empty_stems:
		schema = pd.read_csv(src, nrows=0)
		for stem in empty_stems:
			out_path, _ = targets[stem]
			schema.to_csv(out_path, index=False)

	return [targets[stem][0] for stem in targets]
