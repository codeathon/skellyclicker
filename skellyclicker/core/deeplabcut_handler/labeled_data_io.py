"""DLC labeled-data as the on-disk human-label source of truth.

SkellyClicker keeps a flat in-memory grid (video/frame/{bp}_x|y). This module
converts at the IO boundary to/from DeepLabCut CollectedData folders so we do
not maintain a separate *_skellyclicker_labels.csv.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import cv2
import pandas as pd

from skellyclicker.services.video_path_registry import (
	labeled_data_prefix,
	resolve_video_path,
)

logger = logging.getLogger(__name__)

# Match create_deeplabcut_config.HUMAN_EXPERIMENTER_NAME without importing DLC.
HUMAN_EXPERIMENTER_NAME = "human"

_IMG_FRAME_RE = re.compile(r"^img(\d+)\.png$", re.IGNORECASE)
_COLLECTED_GLOB = "CollectedData_*.csv"


def labeled_data_dir(project_path: str | Path) -> Path:
	"""Return ``{project}/labeled-data``."""
	return Path(project_path).expanduser().resolve() / "labeled-data"


def resolve_human_labels_root(path: str | Path) -> Path:
	"""Normalize a session human-labels path to the labeled-data directory.

	Accepts the labeled-data dir itself, a CollectedData_*.csv inside a video
	folder, a video folder under labeled-data, or a DLC project root.
	"""
	p = Path(path).expanduser().resolve()
	# Allow the canonical folder even before the first save creates it.
	if p.name == "labeled-data":
		return p
	if p.is_dir():
		# Video folder under labeled-data: .../labeled-data/{combined_name}
		if p.parent.name == "labeled-data":
			return p.parent
		# Project root that already has (or will have) labeled-data/
		if (p / "config.yaml").is_file() or (p / "labeled-data").exists():
			return p / "labeled-data"
		return p
	if p.is_file() and p.name.startswith("CollectedData_"):
		# .../labeled-data/{combined}/CollectedData_human.csv
		if p.parent.parent.name == "labeled-data":
			return p.parent.parent
		return p.parent
	raise ValueError(f"Not a DLC labeled-data path: {path}")


def video_labeled_folder(
	project_or_labeled_data: str | Path,
	video_path: str | Path,
) -> Path:
	"""Per-video folder: ``labeled-data/{prefix}_{stem}``."""
	root = Path(project_or_labeled_data).expanduser().resolve()
	if root.name != "labeled-data":
		root = labeled_data_dir(root)
	video = Path(video_path)
	combined = f"{labeled_data_prefix(str(video))}_{video.stem}"
	return root / combined


def collected_data_csv_path(
	folder: Path,
	scorer_name: str = HUMAN_EXPERIMENTER_NAME,
) -> Path:
	return folder / f"CollectedData_{scorer_name}.csv"


def has_human_labels(labeled_data: str | Path) -> bool:
	"""True when any CollectedData_*.csv exists under labeled-data."""
	root = Path(labeled_data).expanduser().resolve()
	if root.name != "labeled-data" and (root / "labeled-data").is_dir():
		root = root / "labeled-data"
	if not root.is_dir():
		return False
	return any(root.glob(f"*/{_COLLECTED_GLOB}"))


def _frame_from_image_index(index_value: object) -> int | None:
	name = Path(str(index_value)).name
	match = _IMG_FRAME_RE.match(name)
	if not match:
		return None
	return int(match.group(1))


def _build_dlc_header(joint_names: list[str], scorer_name: str) -> pd.DataFrame:
	column_tuples = []
	for joint in joint_names:
		column_tuples.append((scorer_name, joint, "x"))
		column_tuples.append((scorer_name, joint, "y"))
	multi_columns = pd.MultiIndex.from_tuples(
		column_tuples, names=["scorer", "bodyparts", "coords"]
	)
	return pd.DataFrame(columns=multi_columns)


def read_collected_data_csv(csv_path: str | Path) -> pd.DataFrame:
	"""Load a DLC CollectedData CSV with MultiIndex columns."""
	path = Path(csv_path)
	df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
	# Normalize level names when present.
	if isinstance(df.columns, pd.MultiIndex):
		df.columns = df.columns.set_names(["scorer", "bodyparts", "coords"])
	return df


def bodyparts_from_collected_df(df: pd.DataFrame) -> list[str]:
	"""Unique bodypart names from MultiIndex columns, preserving order."""
	if not isinstance(df.columns, pd.MultiIndex):
		return []
	# Level 1 is bodyparts when names are scorer/bodyparts/coords.
	level = 1 if df.columns.nlevels >= 2 else 0
	seen: set[str] = set()
	names: list[str] = []
	for bp in df.columns.get_level_values(level):
		name = str(bp)
		if name not in seen:
			seen.add(name)
			names.append(name)
	return names


def bodyparts_from_labeled_data(labeled_data: str | Path) -> list[str]:
	"""Bodyparts from the first CollectedData CSV under labeled-data."""
	root = resolve_human_labels_root(labeled_data)
	for csv_path in sorted(root.glob(f"*/{_COLLECTED_GLOB}")):
		df = read_collected_data_csv(csv_path)
		names = bodyparts_from_collected_df(df)
		if names:
			return names
	return []


def collected_df_to_wide_rows(
	df: pd.DataFrame,
	video_basename: str,
	scorer_name: str = HUMAN_EXPERIMENTER_NAME,
) -> pd.DataFrame:
	"""Convert one CollectedData table to flat video/frame/{bp}_x|y rows."""
	bodyparts = bodyparts_from_collected_df(df)
	rows: list[dict[str, object]] = []
	for index_value, row in df.iterrows():
		frame = _frame_from_image_index(index_value)
		if frame is None:
			logger.warning("Skipping CollectedData row with bad image index: %s", index_value)
			continue
		out: dict[str, object] = {"video": video_basename, "frame": frame}
		for bp in bodyparts:
			try:
				out[f"{bp}_x"] = row[(scorer_name, bp, "x")]
				out[f"{bp}_y"] = row[(scorer_name, bp, "y")]
			except KeyError:
				# Scorer name in file may differ; fall back to first scorer level.
				scorers = df.columns.get_level_values(0)
				scorer = str(scorers[0]) if len(scorers) else scorer_name
				out[f"{bp}_x"] = row[(scorer, bp, "x")]
				out[f"{bp}_y"] = row[(scorer, bp, "y")]
		rows.append(out)
	if not rows:
		cols = ["video", "frame"] + [f"{bp}_{a}" for bp in bodyparts for a in ("x", "y")]
		return pd.DataFrame(columns=cols)
	return pd.DataFrame(rows)


def wide_df_from_labeled_data(
	labeled_data: str | Path,
	video_paths: list[str],
	scorer_name: str = HUMAN_EXPERIMENTER_NAME,
) -> pd.DataFrame:
	"""Load CollectedData for session videos into a flat wide DataFrame."""
	root = resolve_human_labels_root(labeled_data)
	frames: list[pd.DataFrame] = []
	for raw in video_paths:
		video_path = Path(raw).expanduser().resolve()
		folder = video_labeled_folder(root, video_path)
		csv_path = collected_data_csv_path(folder, scorer_name)
		if not csv_path.is_file():
			# Try any CollectedData_*.csv in that folder (alternate scorer).
			matches = sorted(folder.glob(_COLLECTED_GLOB)) if folder.is_dir() else []
			if not matches:
				continue
			csv_path = matches[0]
		df = read_collected_data_csv(csv_path)
		wide = collected_df_to_wide_rows(df, video_path.name, scorer_name=scorer_name)
		if not wide.empty:
			frames.append(wide)
	if not frames:
		return pd.DataFrame(columns=["video", "frame"])
	return pd.concat(frames, ignore_index=True)


def write_video_labeled_data(
	*,
	labeled_data_root: str | Path,
	video_path: str | Path,
	video_rows: pd.DataFrame,
	joint_names: list[str],
	scorer_name: str = HUMAN_EXPERIMENTER_NAME,
) -> Path:
	"""Rewrite one video's labeled-data folder from flat rows (siblings untouched)."""
	root = Path(labeled_data_root).expanduser().resolve()
	if root.name != "labeled-data":
		root = labeled_data_dir(root)
	root.mkdir(parents=True, exist_ok=True)

	video_path = Path(video_path).expanduser().resolve()
	folder = video_labeled_folder(root, video_path)
	# Full rewrite of this video only — stale PNGs/labels must not linger.
	if folder.exists():
		shutil.rmtree(folder)
	folder.mkdir(parents=True)

	header_df = _build_dlc_header(joint_names, scorer_name)
	df = header_df.copy()
	combined_name = folder.name

	if video_rows.empty:
		# Empty labels: still write empty CollectedData so the folder is valid.
		csv_path = collected_data_csv_path(folder, scorer_name)
		df.to_csv(csv_path)
		h5_path = folder / f"CollectedData_{scorer_name}.h5"
		try:
			df.to_hdf(str(h5_path), key="df_with_missing", format="table", mode="w")
		except ImportError:
			logger.warning("pytables not installed; skipped writing %s", h5_path)
		return csv_path

	if not video_path.is_file():
		raise FileNotFoundError(f"Video file not found: {video_path}")

	cap = cv2.VideoCapture(str(video_path))
	try:
		coord_cols = [c for c in video_rows.columns if c not in ("video", "frame")]
		labeled = video_rows
		if coord_cols:
			labeled = video_rows[~video_rows[coord_cols].isna().all(axis=1)]

		for _, row in labeled.iterrows():
			frame_number = int(row["frame"])
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
			ret, frame = cap.read()
			if not ret:
				logger.warning(
					"Could not read frame %s from %s, skipping",
					frame_number,
					video_path,
				)
				continue

			image_name = f"img{frame_number:05d}.png"
			cv2.imwrite(str(folder / image_name), frame)
			image_path = f"labeled-data/{combined_name}/{image_name}"

			frame_data = {}
			for joint in joint_names:
				frame_data[(scorer_name, joint, "x")] = row.get(f"{joint}_x")
				frame_data[(scorer_name, joint, "y")] = row.get(f"{joint}_y")
			df.loc[image_path] = frame_data
	finally:
		cap.release()

	csv_path = collected_data_csv_path(folder, scorer_name)
	df.to_csv(csv_path)
	# H5 is preferred by DLC but optional when pytables is unavailable.
	h5_path = folder / f"CollectedData_{scorer_name}.h5"
	try:
		df.to_hdf(str(h5_path), key="df_with_missing", format="table", mode="w")
	except ImportError:
		logger.warning("pytables not installed; skipped writing %s", h5_path)
	logger.info("Saved DLC human labels to %s", csv_path)
	return csv_path


def write_labeled_data_from_wide(
	*,
	labeled_data_root: str | Path,
	wide_df: pd.DataFrame,
	video_paths: list[str],
	joint_names: list[str] | None = None,
	scorer_name: str = HUMAN_EXPERIMENTER_NAME,
	only_videos: list[str] | None = None,
) -> Path:
	"""Write flat wide labels into per-video labeled-data folders.

	When ``only_videos`` is set (basenames), only those folders are rewritten —
	used for corpus saves so other experiments' labels stay intact.
	"""
	root = Path(labeled_data_root).expanduser().resolve()
	if root.name != "labeled-data":
		root = labeled_data_dir(root)
	root.mkdir(parents=True, exist_ok=True)

	if wide_df.empty and not only_videos:
		return root

	df = wide_df.copy()
	if "video" not in df.columns or "frame" not in df.columns:
		raise ValueError("Wide labels must have 'video' and 'frame' columns")

	df["video"] = df["video"].astype(str)
	if joint_names is None:
		from skellyclicker.core.session_validation import bodypart_names_from_csv_columns

		joint_names = bodypart_names_from_csv_columns(list(df.columns))
	if not joint_names:
		raise ValueError("No bodypart columns found in wide labels")

	only_set = {Path(v).name for v in only_videos} if only_videos else None
	grouped = dict(tuple(df.groupby("video"))) if not df.empty else {}

	# Ensure we rewrite folders for requested videos even if they have zero rows
	# (user cleared all labels on that video).
	targets: list[str]
	if only_set is not None:
		targets = sorted(only_set)
	else:
		targets = sorted(grouped.keys())

	for video_name in targets:
		video_abs = resolve_video_path(video_name, video_paths)
		rows = grouped.get(video_name, pd.DataFrame(columns=df.columns))
		write_video_labeled_data(
			labeled_data_root=root,
			video_path=video_abs,
			video_rows=rows,
			joint_names=joint_names,
			scorer_name=scorer_name,
		)
	return root


def frames_per_video_from_labeled_data(
	labeled_data: str | Path,
	video_paths: list[str] | None = None,
	scorer_name: str = HUMAN_EXPERIMENTER_NAME,
) -> dict[str, list[int]]:
	"""Sorted labeled frame indices per video basename from labeled-data."""
	root = resolve_human_labels_root(labeled_data)
	result: dict[str, list[int]] = {}

	if video_paths:
		wide = wide_df_from_labeled_data(root, video_paths, scorer_name=scorer_name)
		if wide.empty:
			raise ValueError("Human labels have no labeled frames")
		for video_name, group in wide.groupby("video"):
			coord_cols = [c for c in group.columns if c not in ("video", "frame")]
			labeled = group[~group[coord_cols].isna().all(axis=1)] if coord_cols else group
			frames = sorted({int(f) for f in labeled["frame"].unique()})
			if frames:
				result[str(video_name)] = frames
		if not result:
			raise ValueError("Human labels have no labeled frames")
		return result

	# No session video list — scan every CollectedData folder.
	for csv_path in sorted(root.glob(f"*/{_COLLECTED_GLOB}")):
		df = read_collected_data_csv(csv_path)
		# Infer video stem from folder name: {prefix}_{stem}
		folder_name = csv_path.parent.name
		video_key = folder_name
		frames: list[int] = []
		for index_value in df.index:
			frame = _frame_from_image_index(index_value)
			if frame is not None:
				frames.append(frame)
		if frames:
			result[video_key] = sorted(set(frames))
	if not result:
		raise ValueError("Human labels have no labeled frames")
	return result


def is_legacy_skellyclicker_csv(path: str | Path) -> bool:
	"""True for flat video/frame/{bp}_x CSV (not DLC CollectedData)."""
	p = Path(path)
	if not p.is_file() or p.suffix.lower() != ".csv":
		return False
	if p.name.startswith("CollectedData_"):
		return False
	# Peek columns without loading MultiIndex as data.
	head = pd.read_csv(p, nrows=0)
	cols = {str(c) for c in head.columns}
	return "video" in cols and "frame" in cols
