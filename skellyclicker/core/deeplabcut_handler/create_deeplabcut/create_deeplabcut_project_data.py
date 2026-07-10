import logging
from pathlib import Path

import pandas as pd

from skellyclicker.core.deeplabcut_handler.labeled_data_io import (
	HUMAN_EXPERIMENTER_NAME,
	labeled_data_dir,
	write_labeled_data_from_wide,
)
from skellyclicker.core.session_validation import bodypart_names_from_csv_columns

logger = logging.getLogger(__name__)


def build_dlc_formatted_header(labels_dataframe: pd.DataFrame, scorer_name: str):
	"""Creates a dataframe with MultiIndex columns in DLC format (legacy helper)."""
	joint_names_dimension = labels_dataframe.columns.drop(["frame", "video"])
	joint_names = sorted(set(col.rsplit("_", 1)[0] for col in joint_names_dimension))

	column_tuples = []
	for joint in joint_names:
		column_tuples.append((scorer_name, joint, "x"))
		column_tuples.append((scorer_name, joint, "y"))

	multi_columns = pd.MultiIndex.from_tuples(
		column_tuples, names=["scorer", "bodyparts", "coords"]
	)
	header_df = pd.DataFrame(columns=multi_columns)
	return header_df, joint_names


def get_session_name(path_to_videos_for_training: str) -> str:
	"""Prefix for DLC labeled-data folders — session_* if present, else video folder name."""
	path_parts = Path(path_to_videos_for_training).parts
	for part in path_parts:
		if part.startswith("session") or part.startswith("Session"):
			return part

	folder_name = Path(path_to_videos_for_training).name
	if folder_name:
		logger.info(
			"No session_* segment in %s; using folder name %r for labeled-data prefix",
			path_to_videos_for_training,
			folder_name,
		)
		return folder_name

	raise ValueError(
		f"Could not derive a dataset name from path: {path_to_videos_for_training}"
	)


def fill_in_labelled_data_folder(
	path_to_videos_for_training: str,
	path_to_dlc_project_folder: str,
	path_to_image_labels_csv: str,
	scorer_name: str = HUMAN_EXPERIMENTER_NAME,
	video_paths: list[str] | None = None,
):
	"""Convert a legacy flat human CSV into DLC labeled-data folders.

	Prefer saving directly to labeled-data from the labeler. This path remains
	for Import of old *_skellyclicker_labels.csv files and CLI pipelines.
	"""
	labels_dataframe = pd.read_csv(path_to_image_labels_csv)
	joint_names = bodypart_names_from_csv_columns(list(labels_dataframe.columns))
	if not joint_names:
		raise ValueError(f"No bodyparts in labels CSV: {path_to_image_labels_csv}")

	if video_paths is None:
		# Legacy: all videos live under one folder named in the CSV.
		video_paths = [
			str(Path(path_to_videos_for_training) / str(name))
			for name in labels_dataframe["video"].astype(str).unique()
		]

	root = labeled_data_dir(path_to_dlc_project_folder)
	write_labeled_data_from_wide(
		labeled_data_root=root,
		wide_df=labels_dataframe,
		video_paths=video_paths,
		joint_names=joint_names,
		scorer_name=scorer_name,
	)

	# Summary logging for CLI users.
	from skellyclicker.core.deeplabcut_handler.labeled_data_io import (
		frames_per_video_from_labeled_data,
	)

	try:
		frames = frames_per_video_from_labeled_data(root, video_paths=video_paths)
		logger.info("\n=== Summary of Labeled Frames ===")
		for video, frame_list in frames.items():
			logger.info("%s: %s", video, frame_list)
	except ValueError:
		logger.info("No labeled frames written")


if __name__ == "__main__":
	path_to_videos_for_training = ""
	path_to_dlc_project_folder = ""
	path_to_image_labels_csv = ""

	fill_in_labelled_data_folder(
		path_to_videos_for_training=path_to_videos_for_training,
		path_to_dlc_project_folder=path_to_dlc_project_folder,
		path_to_image_labels_csv=path_to_image_labels_csv,
	)
