"""Tests for DLC labeled-data as human-label source of truth."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from skellyclicker.core.deeplabcut_handler.labeled_data_io import (
	bodyparts_from_labeled_data,
	collected_df_to_wide_rows,
	frames_per_video_from_labeled_data,
	has_human_labels,
	labeled_data_dir,
	read_collected_data_csv,
	resolve_human_labels_root,
	video_labeled_folder,
	wide_df_from_labeled_data,
	write_labeled_data_from_wide,
	write_video_labeled_data,
)
from skellyclicker.services.human_label_frames import human_label_frames_per_video


def _tiny_video(path: Path, n_frames: int = 20) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	writer = cv2.VideoWriter(
		str(path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		10.0,
		(32, 32),
	)
	assert writer.isOpened(), f"Could not open VideoWriter for {path}"
	for i in range(n_frames):
		frame = np.full((32, 32, 3), i % 255, dtype=np.uint8)
		writer.write(frame)
	writer.release()
	return path


def test_resolve_human_labels_root_variants(tmp_path: Path):
	project = tmp_path / "proj"
	labeled = project / "labeled-data"
	labeled.mkdir(parents=True)
	(project / "config.yaml").write_text("Task: x\n")
	video_folder = labeled / "session_cam0"
	video_folder.mkdir()
	csv = video_folder / "CollectedData_human.csv"
	csv.write_text("placeholder")

	assert resolve_human_labels_root(labeled) == labeled.resolve()
	assert resolve_human_labels_root(project) == labeled.resolve()
	assert resolve_human_labels_root(video_folder) == labeled.resolve()
	assert resolve_human_labels_root(csv) == labeled.resolve()
	# Not created yet — still resolves by name.
	future = tmp_path / "other" / "labeled-data"
	assert resolve_human_labels_root(future) == future.resolve()


def test_wide_collected_round_trip(tmp_path: Path):
	session_dir = tmp_path / "session_demo"
	video = _tiny_video(session_dir / "cam0.mp4")
	project = tmp_path / "proj"
	root = labeled_data_dir(project)

	wide = pd.DataFrame(
		[
			{"video": "cam0.mp4", "frame": 2, "nose_x": 10.0, "nose_y": 20.0},
			{"video": "cam0.mp4", "frame": 5, "nose_x": 11.0, "nose_y": 21.0},
		]
	)
	write_labeled_data_from_wide(
		labeled_data_root=root,
		wide_df=wide,
		video_paths=[str(video)],
		joint_names=["nose"],
	)

	assert has_human_labels(root)
	folder = video_labeled_folder(root, video)
	assert (folder / "img00002.png").is_file()
	assert (folder / "img00005.png").is_file()
	assert (folder / "CollectedData_human.csv").is_file()

	loaded = wide_df_from_labeled_data(root, [str(video)])
	assert len(loaded) == 2
	assert set(loaded["frame"].tolist()) == {2, 5}
	assert loaded.loc[loaded["frame"] == 2, "nose_x"].iloc[0] == pytest.approx(10.0)
	assert bodyparts_from_labeled_data(root) == ["nose"]

	frames = frames_per_video_from_labeled_data(root, video_paths=[str(video)])
	assert frames == {"cam0.mp4": [2, 5]}
	assert human_label_frames_per_video(root, video_paths=[str(video)]) == frames


def test_corpus_save_leaves_sibling_folder(tmp_path: Path):
	session_dir = tmp_path / "session_demo"
	cam0 = _tiny_video(session_dir / "cam0.mp4")
	cam1 = _tiny_video(session_dir / "cam1.mp4")
	root = labeled_data_dir(tmp_path / "proj")
	paths = [str(cam0), str(cam1)]

	write_labeled_data_from_wide(
		labeled_data_root=root,
		wide_df=pd.DataFrame(
			[
				{"video": "cam0.mp4", "frame": 1, "nose_x": 1.0, "nose_y": 2.0},
				{"video": "cam1.mp4", "frame": 3, "nose_x": 3.0, "nose_y": 4.0},
			]
		),
		video_paths=paths,
		joint_names=["nose"],
	)
	cam1_csv = video_labeled_folder(root, cam1) / "CollectedData_human.csv"
	cam1_before = cam1_csv.read_text()

	# Update only cam0 — cam1 folder must stay intact.
	write_labeled_data_from_wide(
		labeled_data_root=root,
		wide_df=pd.DataFrame(
			[{"video": "cam0.mp4", "frame": 7, "nose_x": 9.0, "nose_y": 8.0}]
		),
		video_paths=paths,
		joint_names=["nose"],
		only_videos=["cam0.mp4"],
	)

	assert cam1_csv.read_text() == cam1_before
	loaded = wide_df_from_labeled_data(root, paths)
	by_video = {v: g for v, g in loaded.groupby("video")}
	assert set(by_video["cam0.mp4"]["frame"]) == {7}
	assert set(by_video["cam1.mp4"]["frame"]) == {3}


def test_collected_df_to_wide_parses_img_index(tmp_path: Path):
	# Build a CollectedData CSV the same way production write does.
	session_dir = tmp_path / "session_x"
	video = _tiny_video(session_dir / "eye.mp4", n_frames=5)
	root = labeled_data_dir(tmp_path / "proj")
	write_video_labeled_data(
		labeled_data_root=root,
		video_path=video,
		video_rows=pd.DataFrame(
			[{"video": "eye.mp4", "frame": 1, "a_x": 1.5, "a_y": 2.5}]
		),
		joint_names=["a"],
	)
	csv_path = video_labeled_folder(root, video) / "CollectedData_human.csv"
	df = read_collected_data_csv(csv_path)
	wide = collected_df_to_wide_rows(df, "eye.mp4")
	assert list(wide["frame"]) == [1]
	assert wide.iloc[0]["a_x"] == pytest.approx(1.5)


def test_regenerate_all_collected_data_h5_rewrites_siblings(tmp_path: Path):
	"""Stale sibling H5 files are rebuilt from CSV before train."""
	session_dir = tmp_path / "session_demo"
	cam0 = _tiny_video(session_dir / "cam0.mp4")
	cam1 = _tiny_video(session_dir / "cam1.mp4")
	root = labeled_data_dir(tmp_path / "proj")
	paths = [str(cam0), str(cam1)]

	write_labeled_data_from_wide(
		labeled_data_root=root,
		wide_df=pd.DataFrame(
			[
				{"video": "cam0.mp4", "frame": 1, "nose_x": 1.0, "nose_y": 2.0},
				{"video": "cam1.mp4", "frame": 2, "nose_x": 3.0, "nose_y": 4.0},
			]
		),
		video_paths=paths,
		joint_names=["nose"],
	)

	from skellyclicker.core.deeplabcut_handler.labeled_data_io import (
		regenerate_all_collected_data_h5,
		video_labeled_folder,
	)

	# Simulate a broken sibling H5 left from an older writer.
	cam1_h5 = video_labeled_folder(root, cam1) / "CollectedData_human.h5"
	cam1_h5.write_bytes(b"not a valid hdf5 file")

	rewritten = regenerate_all_collected_data_h5(root)
	assert len(rewritten) == 2
	# Either a valid H5 exists, or tables is missing and H5 was removed.
	if cam1_h5.is_file():
		pd.read_hdf(str(cam1_h5), key="df_with_missing")
	else:
		assert (video_labeled_folder(root, cam1) / "CollectedData_human.csv").is_file()


def test_labeled_data_session_subset_hides_other_folders(tmp_path: Path):
	"""Non-session labeled-data folders are held aside during the train subset."""
	from skellyclicker.core.deeplabcut_handler.labeled_data_io import (
		labeled_data_session_subset,
		video_labeled_folder,
	)

	session_dir = tmp_path / "session_demo"
	cam0 = _tiny_video(session_dir / "cam0.mp4")
	other = tmp_path / "other_exp" / "camX.mp4"
	_tiny_video(other)
	root = labeled_data_dir(tmp_path / "proj")

	write_labeled_data_from_wide(
		labeled_data_root=root,
		wide_df=pd.DataFrame(
			[
				{"video": "cam0.mp4", "frame": 1, "nose_x": 1.0, "nose_y": 2.0},
				{"video": "camX.mp4", "frame": 1, "nose_x": 3.0, "nose_y": 4.0},
			]
		),
		video_paths=[str(cam0), str(other)],
		joint_names=["nose"],
	)
	keep = video_labeled_folder(root, cam0)
	other_folder = video_labeled_folder(root, other)
	assert keep.is_dir() and other_folder.is_dir()

	with labeled_data_session_subset(root, [str(cam0)]):
		visible = {p.name for p in root.iterdir() if p.is_dir()}
		assert keep.name in visible
		assert other_folder.name not in visible

	assert other_folder.is_dir()
	assert keep.is_dir()


def test_legacy_human_label_frames_still_works(tmp_path: Path):
	csv = tmp_path / "labels.csv"
	pd.DataFrame(
		[
			{"video": "cam0.mp4", "frame": 0, "nose_x": 1.0, "nose_y": 2.0},
			{"video": "cam0.mp4", "frame": 5, "nose_x": 3.0, "nose_y": 4.0},
		]
	).to_csv(csv, index=False)
	assert human_label_frames_per_video(csv) == {"cam0.mp4": [0, 5]}
