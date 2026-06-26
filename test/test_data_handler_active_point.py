"""Tests for bodypart order and active-point selection when loading labels."""

from __future__ import annotations

from skellyclicker.core.click_data_handler.data_handler import DataHandler, DataHandlerConfig
from skellyclicker.core.video_handler.video_models import ClickData


def test_from_csv_uses_canonical_bodypart_order(tmp_path):
	# CSV columns are tail before nose — session/DLC order should win.
	csv_path = tmp_path / "labels.csv"
	csv_path.write_text(
		"video,frame,tail_base_x,tail_base_y,nose_x,nose_y\n"
		"cam0.mp4,0,1.0,2.0,3.0,4.0\n"
	)
	handler = DataHandler.from_csv(
		csv_path,
		video_names=["cam0.mp4"],
		num_frames=100,
		tracked_point_names=["nose", "tail_base"],
	)
	assert handler.config.tracked_point_names == ["nose", "tail_base"]
	assert handler.active_point == "nose"


def test_reset_active_point_starts_at_first_bodypart_on_empty_frame():
	config = DataHandlerConfig(
		num_frames=10,
		video_names=["cam0.mp4"],
		tracked_point_names=["nose", "tail_base", "ear"],
	)
	handler = DataHandler.from_config(config)
	handler.active_point = "tail_base"
	handler.reset_active_point_for_frame(0)
	assert handler.active_point == "nose"


def test_reset_active_point_skips_to_first_unlabeled_on_partial_frame():
	config = DataHandlerConfig(
		num_frames=10,
		video_names=["cam0.mp4"],
		tracked_point_names=["nose", "tail_base", "ear"],
	)
	handler = DataHandler.from_config(config)
	handler.dataframe.loc[("cam0.mp4", 0), "nose_x"] = 5.0
	handler.dataframe.loc[("cam0.mp4", 0), "nose_y"] = 6.0
	handler.reset_active_point_for_frame(0)
	assert handler.active_point == "tail_base"


def test_click_sequence_starts_with_first_bodypart():
	config = DataHandlerConfig(
		num_frames=10,
		video_names=["cam0.mp4"],
		tracked_point_names=["nose", "tail_base"],
	)
	handler = DataHandler.from_config(config)
	assert handler.active_point == "nose"
	click = ClickData(
		video_index=0,
		frame_number=0,
		window_x=0,
		window_y=0,
		video_x=10,
		video_y=20,
	)
	handler.update_dataframe(click)
	handler.move_active_point_by_index(1)
	assert handler.active_point == "tail_base"


def test_from_csv_dense_merge_matches_sparse_rows(tmp_path):
	"""Vectorized merge must preserve labeled coordinates on dense DLC-style CSVs."""
	csv_path = tmp_path / "dense.csv"
	lines = ["video,frame,nose_x,nose_y"]
	for frame in range(200):
		lines.append(f"cam0.mp4,{frame},{frame}.5,{frame}.25")
	csv_path.write_text("\n".join(lines) + "\n")

	handler = DataHandler.from_csv(
		csv_path,
		video_names=["cam0.mp4"],
		num_frames=200,
		tracked_point_names=["nose"],
	)
	data = handler.get_data_by_video_frame(0, 42)
	assert data["nose"].video_x == 42
	assert data["nose"].video_y == 42


def test_from_csv_overlay_skips_empty_frames(tmp_path):
	csv_path = tmp_path / "machine.csv"
	csv_path.write_text(
		"video,frame,nose_x,nose_y\n"
		"cam0.mp4,5,10.0,20.0\n"
	)
	handler = DataHandler.from_csv_overlay(
		csv_path,
		video_names=["cam0.mp4"],
		num_frames=100,
		tracked_point_names=["nose"],
	)
	assert handler.get_data_by_video_frame(0, 5)["nose"].video_x == 10
	assert handler.get_data_by_video_frame(0, 0) == {}


def test_from_csv_overlay_remaps_session_video_names(tmp_path):
	"""Machine overlay must follow session video basenames after import-then-add-video."""
	csv_path = tmp_path / "machine.csv"
	csv_path.write_text(
		"video,frame,nose_x,nose_y\n"
		"session_cam.mp4,3,11.0,22.0\n"
	)
	handler = DataHandler.from_csv_overlay(
		csv_path,
		video_names=["ferret_left.avi"],
		num_frames=100,
		tracked_point_names=["nose"],
	)
	assert handler.get_data_by_video_frame(0, 3)["nose"].video_x == 11
