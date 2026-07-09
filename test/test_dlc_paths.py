"""Tests for DLC path resolution."""

import tempfile
from pathlib import Path

import pytest

from skellyclicker.services.dlc_paths import (
	latest_machine_labels_csv,
	machine_labels_iteration,
	resolve_dlc_project_input,
	resolve_latest_machine_labels_path,
	resolve_partial_machine_labels_path,
)


def test_resolve_project_directory():
	with tempfile.TemporaryDirectory() as tmp:
		root = Path(tmp)
		(root / "config.yaml").write_text("Task: test\n")
		project_dir, config_path = resolve_dlc_project_input(str(root))
		assert project_dir == root.resolve()
		assert config_path == (root / "config.yaml").resolve()


def test_resolve_config_file_path():
	with tempfile.TemporaryDirectory() as tmp:
		root = Path(tmp)
		cfg = root / "config.yaml"
		cfg.write_text("Task: test\n")
		project_dir, config_path = resolve_dlc_project_input(str(cfg))
		assert project_dir == root.resolve()
		assert config_path == cfg.resolve()


def test_resolve_rejects_missing_config():
	with tempfile.TemporaryDirectory() as tmp:
		with pytest.raises(ValueError):
			resolve_dlc_project_input(tmp)


def test_resolve_partial_targets_current_iteration_not_old_session_path(tmp_path, monkeypatch):
	project = tmp_path / "proj"
	project.mkdir()
	config = project / "config.yaml"
	config.write_text("Task: test\niteration: 1\n")
	dense_csv = (
		project
		/ "model_outputs"
		/ "model_outputs_iteration_0"
		/ "skellyclicker_machine_labels_iteration_0.csv"
	)
	dense_csv.parent.mkdir(parents=True)
	dense_csv.write_text("video,frame,x,y\n" + "\n".join(f"v,0,{i},0" for i in range(200)))
	out1 = project / "model_outputs" / "model_outputs_iteration_1"

	def fake_analyze_output_folder(*_args, **_kwargs):
		return out1

	monkeypatch.setattr(
		"skellyclicker.services.dlc_paths.analyze_output_folder",
		fake_analyze_output_folder,
	)

	expected = out1 / "skellyclicker_machine_labels_iteration_1.csv"
	target = resolve_partial_machine_labels_path(
		str(config),
		analyze_iter=1,
		use_training_videos=True,
		video_paths=[],
		session_machine_labels_path=str(dense_csv),
	)
	assert target == expected
	assert expected.is_file()
	assert expected.read_text() == dense_csv.read_text()


def test_resolve_partial_copies_dense_base_when_new_iteration_missing(tmp_path, monkeypatch):
	project = tmp_path / "proj"
	project.mkdir()
	config = project / "config.yaml"
	config.write_text("Task: test\niteration: 1\n")
	dense_csv = (
		project
		/ "model_outputs"
		/ "model_outputs_iteration_0"
		/ "skellyclicker_machine_labels_iteration_0.csv"
	)
	dense_csv.parent.mkdir(parents=True)
	dense_csv.write_text("video,frame,x,y\n" + "\n".join(f"v,0,{i},0" for i in range(200)))
	out1 = project / "model_outputs" / "model_outputs_iteration_1"

	def fake_analyze_output_folder(*_args, **_kwargs):
		return out1

	monkeypatch.setattr(
		"skellyclicker.services.dlc_paths.analyze_output_folder",
		fake_analyze_output_folder,
	)

	target = resolve_partial_machine_labels_path(
		str(config),
		analyze_iter=1,
		use_training_videos=True,
		video_paths=[],
		session_machine_labels_path=None,
	)
	expected = out1 / "skellyclicker_machine_labels_iteration_1.csv"
	assert target == expected
	assert expected.is_file()
	assert expected.read_text() == dense_csv.read_text()


def test_resolve_partial_replaces_sparse_session_csv_with_dense_base(tmp_path, monkeypatch):
	project = tmp_path / "proj"
	project.mkdir()
	config = project / "config.yaml"
	config.write_text("Task: test\niteration: 1\n")
	dense_csv = (
		project
		/ "model_outputs"
		/ "model_outputs_iteration_0"
		/ "skellyclicker_machine_labels_iteration_0.csv"
	)
	dense_csv.parent.mkdir(parents=True)
	dense_csv.write_text("video,frame,x,y\n" + "\n".join(f"v,0,{i},0" for i in range(200)))
	sparse_csv = (
		project
		/ "model_outputs"
		/ "model_outputs_iteration_1"
		/ "skellyclicker_machine_labels_iteration_1.csv"
	)
	sparse_csv.parent.mkdir(parents=True)
	sparse_csv.write_text("video,frame,x,y\nv,0,1,1")

	def fake_analyze_output_folder(*_args, **_kwargs):
		return sparse_csv.parent

	monkeypatch.setattr(
		"skellyclicker.services.dlc_paths.analyze_output_folder",
		fake_analyze_output_folder,
	)

	target = resolve_partial_machine_labels_path(
		str(config),
		analyze_iter=1,
		use_training_videos=True,
		video_paths=[],
		session_machine_labels_path=str(sparse_csv),
	)
	assert target == sparse_csv.resolve()
	assert sparse_csv.read_text() == dense_csv.read_text()


def test_machine_labels_iteration_from_filename():
	assert machine_labels_iteration(Path("skellyclicker_machine_labels_iteration_9.csv")) == 9
	assert machine_labels_iteration(Path("other.csv")) is None


def test_latest_machine_labels_csv_picks_highest_iteration(tmp_path):
	project = tmp_path / "proj"
	project.mkdir()
	for iteration in (0, 2, 1):
		csv = (
			project
			/ "model_outputs"
			/ f"model_outputs_iteration_{iteration}"
			/ f"skellyclicker_machine_labels_iteration_{iteration}.csv"
		)
		csv.parent.mkdir(parents=True)
		csv.write_text(f"iteration={iteration}\n")
	latest = latest_machine_labels_csv(project)
	assert latest is not None
	assert machine_labels_iteration(latest) == 2


def test_resolve_latest_machine_labels_path(tmp_path):
	project = tmp_path / "proj"
	project.mkdir()
	config = project / "config.yaml"
	config.write_text("Task: test\niteration: 1\n")
	csv = (
		project
		/ "model_outputs"
		/ "model_outputs_iteration_1"
		/ "skellyclicker_machine_labels_iteration_1.csv"
	)
	csv.parent.mkdir(parents=True)
	csv.write_text("video,frame,x,y\n")
	assert resolve_latest_machine_labels_path(str(config)) == csv.resolve()


def test_iteration_has_pytorch_model_requires_snapshot_weights(tmp_path):
	from skellyclicker.services.dlc_paths import (
		PYTORCH_MODELS_DIR,
		PYTORCH_TRAIN_CONFIG,
		iteration_has_pytorch_model,
	)

	project = tmp_path / "proj"
	# Shuffle folder name matches DLC layout; only config is not "trained".
	shuffle = project / PYTORCH_MODELS_DIR / "iteration-0" / "shuffle1"
	train = shuffle / "train"
	train.mkdir(parents=True)
	(train / PYTORCH_TRAIN_CONFIG).write_text("method: bu\n")
	assert iteration_has_pytorch_model(project, 0) is False
	(train / "snapshot-100.pt").write_bytes(b"fake")
	assert iteration_has_pytorch_model(project, 0) is True


def test_resolve_latest_ignores_video_folder_csvs_when_video_paths_none(tmp_path):
	"""Project-only sync must not pick up leftover CSVs beside videos."""
	project = tmp_path / "proj"
	project.mkdir()
	config = project / "config.yaml"
	config.write_text("Task: test\niteration: 0\n")
	videos = tmp_path / "videos"
	videos.mkdir()
	stale = (
		videos
		/ "old_model_outputs_iteration_0"
		/ "skellyclicker_machine_labels_iteration_0.csv"
	)
	stale.parent.mkdir(parents=True)
	stale.write_text("video,frame,nose_x,nose_y\n")
	assert resolve_latest_machine_labels_path(str(config), video_paths=None) is None
	# Explicit video_paths still finds them (analyze / partial patch paths).
	assert (
		resolve_latest_machine_labels_path(str(config), video_paths=[str(videos / "a.mp4")])
		== stale.resolve()
	)
