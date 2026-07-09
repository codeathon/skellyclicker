"""Tests for web session store lifecycle (Phase A regressions)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from skellyclicker.services.session_store import SessionStore


@pytest.fixture
def fresh_store():
	return SessionStore()


def test_clear_session_resets_dlc_handler(fresh_store):
	# Simulate a loaded handler without full DLC init.
	fresh_store.dlc_handler = object()  # type: ignore[assignment]
	fresh_store.clear_session()
	assert fresh_store.dlc_handler is None
	assert fresh_store.labeling_engine is None


def test_train_on_machine_requires_csv():
	from skellyclicker.services.session_store import SessionStore
	store = SessionStore()
	store.session.train_on_machine_labels = True
	assert store.session.machine_labels_path is None


def test_bump_generation_on_teardown(fresh_store):
	gen = fresh_store.session.generation
	fresh_store._teardown_all()
	assert fresh_store.session.generation == gen + 1


def test_save_session_bare_filename_uses_home_directory(fresh_store, monkeypatch, tmp_path):
	home = tmp_path / "home"
	home.mkdir()
	monkeypatch.setattr(Path, "home", lambda: home)

	fresh_store.save_session_json("first_test_session.json")
	target = home / "skellyclicker_sessions" / "first_test_session.json"
	assert target.is_file()
	assert fresh_store.session.session_saved_path == str(target.resolve())


def test_save_session_creates_new_json_file(fresh_store, tmp_path):
	path = tmp_path / "first_test_session.json"
	assert not path.is_file()
	fresh_store.save_session_json(str(path))
	assert path.is_file()
	assert fresh_store.session.session_saved_path == str(path.resolve())


def test_save_session_creates_parent_directories(fresh_store, tmp_path):
	path = tmp_path / "nested" / "dir" / "session.json"
	fresh_store.save_session_json(str(path))
	assert path.is_file()


def test_load_session_missing_file_raises_session_error(fresh_store, tmp_path):
	from skellyclicker.services.errors import SessionError

	missing = tmp_path / "first_test_session.json"
	with pytest.raises(SessionError, match="Session file not found"):
		fresh_store.load_session_json(str(missing))


def test_save_session_rejects_non_json_extension(fresh_store, tmp_path):
	from skellyclicker.services.errors import SessionError

	with pytest.raises(SessionError, match="must end with .json"):
		fresh_store.save_session_json(str(tmp_path / "session.txt"))


def test_open_labeler_requires_label_context(fresh_store):
	from skellyclicker.services.errors import SessionError

	fresh_store.session.videos = ["/tmp/fake.mp4"]
	with pytest.raises(SessionError, match="bodyparts"):
		fresh_store.open_labeler()


def test_can_open_labeler_with_dlc_bodyparts_only(fresh_store):
	fresh_store.session.videos = ["/tmp/fake.mp4"]
	fresh_store.session.dlc_project_path = "/tmp/myproject"
	fresh_store.session.tracked_point_names = ["nose", "tail"]
	assert fresh_store._can_open_labeler() is True


def test_can_open_labeler_with_human_csv(fresh_store):
	fresh_store.session.videos = ["/tmp/fake.mp4"]
	fresh_store.session.human_labels_path = "/tmp/labels.csv"
	assert fresh_store._can_open_labeler() is True


def test_remove_video_updates_session(fresh_store, tmp_path):
	v1 = tmp_path / "a.mp4"
	v2 = tmp_path / "b.mp4"
	v1.write_bytes(b"x")
	v2.write_bytes(b"x")
	fresh_store.set_videos([str(v1.resolve()), str(v2.resolve())])
	fresh_store.remove_video(str(v1.resolve()))
	assert fresh_store.session.videos == [str(v2.resolve())]


def test_close_labeler_leaves_labeling_state(fresh_store):
	"""Closing labeler must exit workflow_state=labeling (regression)."""
	from skellyclicker.services.models import WorkflowState
	from skellyclicker.services.workflow import refresh_workflow_state

	fresh_store.session.workflow_state = WorkflowState.labeling
	fresh_store.session.labeling_session_id = None
	fresh_store.session.videos = ["/tmp/fake.mp4"]
	refresh_workflow_state(fresh_store.session)
	assert fresh_store.session.workflow_state != WorkflowState.labeling


def test_training_settings_validation(fresh_store):
	fresh_store.set_training_settings(epochs=50, save_epochs=10, batch_size=4)
	assert fresh_store.session.training_epochs == 50
	assert fresh_store.session.training_save_epochs == 10
	assert fresh_store.session.training_batch_size == 4


def test_training_settings_rejects_invalid(fresh_store):
	from skellyclicker.services.errors import SessionError

	with pytest.raises(SessionError, match="at least"):
		fresh_store.set_training_settings(epochs=0)


def test_analyze_options_round_trip(fresh_store):
	fresh_store.set_analyze_options(filter_predictions=True, annotate_videos=True)
	assert fresh_store.session.filter_predictions is True
	assert fresh_store.session.annotate_videos is True


def test_close_labeler_save_registers_human_labels(fresh_store, tmp_path):
	"""Saving from the labeler always registers human_labels_path."""
	from unittest.mock import MagicMock

	csv_path = tmp_path / "labels.csv"
	csv_path.write_text("video,frame,nose_x,nose_y\ncam1,0,1.0,2.0\n")
	# Real file: finalize drops missing machine paths so Loaded Assets stays honest.
	machine_csv = tmp_path / "machine.csv"
	machine_csv.write_text("video,frame,nose_x,nose_y\n")

	mock_engine = MagicMock()
	mock_engine.session_id = "label-session-1"
	mock_engine.close.return_value = str(csv_path)
	mock_engine.video_handler.data_handler.get_nonempty_frames.return_value = [0]

	fresh_store.labeling_engine = mock_engine
	fresh_store.session.labeling_session_id = "label-session-1"
	fresh_store.session.train_on_machine_labels = True
	fresh_store.session.machine_labels_path = str(machine_csv)

	fresh_store.close_labeler(save=True, save_path=str(csv_path))

	assert fresh_store.session.human_labels_path == str(csv_path)
	assert fresh_store.session.machine_labels_path == str(machine_csv)
	assert fresh_store.labeling_engine is None
	assert "Labels saved to" in fresh_store.session.status_message


def test_save_labeler_keeps_labeler_open(fresh_store, tmp_path):
	"""Save writes CSV without closing the labeler session."""
	from unittest.mock import MagicMock, patch

	csv_path = tmp_path / "labels.csv"
	csv_path.write_text("video,frame,nose_x,nose_y\ncam1,0,1.0,2.0\n")

	mock_engine = MagicMock()
	mock_engine.session_id = "label-session-1"
	mock_engine.save_labels.return_value = str(csv_path)
	mock_engine.video_handler.data_handler.config.tracked_point_names = ["nose"]

	fresh_store.labeling_engine = mock_engine
	fresh_store.session.labeling_session_id = "label-session-1"

	with patch.object(fresh_store, "_labeled_frame_count", return_value=3):
		fresh_store.save_labeler(save_path=str(csv_path))

	mock_engine.save_labels.assert_called_once_with(str(csv_path))
	assert fresh_store.labeling_engine is mock_engine
	assert fresh_store.session.human_labels_path == str(csv_path)
	assert fresh_store.session.labeled_frame_count == 3
	assert "Labels saved to" in fresh_store.session.status_message


def test_save_labeler_rejects_machine_labels_path(fresh_store, tmp_path):
	from unittest.mock import MagicMock

	machine_csv = tmp_path / "machine.csv"
	machine_csv.write_text("video,frame,nose_x,nose_y\n")
	mock_engine = MagicMock()
	fresh_store.labeling_engine = mock_engine
	fresh_store.session.machine_labels_path = str(machine_csv)

	with pytest.raises(Exception, match="machine labels"):
		fresh_store.save_labeler(save_path=str(machine_csv))


def test_finalize_session_keeps_existing_project_machine_labels(fresh_store, tmp_path):
	"""An analyze-set path under the project stays; sync does not invent a newer one."""
	project = tmp_path / "proj"
	project.mkdir()
	config = project / "config.yaml"
	config.write_text("Task: test\niteration: 2\n")
	old_csv = (
		project
		/ "model_outputs"
		/ "model_outputs_iteration_0"
		/ "skellyclicker_machine_labels_iteration_0.csv"
	)
	new_csv = (
		project
		/ "model_outputs"
		/ "model_outputs_iteration_2"
		/ "skellyclicker_machine_labels_iteration_2.csv"
	)
	for csv in (old_csv, new_csv):
		csv.parent.mkdir(parents=True)
		csv.write_text("video,frame,x,y\n")

	fresh_store.session.dlc_project_path = str(project)
	fresh_store.session.machine_labels_path = str(old_csv)
	session = fresh_store.get_session()
	# Sync no longer auto-upgrades to newest on disk — only analyze/import set the path.
	assert session.machine_labels_path == str(old_csv)


def test_finalize_does_not_invent_machine_labels_when_unset(fresh_store, tmp_path):
	"""Loaded Assets must stay empty until analyze/import sets machine labels."""
	project = tmp_path / "proj"
	project.mkdir()
	(project / "config.yaml").write_text("Task: test\niteration: 1\n")
	csv = (
		project
		/ "model_outputs"
		/ "model_outputs_iteration_1"
		/ "skellyclicker_machine_labels_iteration_1.csv"
	)
	csv.parent.mkdir(parents=True)
	csv.write_text("video,frame,nose_x,nose_y\n")

	fresh_store.session.dlc_project_path = str(project)
	fresh_store.session.machine_labels_path = None
	session = fresh_store.get_session()
	assert session.machine_labels_path is None


def test_add_videos_clears_machine_labels_from_assets(fresh_store, tmp_path):
	"""Adding videos must clear Loaded Assets machine path (not auto-discover)."""
	v1 = tmp_path / "a.mp4"
	v1.write_bytes(b"x")
	fresh_store.session.machine_labels_path = str(
		tmp_path / "old_model_outputs_iteration_0" / "skellyclicker_machine_labels_iteration_0.csv"
	)
	with patch(
		"skellyclicker.services.session_store.detect_labeling_mode",
		return_value=__import__(
			"skellyclicker.services.models", fromlist=["LabelingMode"]
		).LabelingMode.single,
	):
		session = fresh_store.add_videos([str(v1)])
	assert session.machine_labels_path is None


def test_finalize_clears_stale_video_folder_machine_labels(fresh_store, tmp_path):
	"""New/unanalyzed project must not keep a CSV from beside the videos."""
	project = tmp_path / "new_proj"
	project.mkdir()
	(project / "config.yaml").write_text("Task: new\niteration: 0\n")
	stale = (
		tmp_path
		/ "videos"
		/ "old_model_outputs_iteration_0"
		/ "skellyclicker_machine_labels_iteration_0.csv"
	)
	stale.parent.mkdir(parents=True)
	stale.write_text("video,frame,a_x,a_y,b_x,b_y,c_x,c_y,d_x,d_y,e_x,e_y,f_x,f_y\n")

	fresh_store.session.dlc_project_path = str(project)
	fresh_store.session.videos = [str(tmp_path / "videos" / "cam.mp4")]
	fresh_store.session.machine_labels_path = str(stale)
	session = fresh_store.get_session()
	assert session.machine_labels_path is None


def test_load_dlc_project_clears_prior_machine_labels(fresh_store, tmp_path, monkeypatch):
	"""Switching projects must not keep the previous machine CSV in assets."""
	project = tmp_path / "proj"
	project.mkdir()
	config = project / "config.yaml"
	config.write_text("Task: t\niteration: 0\n")

	class _Fake:
		iteration = 0
		tracked_point_names = ["nose"]
		project_config_path = str(config)

	# Avoid importing deeplabcut in unit tests — stub the handler load path.
	import types
	import sys

	fake_mod = types.ModuleType("skellyclicker.core.deeplabcut_handler.deeplabcut_handler")

	class DeeplabcutHandler:
		@classmethod
		def load_deeplabcut_project(cls, project_config_path: str):
			return _Fake()

	fake_mod.DeeplabcutHandler = DeeplabcutHandler
	monkeypatch.setitem(
		sys.modules,
		"skellyclicker.core.deeplabcut_handler.deeplabcut_handler",
		fake_mod,
	)
	fresh_store.session.machine_labels_path = "/tmp/old_machine.csv"
	# Also stub live-inference so load does not need a real model tree.
	monkeypatch.setattr(fresh_store, "_ensure_live_inference", lambda: None)
	fresh_store.load_dlc_project(str(project))
	assert fresh_store.session.machine_labels_path is None


def test_ensure_live_inference_skips_without_trained_weights(fresh_store, tmp_path):
	"""Live overlays stay off until a .pt snapshot exists under dlc-models-pytorch."""
	from skellyclicker.services.dlc_paths import PYTORCH_MODELS_DIR, PYTORCH_TRAIN_CONFIG

	project = tmp_path / "proj"
	config = project / "config.yaml"
	config.parent.mkdir(parents=True)
	config.write_text("Task: t\niteration: 0\n")
	train = project / PYTORCH_MODELS_DIR / "iteration-0" / "shuffle1" / "train"
	train.mkdir(parents=True)
	(train / PYTORCH_TRAIN_CONFIG).write_text("method: bu\n")

	class _FakeDlc:
		project_config_path = str(config)

	fresh_store.dlc_handler = _FakeDlc()  # type: ignore[assignment]
	fresh_store.live_inference = object()  # type: ignore[assignment]
	fresh_store._ensure_live_inference()
	assert fresh_store.live_inference is None
