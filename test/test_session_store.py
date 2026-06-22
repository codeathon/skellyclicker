"""Tests for web session store lifecycle (Phase A regressions)."""

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


def test_close_labeler_leaves_labeling_state(fresh_store):
	"""Closing labeler must exit workflow_state=labeling (regression)."""
	from skellyclicker.services.models import WorkflowState
	from skellyclicker.services.workflow import refresh_workflow_state

	fresh_store.session.workflow_state = WorkflowState.labeling
	fresh_store.session.labeling_session_id = None
	fresh_store.session.videos = ["/tmp/fake.mp4"]
	refresh_workflow_state(fresh_store.session)
	assert fresh_store.session.workflow_state != WorkflowState.labeling


def test_close_labeler_save_registers_human_labels(fresh_store, tmp_path):
	"""Saving from the labeler always registers human_labels_path."""
	from unittest.mock import MagicMock

	csv_path = tmp_path / "labels.csv"
	csv_path.write_text("video,frame,nose_x,nose_y\ncam1,0,1.0,2.0\n")

	mock_engine = MagicMock()
	mock_engine.session_id = "label-session-1"
	mock_engine.close.return_value = str(csv_path)
	mock_engine.video_handler.data_handler.get_nonempty_frames.return_value = [0]

	fresh_store.labeling_engine = mock_engine
	fresh_store.session.labeling_session_id = "label-session-1"
	fresh_store.session.train_on_machine_labels = True
	fresh_store.session.machine_labels_path = "/tmp/machine.csv"

	fresh_store.close_labeler(save=True, save_path=str(csv_path))

	assert fresh_store.session.human_labels_path == str(csv_path)
	assert fresh_store.session.machine_labels_path == "/tmp/machine.csv"
	assert fresh_store.labeling_engine is None
	assert "Labels saved to" in fresh_store.session.status_message
