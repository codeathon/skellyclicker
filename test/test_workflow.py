"""Tests for workflow hints and state helpers."""

from skellyclicker.services.models import AppSession, WorkflowState
from skellyclicker.services.workflow import build_workflow_hints, derive_workflow_state


def _base_session(**kwargs) -> AppSession:
	data = {
		"session_id": "test",
		"generation": 1,
		"workflow_state": WorkflowState.idle,
		"session_saved_path": None,
		"videos": None,
		"human_labels_path": None,
		"machine_labels_path": None,
		"dlc_project_path": None,
		"dlc_iteration": None,
		"tracked_point_names": [],
		"labeled_frame_count": 0,
		"frame_count": 0,
		"train_on_machine_labels": False,
		"auto_save_session": False,
		"labeling_session_id": None,
		"active_job_id": None,
		"status_message": "",
		"training_epochs": 200,
		"training_save_epochs": 20,
		"training_batch_size": 8,
		"filter_predictions": False,
		"annotate_videos": False,
	}
	data.update(kwargs)
	return AppSession(**data)


def test_build_workflow_hints_empty_session():
	hints = build_workflow_hints(_base_session())
	assert hints["can_train"] is False
	assert hints["can_analyze"] is False
	assert "Add videos" in hints["missing_train"]
	assert "Train the network before analyzing" in hints["missing_analyze"]


def test_build_workflow_hints_ready_to_train():
	session = _base_session(
		videos=["/tmp/v.mp4"],
		dlc_project_path="/tmp/proj",
		human_labels_path="/tmp/labels.csv",
	)
	hints = build_workflow_hints(session)
	assert hints["can_train"] is True
	assert hints["can_analyze"] is False
	assert "Train the network before analyzing" in hints["missing_analyze"]


def test_build_workflow_hints_ready_to_analyze():
	session = _base_session(
		videos=["/tmp/v.mp4"],
		dlc_project_path="/tmp/proj",
		human_labels_path="/tmp/labels.csv",
		dlc_iteration=0,
	)
	hints = build_workflow_hints(session)
	assert hints["can_train"] is True
	assert hints["can_analyze"] is True


def test_build_workflow_hints_blocks_during_job():
	session = _base_session(
		videos=["/tmp/v.mp4"],
		dlc_project_path="/tmp/proj",
		human_labels_path="/tmp/labels.csv",
		dlc_iteration=0,
		active_job_id="job-1",
	)
	hints = build_workflow_hints(session)
	assert hints["can_train"] is False
	assert hints["can_analyze"] is False


def test_derive_workflow_state_review_with_machine_labels():
	session = _base_session(
		videos=["/tmp/v.mp4"],
		machine_labels_path="/tmp/machine.csv",
		labeled_frame_count=0,
	)
	assert derive_workflow_state(session) == WorkflowState.review
