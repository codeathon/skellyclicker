"""Workflow state derivation and transition helpers."""

from skellyclicker.services.models import AppSession, WorkflowState


def derive_workflow_state(session: AppSession) -> WorkflowState:
	"""Compute workflow state from session fields (unless a job is active)."""
	if session.active_job_id and session.workflow_state in (
		WorkflowState.training,
		WorkflowState.analyzing,
	):
		return session.workflow_state
	if session.labeling_session_id:
		return WorkflowState.labeling
	if not session.videos:
		return WorkflowState.needs_videos if session.session_saved_path else WorkflowState.idle
	if session.machine_labels_path and session.labeled_frame_count == 0:
		return WorkflowState.review
	if not session.dlc_project_path:
		return WorkflowState.needs_project
	if session.dlc_iteration is None:
		return WorkflowState.ready_to_train
	return WorkflowState.ready_to_analyze


def refresh_workflow_state(session: AppSession) -> AppSession:
	"""Recompute workflow_state from current assets and runtime handles."""
	if session.active_job_id and session.workflow_state in (
		WorkflowState.training,
		WorkflowState.analyzing,
	):
		return session
	session.workflow_state = derive_workflow_state(session)
	return session


def _has_videos(session: AppSession) -> bool:
	return bool(session.videos)


def _has_dlc(session: AppSession) -> bool:
	return bool(session.dlc_project_path)


def _labels_for_train(session: AppSession) -> str | None:
	# Training always requires human labels (machine CSV is overlay / full analyze only).
	return session.human_labels_path


def build_workflow_hints(session: AppSession) -> dict:
	"""Prerequisites for train/analyze — mirrors frontend workflowSteps gating."""
	missing_train: list[str] = []
	missing_analyze: list[str] = []

	if not _has_dlc(session):
		missing_train.append("Load or create a DLC project")
		missing_analyze.append("Load or create a DLC project")
	if not _has_videos(session):
		missing_train.append("Add videos")
		missing_analyze.append("Add videos")
	if not _labels_for_train(session):
		missing_train.append("Label videos or import human labels before training")
	if session.dlc_iteration is None:
		missing_analyze.append("Train the network before analyzing")

	if session.active_job_id:
		missing_train.append("Wait for the current job to finish")
		missing_analyze.append("Wait for the current job to finish")

	return {
		"can_train": len(missing_train) == 0,
		"can_analyze": len(missing_analyze) == 0,
		"missing_train": missing_train,
		"missing_analyze": missing_analyze,
	}
