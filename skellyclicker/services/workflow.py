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
