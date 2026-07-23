"""Shared session and workflow models for the web API."""

from enum import Enum
from typing import List
from uuid import uuid4

from pydantic import BaseModel, Field


class WorkflowState(str, Enum):
	"""Drives stepper UI and API gating."""
	idle = "idle"
	needs_videos = "needs_videos"
	ready_to_label = "ready_to_label"
	labeling = "labeling"
	needs_project = "needs_project"
	ready_to_train = "ready_to_train"
	training = "training"
	ready_to_analyze = "ready_to_analyze"
	analyzing = "analyzing"
	review = "review"


class LabelingMode(str, Enum):
	"""Transparent labeler layout — picked from uploaded videos, not a user mode picker."""

	single = "single"
	synced = "synced"
	corpus = "corpus"


class AssetPathCheck(BaseModel):
	"""Filesystem check for a path referenced by the session (computed, not persisted)."""
	kind: str
	path: str
	exists: bool


class AppSession(BaseModel):
	"""Single source of truth for application state."""
	session_id: str = Field(default_factory=lambda: str(uuid4()))
	generation: int = 0
	workflow_state: WorkflowState = WorkflowState.idle
	session_saved_path: str | None = None
	videos: List[str] | None = None
	# Auto-detected from video frame counts; drives grid vs single-video labeler.
	labeling_mode: LabelingMode = LabelingMode.single
	# Absolute path of the video shown in corpus/single labeler (None = synced grid).
	active_video_path: str | None = None
	human_labels_path: str | None = None
	machine_labels_path: str | None = None
	dlc_project_path: str | None = None
	dlc_iteration: int | None = None
	tracked_point_names: List[str] = []
	labeled_frame_count: int = 0
	frame_count: int = 0
	train_on_machine_labels: bool = False
	auto_save_session: bool = False
	labeling_session_id: str | None = None
	active_job_id: str | None = None
	status_message: str = "Ready"
	training_epochs: int = 200
	training_save_epochs: int = 20
	training_batch_size: int = 8
	filter_predictions: bool = False
	annotate_videos: bool = False
	# Full Analysis: max videos analyzed at once. 0 = auto (one worker per GPU).
	analyze_parallel_workers: int = 0
	asset_path_checks: List[AssetPathCheck] = Field(default_factory=list)


class JobStatus(str, Enum):
	pending = "pending"
	running = "running"
	completed = "completed"
	failed = "failed"


class BackgroundJob(BaseModel):
	job_id: str = Field(default_factory=lambda: str(uuid4()))
	name: str
	status: JobStatus = JobStatus.pending
	message: str = ""
	log_lines: List[str] = Field(default_factory=list)
	# None = indeterminate (e.g. long DLC train); 0.0–1.0 = determinate fraction.
	progress_percent: float | None = None
