"""Central session store — single source of truth for web API."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from skellyclicker.core.session_validation import bodypart_names_from_csv_columns
from skellyclicker.services.dlc_job_runner import DLCJobRunner
from skellyclicker.services.errors import SessionConflictError, SessionError
from skellyclicker.services.labeling_engine import LabelingEngine
from skellyclicker.services.models import AppSession, BackgroundJob, WorkflowState
from skellyclicker.services.workflow import refresh_workflow_state

from skellyclicker.services.dlc_paths import resolve_dlc_project_input


class SessionStore:
	"""Thread-unsafe singleton for local lab use; one user per server process."""

	def __init__(self) -> None:
		self.session = AppSession()
		self.labeling_engine: LabelingEngine | None = None
		self.dlc_handler: Any | None = None
		self.jobs: dict[str, BackgroundJob] = {}
		self.job_runner = DLCJobRunner(self)

	def get_session(self) -> AppSession:
		return self.session

	def _bump_generation(self) -> None:
		self.session.generation += 1

	def _assert_no_active_job(self) -> None:
		if self.session.active_job_id:
			raise SessionConflictError(
				"A background job is running. Wait for it to finish.",
			)

	def _teardown_labeler(self) -> None:
		if self.labeling_engine:
			self.labeling_engine.video_handler.close(save_data=False)
			self.labeling_engine = None
			self.session.labeling_session_id = None

	def _teardown_all(self) -> None:
		self._teardown_labeler()
		self.dlc_handler = None
		self._bump_generation()

	def start_new_session(self) -> AppSession:
		self._assert_no_active_job()
		self._teardown_all()
		self.session = AppSession(workflow_state=WorkflowState.needs_videos)
		return self.session

	def clear_session(self) -> AppSession:
		self._assert_no_active_job()
		self._teardown_all()
		self.session = AppSession()
		return refresh_workflow_state(self.session)

	def _validate_video_paths(self, paths: list[str]) -> list[str]:
		if not paths:
			raise SessionError("No video paths provided")
		resolved: list[str] = []
		for path in paths:
			p = Path(path).expanduser()
			if not p.is_file():
				raise SessionError(f"Video not found: {path}")
			resolved.append(str(p.resolve()))
		return resolved

	def _apply_videos(self, paths: list[str]) -> None:
		"""Store video list and reset frame stats; close labeler if video set changed."""
		if paths != (self.session.videos or []):
			self._teardown_labeler()
		self.session.videos = paths
		self.session.frame_count = 0
		self.session.workflow_state = WorkflowState.ready_to_label
		self.session.status_message = f"{len(paths)} video(s) selected"

	def set_videos(self, paths: list[str]) -> AppSession:
		self._assert_no_active_job()
		self._apply_videos(self._validate_video_paths(paths))
		return self.session

	def add_videos(self, paths: list[str]) -> AppSession:
		"""Append videos to the session list (deduplicated, order preserved)."""
		self._assert_no_active_job()
		new_paths = self._validate_video_paths(paths)
		existing = list(self.session.videos or [])
		seen = set(existing)
		for path in new_paths:
			if path not in seen:
				existing.append(path)
				seen.add(path)
		self._apply_videos(existing)
		return self.session

	def set_human_labels_path(self, path: str) -> AppSession:
		self._assert_no_active_job()
		if not Path(path).is_file():
			raise SessionError(f"CSV not found: {path}")
		self._teardown_labeler()
		import pandas as pd
		df = pd.read_csv(path)
		self.session.human_labels_path = path
		self.session.tracked_point_names = bodypart_names_from_csv_columns(list(df.columns))
		return refresh_workflow_state(self.session)

	def set_machine_labels_path(self, path: str) -> AppSession:
		self._assert_no_active_job()
		if not Path(path).is_file():
			raise SessionError(f"CSV not found: {path}")
		self._teardown_labeler()
		import pandas as pd
		df = pd.read_csv(path)
		self.session.machine_labels_path = path
		if self.session.train_on_machine_labels:
			self.session.tracked_point_names = bodypart_names_from_csv_columns(list(df.columns))
		self.session.workflow_state = WorkflowState.review
		return self.session

	def open_labeler(self) -> AppSession:
		self._assert_no_active_job()
		if not self.session.videos:
			raise SessionError("Select videos first")
		self._teardown_labeler()
		engine = LabelingEngine.open(
			video_paths=self.session.videos,
			human_labels_path=self.session.human_labels_path,
			machine_labels_path=self.session.machine_labels_path,
			train_on_machine_labels=self.session.train_on_machine_labels,
		)
		self.labeling_engine = engine
		self.session.labeling_session_id = engine.session_id
		self.session.frame_count = engine.video_handler.frame_count
		self.session.tracked_point_names = (
			engine.video_handler.data_handler.config.tracked_point_names
		)
		self.session.workflow_state = WorkflowState.labeling
		self.session.status_message = "Labeling"
		return self.session

	def close_labeler(self, save: bool, save_path: str | None = None) -> AppSession:
		if not self.labeling_engine:
			raise SessionError("Labeler is not open")
		engine = self.labeling_engine
		labeling_id = engine.session_id
		labeled_count = len(engine.video_handler.data_handler.get_nonempty_frames())
		path = engine.close(save=save, save_path=save_path)
		if self.session.labeling_session_id != labeling_id:
			return self.session
		if save and path:
			if self.session.train_on_machine_labels:
				self.session.machine_labels_path = path
			else:
				self.session.human_labels_path = path
		self.labeling_engine = None
		self.session.labeling_session_id = None
		self.session.labeled_frame_count = labeled_count
		self.session.status_message = "Labels saved" if save else "Labeling closed"
		return refresh_workflow_state(self.session)

	def _sync_dlc_iteration_from_handler(self) -> None:
		if self.dlc_handler is not None:
			self.session.dlc_iteration = self.dlc_handler.iteration

	def load_dlc_project(self, project_path: str) -> AppSession:
		from skellyclicker.core.deeplabcut_handler.deeplabcut_handler import (
			DeeplabcutHandler,
		)

		self._assert_no_active_job()
		try:
			project_dir, config_path = resolve_dlc_project_input(project_path)
		except ValueError as exc:
			raise SessionError(str(exc)) from exc
		self.dlc_handler = DeeplabcutHandler.load_deeplabcut_project(
			project_config_path=str(config_path.resolve())
		)
		# Store resolved project dir — same folder that contains the loaded config.yaml.
		self.session.dlc_project_path = str(project_dir)
		self._sync_dlc_iteration_from_handler()
		if self.dlc_handler.tracked_point_names:
			self.session.tracked_point_names = self.dlc_handler.tracked_point_names
		return refresh_workflow_state(self.session)

	def save_session_json(self, path: str) -> AppSession:
		self.session.session_saved_path = path
		Path(path).write_text(self.session.model_dump_json(indent=2))
		return self.session

	def load_session_json(self, path: str) -> AppSession:
		self._assert_no_active_job()
		self._teardown_all()
		import json
		data = json.loads(Path(path).read_text())
		self.session = AppSession.model_validate(data)
		if self.session.dlc_project_path:
			from skellyclicker.core.deeplabcut_handler.deeplabcut_handler import (
				DeeplabcutHandler,
			)

			try:
				_, config_path = resolve_dlc_project_input(self.session.dlc_project_path)
			except ValueError:
				config_path = None
			if config_path is not None:
				self.dlc_handler = DeeplabcutHandler.load_deeplabcut_project(
					project_config_path=str(config_path.resolve())
				)
				self._sync_dlc_iteration_from_handler()
		return refresh_workflow_state(self.session)


# Module-level singleton used by FastAPI routes.
store = SessionStore()
