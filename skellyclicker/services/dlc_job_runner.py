"""Background DLC train/analyze jobs with WebSocket-friendly status updates."""

import threading
from collections.abc import Callable
from pathlib import Path

from skellyclicker.services.models import BackgroundJob, JobStatus, WorkflowState

ProgressCallback = Callable[[float | None, str], None]


class DLCJobRunner:
	"""Runs train/analyze in daemon threads; snapshots handler at job start."""

	def __init__(self, session_store) -> None:
		self._store = session_store

	def _append_log(self, job: BackgroundJob, line: str) -> None:
		job.log_lines.append(line)
		job.message = line

	def _set_progress(
		self,
		job: BackgroundJob,
		fraction: float | None,
		message: str,
	) -> None:
		"""Update job progress for WebSocket clients; None = indeterminate bar."""
		job.progress_percent = fraction
		job.message = message
		if not job.log_lines or job.log_lines[-1] != message:
			job.log_lines.append(message)

	def start_train(self) -> BackgroundJob:
		from skellyclicker.core.deeplabcut_handler.create_deeplabcut.deelabcut_project_config import (
			DeeplabcutTrainingConfig,
		)

		store = self._store
		session = store.session
		if not store.dlc_handler:
			raise ValueError("Load a DLC project first")
		if not session.videos:
			raise ValueError("Select videos first")
		csv_path = (
			session.machine_labels_path
			if session.train_on_machine_labels
			else session.human_labels_path
		)
		if not csv_path:
			raise ValueError("Load or save labels before training")

		handler = store.dlc_handler
		videos = list(session.videos)
		config = DeeplabcutTrainingConfig(
			epochs=session.training_epochs,
			save_epochs=session.training_save_epochs,
			batch_size=session.training_batch_size,
		)
		job = BackgroundJob(name="Train Network")
		store.jobs[job.job_id] = job
		session.active_job_id = job.job_id
		session.workflow_state = WorkflowState.training
		session.status_message = "Training network…"

		def worker() -> None:
			try:
				job.status = JobStatus.running
				self._set_progress(job, None, "Training started…")

				def on_train_progress(fraction: float | None, message: str) -> None:
					self._set_progress(job, fraction, message)

				handler.train_model(
					labels_csv_path=csv_path,
					video_paths=videos,
					training_config=config,
					progress_callback=on_train_progress,
				)
				session.dlc_iteration = handler.iteration
				job.status = JobStatus.completed
				self._set_progress(
					job,
					1.0,
					f"Training complete (iteration {handler.iteration})",
				)
			except Exception as exc:
				job.status = JobStatus.failed
				self._set_progress(job, None, str(exc))
			finally:
				session.active_job_id = None
				session.workflow_state = WorkflowState.ready_to_analyze
				session.status_message = job.message

		threading.Thread(target=worker, daemon=True).start()
		return job

	def start_analyze(
		self,
		video_paths: list[str],
		use_training_videos: bool,
	) -> BackgroundJob:
		store = self._store
		session = store.session
		if not store.dlc_handler:
			raise ValueError("Load a DLC project first")
		if not video_paths:
			raise ValueError("No videos to analyze")

		handler = store.dlc_handler
		job = BackgroundJob(name="Analyze Videos")
		store.jobs[job.job_id] = job
		session.active_job_id = job.job_id
		session.workflow_state = WorkflowState.analyzing
		session.status_message = "Analyzing videos…"

		def worker() -> None:
			try:
				from deeplabcut.utils import auxiliaryfunctions

				from skellyclicker.services.dlc_paths import (
					analyze_output_folder,
					dlc_project_dir,
					resolve_analyze_iteration,
				)

				job.status = JobStatus.running
				self._set_progress(job, 0.0, "Analysis started…")
				project_dir = dlc_project_dir(handler.project_config_path)
				cfg = auxiliaryfunctions.read_config(handler.project_config_path)
				analyze_iter = resolve_analyze_iteration(project_dir, cfg)
				handler.iteration = analyze_iter
				session.dlc_iteration = analyze_iter
				if analyze_iter != int(cfg["iteration"]):
					self._set_progress(
						job,
						0.02,
						f"Using iteration-{analyze_iter} (config.yaml says {cfg['iteration']})",
					)
				output_folder = analyze_output_folder(
					handler.project_config_path,
					use_training_videos,
					video_paths,
					iteration=analyze_iter,
				)
				self._set_progress(job, 0.04, f"Output folder: {output_folder}")

				def on_analyze_progress(fraction: float | None, message: str) -> None:
					self._set_progress(job, fraction, message)

				machine_path = handler.analyze_videos(
					video_paths=video_paths,
					annotate_videos=session.annotate_videos,
					filter_videos=session.filter_predictions,
					output_folder=output_folder,
					progress_callback=on_analyze_progress,
				)
				if use_training_videos:
					session.machine_labels_path = machine_path
					session.workflow_state = WorkflowState.review
				job.status = JobStatus.completed
				self._set_progress(job, 1.0, f"Analysis complete: {machine_path}")
			except Exception as exc:
				job.status = JobStatus.failed
				self._set_progress(job, None, str(exc))
			finally:
				session.active_job_id = None
				if session.workflow_state == WorkflowState.analyzing:
					session.workflow_state = WorkflowState.ready_to_analyze
				session.status_message = job.message

		threading.Thread(target=worker, daemon=True).start()
		return job

	def start_partial_analyze(
		self,
		video_paths: list[str],
		use_training_videos: bool,
	) -> BackgroundJob:
		store = self._store
		session = store.session
		if not store.dlc_handler:
			raise ValueError("Load a DLC project first")
		if not video_paths:
			raise ValueError("No videos to analyze")
		if not session.human_labels_path:
			raise ValueError("Load or save human labels before partial analysis")

		handler = store.dlc_handler
		job = BackgroundJob(name="Partial Analysis")
		store.jobs[job.job_id] = job
		session.active_job_id = job.job_id
		session.workflow_state = WorkflowState.analyzing
		session.status_message = "Partial analysis…"

		def worker() -> None:
			try:
				from deeplabcut.utils import auxiliaryfunctions

				from skellyclicker.services.dlc_paths import (
					analyze_output_folder,
					dlc_project_dir,
					resolve_analyze_iteration,
					resolve_partial_machine_labels_path,
				)

				job.status = JobStatus.running
				self._set_progress(job, 0.0, "Partial analysis started…")
				project_dir = dlc_project_dir(handler.project_config_path)
				cfg = auxiliaryfunctions.read_config(handler.project_config_path)
				analyze_iter = resolve_analyze_iteration(project_dir, cfg)
				handler.iteration = analyze_iter
				session.dlc_iteration = analyze_iter

				output_folder = analyze_output_folder(
					handler.project_config_path,
					use_training_videos,
					video_paths,
					iteration=analyze_iter,
				)
				machine_path = resolve_partial_machine_labels_path(
					handler.project_config_path,
					analyze_iter,
					use_training_videos,
					video_paths,
					session.machine_labels_path,
				)

				def on_progress(fraction: float | None, message: str) -> None:
					self._set_progress(job, fraction, message)

				result_path = handler.partial_analyze_videos(
					human_labels_csv=session.human_labels_path,
					video_paths=video_paths,
					machine_labels_csv=str(machine_path),
					progress_callback=on_progress,
				)
				if use_training_videos:
					session.machine_labels_path = result_path
					session.workflow_state = WorkflowState.review
				job.status = JobStatus.completed
				self._set_progress(job, 1.0, f"Partial analysis complete: {result_path}")
			except Exception as exc:
				job.status = JobStatus.failed
				self._set_progress(job, None, str(exc))
			finally:
				session.active_job_id = None
				if session.workflow_state == WorkflowState.analyzing:
					session.workflow_state = WorkflowState.ready_to_analyze
				session.status_message = job.message

		threading.Thread(target=worker, daemon=True).start()
		return job

	def get_job(self, job_id: str) -> BackgroundJob | None:
		return self._store.jobs.get(job_id)
