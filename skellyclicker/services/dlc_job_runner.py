"""Background DLC train/analyze jobs with WebSocket-friendly status updates."""

import threading

from skellyclicker.services.models import BackgroundJob, JobStatus, WorkflowState


class DLCJobRunner:
	"""Runs train/analyze in daemon threads; snapshots handler at job start."""

	def __init__(self, session_store) -> None:
		self._store = session_store

	def _append_log(self, job: BackgroundJob, line: str) -> None:
		job.log_lines.append(line)
		job.message = line

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
				self._append_log(job, "Training started")
				handler.train_model(
					labels_csv_path=csv_path,
					video_paths=videos,
					training_config=config,
				)
				session.dlc_iteration = handler.iteration
				job.status = JobStatus.completed
				self._append_log(job, f"Training complete (iteration {handler.iteration})")
			except Exception as exc:
				job.status = JobStatus.failed
				self._append_log(job, str(exc))
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
		from skellyclicker.services.dlc_paths import analyze_output_folder

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
				job.status = JobStatus.running
				self._append_log(job, "Analysis started")
				# Anchor output to loaded config.yaml's directory, not yaml project_path field.
				output_folder = analyze_output_folder(
					handler.project_config_path,
					use_training_videos,
					video_paths,
					iteration=handler.iteration,
				)
				self._append_log(job, f"Output folder: {output_folder}")
				machine_path = handler.analyze_videos(
					video_paths=video_paths,
					annotate_videos=session.annotate_videos,
					filter_videos=session.filter_predictions,
					output_folder=output_folder,
				)
				if use_training_videos:
					session.machine_labels_path = machine_path
					session.workflow_state = WorkflowState.review
				job.status = JobStatus.completed
				self._append_log(job, f"Analysis complete: {machine_path}")
			except Exception as exc:
				job.status = JobStatus.failed
				self._append_log(job, str(exc))
			finally:
				session.active_job_id = None
				if session.workflow_state == WorkflowState.analyzing:
					session.workflow_state = WorkflowState.ready_to_analyze
				session.status_message = job.message

		threading.Thread(target=worker, daemon=True).start()
		return job

	def get_job(self, job_id: str) -> BackgroundJob | None:
		return self._store.jobs.get(job_id)
