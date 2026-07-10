"""Central session store — single source of truth for web API."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from skellyclicker.core.session_validation import (
	bodypart_names_from_csv_columns,
)
from skellyclicker.core.deeplabcut_handler.labeled_data_io import (
	bodyparts_from_labeled_data,
	has_human_labels,
	is_legacy_skellyclicker_csv,
	labeled_data_dir,
	resolve_human_labels_root,
)
from skellyclicker.services.dlc_job_runner import DLCJobRunner
from skellyclicker.services.errors import SessionConflictError, SessionError
from skellyclicker.services.labeling_engine import LabelingEngine
from skellyclicker.services.labeling_mode import detect_labeling_mode
from skellyclicker.services.models import AppSession, BackgroundJob, LabelingMode, WorkflowState
from skellyclicker.services.session_paths import collect_asset_path_checks
from skellyclicker.services.workflow import refresh_workflow_state

from skellyclicker.services.dlc_paths import (
	resolve_dlc_project_input,
)


class SessionStore:
	"""Thread-unsafe singleton for local lab use; one user per server process."""

	def __init__(self) -> None:
		self.session = AppSession()
		self.labeling_engine: LabelingEngine | None = None
		self.dlc_handler: Any | None = None
		self.jobs: dict[str, BackgroundJob] = {}
		self.job_runner = DLCJobRunner(self)
		# Warm DLC runners for scrub-time machine overlays (loaded with DLC project).
		self.live_inference: Any | None = None

	def get_session(self) -> AppSession:
		return self._finalize_session()

	def _sync_machine_labels_path_to_latest(self) -> None:
		"""Validate an existing machine-labels path; never invent one from disk.

		Loaded Assets must stay empty until Full Analysis or Import Machine Labels
		sets ``machine_labels_path``. Do not upgrade/replace an existing path by
		scanning the project tree — that re-surfaces leftover CSVs.
		"""
		current = self.session.machine_labels_path
		if not current:
			return
		# Drop missing files so the UI does not show a dead path.
		if not Path(current).expanduser().is_file():
			self.session.machine_labels_path = None
			return
		if not self.session.dlc_project_path:
			return
		try:
			project_dir, _ = resolve_dlc_project_input(self.session.dlc_project_path)
		except ValueError:
			return
		try:
			Path(current).expanduser().resolve().relative_to(project_dir.resolve())
		except (ValueError, OSError):
			# Path is outside the loaded project (e.g. video-folder leftover).
			self.session.machine_labels_path = None

	def _finalize_session(self) -> AppSession:
		"""Attach fresh path checks and derived workflow state before API responses."""
		self._sync_machine_labels_path_to_latest()
		self.session.asset_path_checks = collect_asset_path_checks(self.session)
		return refresh_workflow_state(self.session)

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
		if self.live_inference is not None:
			self.live_inference.close()
			self.live_inference = None
		self._bump_generation()

	def _close_live_inference(self) -> None:
		"""Drop warm runners (e.g. when switching to an untrained / new project)."""
		if self.live_inference is not None:
			close = getattr(self.live_inference, "close", None)
			if callable(close):
				close()
			self.live_inference = None
		if self.labeling_engine is not None:
			self.labeling_engine.attach_live_inference(None)

	def _ensure_live_inference(self) -> None:
		"""Load warm runners only when the project has trained .pt snapshots."""
		if not self.dlc_handler:
			self._close_live_inference()
			return
		config_path = getattr(self.dlc_handler, "project_config_path", None)
		if not config_path:
			self._close_live_inference()
			return
		from skellyclicker.services.dlc_paths import (
			dlc_project_dir,
			latest_iteration_with_pytorch_model,
		)
		from skellyclicker.services.live_inference import LiveInferenceService

		project_dir = dlc_project_dir(str(config_path))
		# Require real weights — pytorch_config alone exists before train_network.
		if latest_iteration_with_pytorch_model(project_dir) is None:
			self._close_live_inference()
			return

		if self.live_inference is None:
			self.live_inference = LiveInferenceService()
		try:
			self.live_inference.load(str(config_path))
		except Exception as exc:
			# Labeler still works without live scrub; clear so overlays stay off.
			self._close_live_inference()
			self.session.status_message = f"Live machine labels unavailable: {exc}"


	def start_new_session(self) -> AppSession:
		self._assert_no_active_job()
		self._teardown_all()
		self.session = AppSession(workflow_state=WorkflowState.needs_videos)
		return self._finalize_session()

	def clear_session(self) -> AppSession:
		self._assert_no_active_job()
		self._teardown_all()
		self.session = AppSession()
		return self._finalize_session()

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

	def _refresh_labeling_mode(self, paths: list[str]) -> None:
		"""Auto-pick synced vs corpus from frame counts; keep active video if still valid."""
		if not paths:
			self.session.labeling_mode = LabelingMode.single
			self.session.active_video_path = None
			return
		try:
			mode = detect_labeling_mode(paths)
		except ValueError:
			# Unreadable/corrupt files: don't block add/remove; prefer one-at-a-time.
			mode = LabelingMode.corpus if len(paths) > 1 else LabelingMode.single
		self.session.labeling_mode = mode
		if mode == LabelingMode.synced:
			self.session.active_video_path = None
			return
		# Single / corpus: keep current active video when still in the set.
		active = self.session.active_video_path
		if active and active in paths:
			return
		self.session.active_video_path = paths[0]

	def _apply_videos(self, paths: list[str]) -> None:
		"""Store video list and reset frame stats; close labeler if video set changed."""
		if paths != (self.session.videos or []):
			self._teardown_labeler()
			# Changing videos must not keep a previous analyze CSV in Loaded Assets.
			# Full Analysis / Import Machine Labels set the path when a file is produced.
			self.session.machine_labels_path = None
		self.session.videos = paths if paths else None
		self.session.frame_count = 0
		self._refresh_labeling_mode(paths)
		if paths:
			self.session.workflow_state = WorkflowState.ready_to_label
			mode = self.session.labeling_mode.value
			self.session.status_message = f"{len(paths)} video(s) selected ({mode})"
		else:
			self.session.status_message = "No videos selected"

	def set_videos(self, paths: list[str]) -> AppSession:
		self._assert_no_active_job()
		self._apply_videos(self._validate_video_paths(paths))
		return self._finalize_session()

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
		return self._finalize_session()

	def remove_video(self, path: str) -> AppSession:
		"""Drop one video from the session list."""
		self._assert_no_active_job()
		resolved = str(Path(path).expanduser().resolve())
		existing = list(self.session.videos or [])
		if resolved not in existing:
			raise SessionError(f"Video not in session: {path}")
		existing.remove(resolved)
		self._apply_videos(existing)
		return self._finalize_session()

	def set_human_labels_path(self, path: str) -> AppSession:
		self._assert_no_active_job()
		raw = Path(path).expanduser()
		if not raw.exists():
			raise SessionError(f"Human labels path not found: {path}")
		self._teardown_labeler()

		# Legacy flat CSV → convert once into the loaded project's labeled-data.
		if raw.is_file() and is_legacy_skellyclicker_csv(raw):
			if not self.session.dlc_project_path:
				raise SessionError(
					"Create or load a DLC project before importing a legacy labels CSV"
				)
			if not self.session.videos:
				raise SessionError("Select videos before importing human labels")
			from skellyclicker.core.deeplabcut_handler.create_deeplabcut.create_deeplabcut_project_data import (
				fill_in_labelled_data_folder,
			)

			project_dir, _ = resolve_dlc_project_input(self.session.dlc_project_path)
			fill_in_labelled_data_folder(
				path_to_videos_for_training=str(Path(self.session.videos[0]).parent),
				path_to_dlc_project_folder=str(project_dir),
				path_to_image_labels_csv=str(raw.resolve()),
				video_paths=list(self.session.videos),
			)
			root = labeled_data_dir(project_dir)
			self.session.human_labels_path = str(root)
			bodyparts = bodyparts_from_labeled_data(root)
			if bodyparts:
				self.session.tracked_point_names = bodyparts
			self.session.status_message = (
				f"Imported legacy labels into {root}"
			)
			return self._finalize_session()

		try:
			root = resolve_human_labels_root(raw)
		except ValueError as exc:
			raise SessionError(str(exc)) from exc
		self.session.human_labels_path = str(root)
		bodyparts = bodyparts_from_labeled_data(root)
		if bodyparts:
			self.session.tracked_point_names = bodyparts
		elif raw.is_file():
			# Fallback for unexpected flat CSV that wasn't detected as legacy.
			import pandas as pd

			df = pd.read_csv(raw)
			self.session.tracked_point_names = bodypart_names_from_csv_columns(
				list(df.columns)
			)
		if self.session.videos and root.is_dir():
			# Soft validation: warn when no CollectedData matches session videos.
			if not has_human_labels(root):
				self.session.status_message = (
					f"labeled-data has no CollectedData yet: {root}"
				)
		return self._finalize_session()

	def _labeled_data_save_path(self) -> str:
		"""Resolve the DLC labeled-data directory for human-label saves."""
		if self.session.human_labels_path:
			try:
				return str(resolve_human_labels_root(self.session.human_labels_path))
			except ValueError:
				pass
		if not self.session.dlc_project_path:
			raise SessionError(
				"Create or load a DLC project before saving human labels"
			)
		try:
			project_dir, _ = resolve_dlc_project_input(self.session.dlc_project_path)
		except ValueError as exc:
			raise SessionError(str(exc)) from exc
		return str(labeled_data_dir(project_dir))

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

	@staticmethod
	def _validate_training_int(name: str, value: int, *, minimum: int = 1, maximum: int = 1000) -> int:
		"""Match legacy Tk spinbox bounds (1–1000) for training hyperparameters."""
		if value < minimum:
			raise SessionError(f"{name} must be at least {minimum}")
		if value > maximum:
			raise SessionError(f"{name} must be at most {maximum}")
		return value

	def set_training_settings(
		self,
		*,
		epochs: int | None = None,
		save_epochs: int | None = None,
		batch_size: int | None = None,
	) -> AppSession:
		"""Update DLC training hyperparameters stored on the session."""
		if epochs is not None:
			self.session.training_epochs = self._validate_training_int("Epochs", epochs)
		if save_epochs is not None:
			self.session.training_save_epochs = self._validate_training_int(
				"Save epochs", save_epochs
			)
		if batch_size is not None:
			self.session.training_batch_size = self._validate_training_int(
				"Batch size", batch_size
			)
		return self.session

	def set_analyze_options(
		self,
		*,
		filter_predictions: bool | None = None,
		annotate_videos: bool | None = None,
	) -> AppSession:
		"""Update analyze-time options (filter CSV, annotated output videos)."""
		if filter_predictions is not None:
			self.session.filter_predictions = filter_predictions
		if annotate_videos is not None:
			self.session.annotate_videos = annotate_videos
		return self.session

	def _can_open_labeler(self) -> bool:
		"""Labeler needs videos plus imported labels or a DLC project with bodyparts."""
		if not self.session.videos:
			return False
		if self.session.human_labels_path or self.session.machine_labels_path:
			return True
		return bool(
			self.session.dlc_project_path and self.session.tracked_point_names
		)

	def _labeler_video_paths(self) -> list[str]:
		"""Videos to open in the labeler: all (synced) or one active (single/corpus)."""
		videos = list(self.session.videos or [])
		if not videos:
			raise SessionError("Select videos first")
		# Re-detect in case session JSON was loaded with a stale mode.
		self._refresh_labeling_mode(videos)
		# Hard rule until synced multi-cam is an explicit opt-in: never open a
		# multi-video grid. Training-corpus sessions always label one at a time.
		if len(videos) > 1:
			self.session.labeling_mode = LabelingMode.corpus
			active = self.session.active_video_path or videos[0]
			if active not in videos:
				active = videos[0]
			self.session.active_video_path = active
			return [active]
		if self.session.labeling_mode == LabelingMode.synced:
			return videos
		active = self.session.active_video_path or videos[0]
		if active not in videos:
			raise SessionError(f"Active video not in session: {active}")
		self.session.active_video_path = active
		return [active]

	def _open_labeling_engine(self, open_paths: list[str]) -> LabelingEngine:
		"""Construct LabelingEngine for the current session mode/paths."""
		return LabelingEngine.open(
			video_paths=open_paths,
			human_labels_path=self.session.human_labels_path,
			machine_labels_path=self.session.machine_labels_path,
			# Web labeler always edits human labels; machine CSV is overlay-only.
			train_on_machine_labels=False,
			tracked_point_names=list(self.session.tracked_point_names),
			labeling_mode=self.session.labeling_mode,
			session_video_paths=list(self.session.videos or []),
			active_video_path=self.session.active_video_path,
		)

	def _open_labeling_engine_as_corpus(self) -> tuple[LabelingEngine, list[str]]:
		"""Force one-video corpus open after a synced/multi-path failure."""
		videos = list(self.session.videos or [])
		active = self.session.active_video_path or videos[0]
		if active not in videos:
			active = videos[0]
		self.session.labeling_mode = LabelingMode.corpus
		self.session.active_video_path = active
		open_paths = [active]
		engine = LabelingEngine.open(
			video_paths=open_paths,
			human_labels_path=self.session.human_labels_path,
			machine_labels_path=self.session.machine_labels_path,
			train_on_machine_labels=False,
			tracked_point_names=list(self.session.tracked_point_names),
			labeling_mode=LabelingMode.corpus,
			session_video_paths=videos,
			active_video_path=active,
		)
		return engine, open_paths

	def open_labeler(self) -> AppSession:
		self._assert_no_active_job()
		if not self.session.videos:
			raise SessionError("Select videos first")
		if not self._can_open_labeler():
			raise SessionError(
				"Import Human or Machine labels, or load/create a DLC project "
				"to define bodyparts before opening the labeler."
			)
		self._teardown_labeler()
		open_paths = self._labeler_video_paths()
		# Best-effort warm model for on-the-fly scrub predictions (corpus/single).
		self._ensure_live_inference()
		try:
			engine = self._open_labeling_engine(open_paths)
		except Exception as exc:
			# Never leave the UI with an unhandled 500 for multi-video opens.
			# Synced mis-detect (lying CAP_PROP / mixed folders) is the usual cause.
			msg = str(exc)
			can_fallback = len(self.session.videos or []) > 1 and len(open_paths) > 1
			if can_fallback:
				try:
					engine, open_paths = self._open_labeling_engine_as_corpus()
				except Exception as fallback_exc:
					raise SessionError(
						f"Could not open labeler: {fallback_exc}"
					) from fallback_exc
			else:
				raise SessionError(f"Could not open labeler: {msg}") from exc
		if self.live_inference is not None and self.live_inference.ready:
			engine.attach_live_inference(self.live_inference)
		self.labeling_engine = engine
		self.session.labeling_session_id = engine.session_id
		self.session.frame_count = engine.video_handler.frame_count
		self.session.tracked_point_names = (
			engine.video_handler.data_handler.config.tracked_point_names
		)
		self.session.workflow_state = WorkflowState.labeling
		mode = self.session.labeling_mode
		live = (
			self.live_inference is not None and self.live_inference.ready
		)
		live_note = " · live predictions" if live else ""
		if mode == LabelingMode.synced:
			self.session.status_message = (
				f"Labeling ({len(open_paths)} cameras · shared timeline{live_note})"
			)
		elif mode == LabelingMode.corpus:
			name = Path(open_paths[0]).name
			self.session.status_message = (
				f"Labeling ({len(self.session.videos)} videos · {name}{live_note})"
			)
		else:
			self.session.status_message = f"Labeling{live_note}"
		return self._finalize_session()

	def set_active_labeling_video(self, video_path: str) -> AppSession:
		"""Switch the corpus/single labeler to another session video (merge-save first)."""
		if not self.labeling_engine:
			raise SessionError("Labeler is not open")
		if self.session.labeling_mode == LabelingMode.synced:
			raise SessionError("Active video selection is only for per-video labeling")
		videos = list(self.session.videos or [])
		resolved = str(Path(video_path).expanduser().resolve())
		if resolved not in videos:
			# Also accept basename match for convenience from the UI.
			by_name = {Path(p).name: p for p in videos}
			resolved = by_name.get(Path(video_path).name, "")
			if not resolved:
				raise SessionError(f"Video not in session: {video_path}")
		if resolved == self.session.active_video_path:
			return self.session
		# Persist current video's labels into labeled-data before switching.
		if self.session.human_labels_path or self.session.dlc_project_path:
			self.save_labeler(None)
		else:
			raise SessionError(
				"Create or load a DLC project before switching videos (labels must be saved)"
			)
		self.session.active_video_path = resolved
		self._teardown_labeler()
		return self.open_labeler()

	def _labeled_frame_count(self, engine) -> int:
		handler = engine.video_handler.data_handler
		mask = handler.dataframe.notna().any(axis=1)
		if mask.any():
			return int(
				handler.dataframe.index[mask].get_level_values("frame").nunique()
			)
		return 0

	def _assert_human_label_save_path(self, save_path: str | None) -> None:
		"""Block writes that would overwrite the machine-labels CSV from the labeler."""
		if not save_path or not self.session.machine_labels_path:
			return
		try:
			if Path(save_path).resolve() == Path(self.session.machine_labels_path).resolve():
				raise SessionError(
					"Cannot save human labels to the machine labels file. "
					"Human labels are stored in the DLC project labeled-data folder."
				)
		except OSError:
			return

	def save_labeler(self, save_path: str | None = None) -> AppSession:
		if not self.labeling_engine:
			raise SessionError("Labeler is not open")
		# Always write to project labeled-data — ignore client-picked skellyclicker paths.
		target = self._labeled_data_save_path()
		self._assert_human_label_save_path(target)
		engine = self.labeling_engine
		handler = engine.video_handler.data_handler
		tracked_names = list(handler.config.tracked_point_names)
		try:
			path = engine.save_labels(target)
		except ValueError as exc:
			raise SessionError(str(exc)) from exc
		if not path:
			raise SessionError("Could not save labels to labeled-data.")
		self.session.human_labels_path = path
		self.session.tracked_point_names = tracked_names
		self.session.labeled_frame_count = self._labeled_frame_count(engine)
		self.session.status_message = f"Labels saved to {path}"
		return self._finalize_session()

	def close_labeler(self, save: bool, save_path: str | None = None) -> AppSession:
		if not self.labeling_engine:
			raise SessionError("Labeler is not open")
		target = self._labeled_data_save_path() if save else None
		if save:
			self._assert_human_label_save_path(target)
		engine = self.labeling_engine
		labeling_id = engine.session_id
		handler = engine.video_handler.data_handler
		tracked_names = list(handler.config.tracked_point_names)
		labeled_count = self._labeled_frame_count(engine)
		try:
			path = engine.close(save=save, save_path=target)
		except ValueError as exc:
			raise SessionError(str(exc)) from exc
		if self.session.labeling_session_id != labeling_id:
			if save and path:
				raise SessionError(
					f"Labels were saved to {path} but the session changed. "
					"Use Import Human Labels with that path."
				)
			return self.session
		if save:
			if not path:
				raise SessionError(
					"Could not save labels to labeled-data. Try closing the labeler again."
				)
			self.session.human_labels_path = path
			self.session.tracked_point_names = tracked_names
			self.session.status_message = f"Labels saved to {path}"
		else:
			self.session.status_message = "Labeling closed without saving"
		self.labeling_engine = None
		self.session.labeling_session_id = None
		self.session.labeled_frame_count = labeled_count
		return self._finalize_session()

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
		# Reuse existing DLC human labels when present (single source of truth).
		labeled_root = labeled_data_dir(project_dir)
		if has_human_labels(labeled_root):
			self.session.human_labels_path = str(labeled_root)
			bodyparts = bodyparts_from_labeled_data(labeled_root)
			if bodyparts and not self.session.tracked_point_names:
				self.session.tracked_point_names = bodyparts
		# Loading a project must not keep another project's machine CSV in Loaded Assets.
		# Full Analysis / Import Machine Labels set this path when needed.
		self.session.machine_labels_path = None
		# Warm live runners only if this project has trained weights; else clear.
		self._ensure_live_inference()
		return self._finalize_session()

	def _resolve_session_json_path(self, path: str) -> Path:
		"""Normalize session JSON path; bare filenames go under ~/skellyclicker_sessions/."""
		raw = path.strip()
		if not raw:
			raise SessionError("Session path is empty")
		resolved = Path(raw).expanduser()
		# A filename alone has no parent dir — avoid writing into the server's cwd.
		if resolved.parent == Path("."):
			resolved = Path.home() / "skellyclicker_sessions" / resolved.name
		if resolved.suffix.lower() != ".json":
			raise SessionError("Session path must end with .json")
		if resolved.exists() and resolved.is_dir():
			raise SessionError(f"Session path is a directory, not a file: {resolved}")
		return resolved

	def save_session_json(self, path: str) -> AppSession:
		# Save creates or overwrites — never requires the file to exist beforehand.
		target = self._resolve_session_json_path(path)
		try:
			target.parent.mkdir(parents=True, exist_ok=True)
			target.write_text(
				self.session.model_dump_json(indent=2, exclude={"asset_path_checks"})
			)
		except OSError as exc:
			raise SessionError(
				f"Could not write session file: {target.resolve()}. {exc}"
			) from exc
		self.session.session_saved_path = str(target.resolve())
		return self.session

	def load_session_json(self, path: str) -> AppSession:
		self._assert_no_active_job()
		self._teardown_all()
		import json
		target = self._resolve_session_json_path(path)
		if not target.is_file():
			raise SessionError(
				f"Session file not found: {target.resolve()}. "
				"Use Save Session first to create it, or check the full path."
			)
		data = json.loads(target.read_text())
		self.session = AppSession.model_validate(data)
		self.session.session_saved_path = str(target.resolve())
		# Do not restore machine CSV into Loaded Assets from an old session file —
		# Full Analysis / Import Machine Labels set this path when a file is produced.
		self.session.machine_labels_path = None
		# Re-detect mode from current video files (stale session JSON is OK).
		if self.session.videos:
			try:
				self._refresh_labeling_mode(list(self.session.videos))
			except SessionError:
				pass
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
				self._ensure_live_inference()
		session = self._finalize_session()
		missing = [c.path for c in session.asset_path_checks if not c.exists]
		if missing:
			session.status_message = (
				f"Session loaded; {len(missing)} referenced path(s) not found on disk"
			)
		return session


# Module-level singleton used by FastAPI routes.
store = SessionStore()
