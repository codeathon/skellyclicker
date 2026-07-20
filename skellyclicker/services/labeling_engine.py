"""Headless labeling session — wraps VideoHandler without OpenCV windows."""

import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from skellyclicker import (
	LABELER_JPEG_QUALITY_COMMITTED,
	LABELER_JPEG_QUALITY_PREVIEW,
)
from skellyclicker.core.video_handler.image_annotator import get_colors_for_css
from skellyclicker.core.video_handler.video_handler import VideoHandler
from skellyclicker.core.deeplabcut_handler.labeled_data_io import (
	resolve_human_labels_root,
	write_labeled_data_from_wide,
)
from skellyclicker.services.label_nav_frames import build_nav_frame_list
from skellyclicker.services.models import LabelingMode
from skellyclicker.services.live_inference import LiveInferenceService


class LabelingEngine(BaseModel):
	"""Server-side labeler for one open labeling session."""
	model_config = ConfigDict(arbitrary_types_allowed=True)

	session_id: str = Field(default_factory=lambda: str(uuid4()))
	# Any so unit tests can inject MagicMock; runtime always uses VideoHandler.
	video_handler: Any
	frame_number: int = 0
	auto_next_point: bool = True
	show_machine_labels: bool = False
	show_help: bool = False
	show_names: bool = True
	# Synced grid vs single-video corpus — drives save merge and state_dict fields.
	labeling_mode: LabelingMode = LabelingMode.single
	# Full session video list (absolute paths); used for corpus video selector.
	session_video_paths: list[str] = Field(default_factory=list)
	active_video_path: str | None = None
	# Path of the corpus human CSV before this open — merge other videos on save.
	_corpus_labels_path: str | None = PrivateAttr(default=None)
	# OpenCV VideoCapture is not thread-safe; scrub previews hit this concurrently.
	_render_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
	_undo_stack: list[dict[str, Any]] = PrivateAttr(default_factory=list)
	# Snapshot at frame navigation — machine overlay must not flip us into review mode mid-click.
	_frame_had_human_on_entry: bool = PrivateAttr(default=False)
	# Optional warm DLC runners for scrub-time predictions (corpus/single).
	_live_inference: Any | None = PrivateAttr(default=None)

	@classmethod
	def open(
		cls,
		video_paths: list[str],
		human_labels_path: str | None,
		machine_labels_path: str | None,
		train_on_machine_labels: bool,
		tracked_point_names: list[str],
		*,
		labeling_mode: LabelingMode = LabelingMode.single,
		session_video_paths: list[str] | None = None,
		active_video_path: str | None = None,
	) -> "LabelingEngine":
		# Web labeler: human CSV is always editable; machine CSV is read-only overlay.
		# train_on_machine_labels is ignored when opening from the web API.
		primary_csv: str | None = None
		overlay: str | None = None
		bodyparts: list[str] | None = None

		if train_on_machine_labels and machine_labels_path:
			primary_csv = machine_labels_path
		elif human_labels_path:
			primary_csv = human_labels_path
			overlay = machine_labels_path
		else:
			bodyparts = tracked_point_names
			overlay = machine_labels_path

		handler = VideoHandler.from_videos_for_labeler(
			video_paths=video_paths,
			data_handler_path=primary_csv,
			tracked_point_names=list(tracked_point_names) if tracked_point_names else None,
			machine_labels_path=overlay,
		)
		session_paths = list(session_video_paths or video_paths)
		active = active_video_path
		if labeling_mode != LabelingMode.synced and not active:
			active = video_paths[0] if video_paths else None
		engine = cls(
			video_handler=handler,
			labeling_mode=labeling_mode,
			session_video_paths=session_paths,
			active_video_path=active,
		)
		# Remember corpus CSV so single-video saves merge other experiments' rows.
		if labeling_mode != LabelingMode.synced and human_labels_path:
			engine._corpus_labels_path = human_labels_path
		annotator_cfg = engine.video_handler.image_annotator.config
		annotator_cfg.web_help = True
		annotator_cfg.external_hud = True
		annotator_cfg.show_clicks = False
		engine.sync_active_point()
		engine._frame_had_human_on_entry = engine._frame_has_human_labels()
		engine._sync_auto_next_point_for_frame()
		engine._sync_annotator_overlay_flags()
		return engine

	def set_frame(self, frame_number: int) -> None:
		"""Commit frame navigation and refresh per-frame labeler behavior."""
		from skellyclicker.services.errors import SessionError

		target = int(frame_number)
		# Once the user starts labeling a frame, require every bodypart before leaving.
		if target != self.frame_number:
			block = self.video_handler.data_handler.incomplete_labeling_message(
				self.frame_number
			)
			if block:
				raise SessionError(block)
		self.frame_number = target
		self._frame_had_human_on_entry = self._frame_has_human_labels()
		self._sync_auto_next_point_for_frame()
		if self.auto_next_point:
			self.video_handler.data_handler.reset_active_point_for_frame(target)
		# Prefetch display-only live prediction so the stopped scrub frame can
		# guide human labeling (live overlay is never saved).
		self._maybe_request_live_infer(target)

	def _frame_has_human_labels(self) -> bool:
		"""True when any human label exists on this frame (any camera)."""
		dh = self.video_handler.data_handler
		for video_index in range(len(dh.config.video_names)):
			if dh.get_data_by_video_frame(video_index, self.frame_number):
				return True
		return False

	def _frame_has_machine_labels(self) -> bool:
		"""True when any machine prediction exists on this frame (any camera)."""
		handler = self.video_handler
		if not handler.machine_labels_path:
			return False
		handler.ensure_machine_labels_loaded()
		mlh = handler.machine_labels_handler
		if mlh is None:
			return False
		for video_index in range(len(mlh.config.video_names)):
			if mlh.get_data_by_video_frame(video_index, self.frame_number):
				return True
		return False

	def _sync_auto_next_point_for_frame(self) -> None:
		"""Auto-advance on fresh frames; manual pick when reviewing existing human labels."""
		# Machine-only frames still auto-advance while the user places human labels.
		self.auto_next_point = not self._frame_had_human_on_entry

	def sync_active_point(self) -> None:
		"""Pick first unlabeled bodypart on session open only (not on every frame change)."""
		self.video_handler.data_handler.reset_active_point_for_frame(
			self.frame_number,
		)

	def save_labels(self, save_path: str | None = None) -> str:
		"""Write human labels to DLC labeled-data without closing the labeler."""
		# Synced / multi-open: write all open videos. Corpus: only the active video
		# folder so sibling experiment folders stay intact.
		if self.labeling_mode == LabelingMode.synced or not self.active_video_path:
			return self.video_handler.save_labels(save_path)

		if save_path is None:
			raise ValueError(
				"Create or load a DLC project before saving human labels"
			)
		save_path_p = Path(save_path).expanduser().resolve()
		if save_path_p.name == "labeled-data":
			root = save_path_p
		elif save_path_p.is_dir() and (save_path_p / "labeled-data").exists():
			root = save_path_p / "labeled-data"
		elif save_path_p.name != "labeled-data" and save_path_p.suffix == "":
			root = save_path_p / "labeled-data"
		else:
			try:
				root = resolve_human_labels_root(save_path_p)
			except ValueError as exc:
				raise ValueError(
					"Human labels must be saved to the DLC project labeled-data folder"
				) from exc
		root.mkdir(parents=True, exist_ok=True)

		handler = self.video_handler
		mask = handler.data_handler.dataframe.notna().any(axis=1)
		new_rows = handler.data_handler.dataframe.loc[mask].reset_index()
		active_name = Path(self.active_video_path).name
		# Absolute paths for every session video so other folders are not touched.
		video_paths = list(self.session_video_paths) or [
			str(p) for p in handler.videos.keys()
		]
		write_labeled_data_from_wide(
			labeled_data_root=root,
			wide_df=new_rows,
			video_paths=video_paths,
			joint_names=list(handler.data_handler.config.tracked_point_names),
			only_videos=[active_name],
		)
		self._corpus_labels_path = str(root)
		return str(root)

	def set_active_point(self, point_name: str) -> None:
		self.video_handler.data_handler.set_active_point_by_name(point_name)

	def _sync_annotator_overlay_flags(self) -> None:
		"""Apply web labeler overlay toggles to human and machine annotators."""
		self.video_handler.image_annotator.config.show_help = self.show_help
		self.video_handler.image_annotator.config.show_names = self.show_names
		machine_annotator = self.video_handler.machine_labels_annotator
		if machine_annotator is not None:
			machine_annotator.config.show_names = self.show_names

	def toggle_show_names(self) -> None:
		self.show_names = not self.show_names
		self._sync_annotator_overlay_flags()

	def attach_live_inference(self, service: LiveInferenceService | None) -> None:
		"""Wire session-scoped warm runners for display-only scrub overlays (never saved)."""
		self._live_inference = service
		if service is None:
			self.video_handler.live_points_lookup = None
			return
		# Sticky-aware lookup for scrub; exact frame when sticky is off.
		self.video_handler.live_points_lookup = service.get_overlay_points
		# Live guide starts visible; m toggles both live and CSV machine overlays.
		if service.ready:
			self.show_machine_labels = True

	def _live_infer_video_path(self) -> str | None:
		"""Active video path for live scrub infer (corpus/single)."""
		if self.labeling_mode == LabelingMode.synced:
			return None
		if self.active_video_path:
			return self.active_video_path
		paths = list(self.video_handler.videos.keys())
		return paths[0] if paths else None

	def _maybe_request_live_infer(self, frame_number: int) -> None:
		"""Kick coalesced background infer for the active video when CSV has no row."""
		# Respect m toggle — do not burn GPU while machine overlay is hidden.
		if not self.show_machine_labels:
			return
		service = self._live_inference
		if service is None or not service.ready:
			return
		video_path = self._live_infer_video_path()
		if not video_path:
			return
		video_name = Path(video_path).name
		if service.get_cached(video_name, frame_number) is not None:
			return
		# If saved machine CSV already has this frame, skip live infer.
		self.video_handler.ensure_machine_labels_loaded()
		mlh = self.video_handler.machine_labels_handler
		if mlh is not None and mlh.get_data_by_video_frame(0, frame_number):
			return
		service.request_infer(video_path, frame_number)

	def _wait_briefly_for_live_cache(self, frame_number: int, *, timeout_s: float = 0.18) -> None:
		"""On committed frames, wait briefly so the first paint can include live crosses."""
		import time

		if not self.show_machine_labels:
			return
		service = self._live_inference
		video_path = self._live_infer_video_path()
		if service is None or not service.ready or not video_path:
			return
		video_name = Path(video_path).name
		if service.get_cached(video_name, frame_number) is not None:
			return
		deadline = time.monotonic() + timeout_s
		while time.monotonic() < deadline:
			if service.get_cached(video_name, frame_number) is not None:
				return
			time.sleep(0.01)

	def render_frame_jpeg(
		self,
		frame_number: int | None = None,
		*,
		preview: bool = False,
	) -> bytes:
		"""Render grid as JPEG. Preview mode is for smooth scrubbing (lighter, machine overlay)."""
		with self._render_lock:
			render_at = self.frame_number if frame_number is None else frame_number
			# Preview must not move the committed frame — only POST /labeling/frame does that.
			if frame_number is not None and not preview and frame_number != self.frame_number:
				self.set_frame(frame_number)
			elif frame_number is not None and not preview:
				render_at = frame_number

			prev_show = self.video_handler.show_machine_labels
			prev_help = self.video_handler.image_annotator.config.show_help
			prev_names = self.video_handler.image_annotator.config.show_names
			prev_sticky = self.video_handler.live_overlay_sticky
			try:
				live = self._live_inference
				has_csv = bool(self.video_handler.machine_labels_path)
				has_live = bool(live and live.ready)
				# m toggles both saved CSV overlays and live on-the-fly predictions.
				want_machine = self.show_machine_labels and (has_csv or has_live)
				if preview:
					# Scrub: sticky last/nearby live points so crosses stay visible
					# while coalesced infer lags behind a fast drag.
					self.video_handler.show_machine_labels = want_machine
					self.video_handler.live_overlay_sticky = want_machine and has_live
					if want_machine and has_csv:
						self.video_handler.ensure_machine_labels_loaded()
					if want_machine and has_live:
						self._maybe_request_live_infer(render_at)
				else:
					# Stopped/committed frame: prefer exact-frame cache (not sticky).
					self.video_handler.live_overlay_sticky = False
					self.video_handler.show_machine_labels = want_machine
					if want_machine and has_csv:
						self.video_handler.ensure_machine_labels_loaded()
					if want_machine and has_live:
						self._maybe_request_live_infer(render_at)
						# Brief wait so release-scrub paint often includes the guide.
						self._wait_briefly_for_live_cache(render_at)
				self._sync_annotator_overlay_flags()
				image = self.video_handler.create_grid_image(
					render_at,
					annotate_images=not preview,
					preview=preview,
				)
			finally:
				# Always restore — live preview must not leave overlay permanently on.
				self.video_handler.show_machine_labels = prev_show
				self.video_handler.live_overlay_sticky = prev_sticky
				self.video_handler.image_annotator.config.show_help = prev_help
				self.video_handler.image_annotator.config.show_names = prev_names
				if self.video_handler.machine_labels_annotator is not None:
					self.video_handler.machine_labels_annotator.config.show_names = prev_names

			quality = (
				LABELER_JPEG_QUALITY_PREVIEW if preview else LABELER_JPEG_QUALITY_COMMITTED
			)
			ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
			if not ok:
				raise RuntimeError("Failed to encode frame as JPEG")
			return encoded.tobytes()

	def handle_click(self, x: int, y: int) -> None:
		click_data = self.video_handler.click_handler.process_click(
			x, y, self.frame_number
		)
		if click_data is None:
			return
		dh = self.video_handler.data_handler
		point_name = dh.active_point
		prev_x, prev_y = dh.get_point_coords(
			click_data.video_index,
			click_data.frame_number,
			point_name,
		)
		dh.update_dataframe(click_data, point_name=point_name)
		self._undo_stack.append(
			{
				"frame_number": click_data.frame_number,
				"video_index": click_data.video_index,
				"point_name": point_name,
				"prev_x": prev_x,
				"prev_y": prev_y,
			}
		)
		if self.auto_next_point:
			dh.move_active_point_by_index(index_change=1)

	def undo_last_label(self) -> bool:
		"""Revert the most recent label placement."""
		if not self._undo_stack:
			return False
		entry = self._undo_stack.pop()
		dh = self.video_handler.data_handler
		if entry["prev_x"] is None:
			dh.clear_point(
				entry["video_index"],
				entry["frame_number"],
				entry["point_name"],
			)
		else:
			dh.set_point_coords(
				entry["video_index"],
				entry["frame_number"],
				entry["point_name"],
				entry["prev_x"],
				entry["prev_y"],
			)
		self.frame_number = int(entry["frame_number"])
		dh.set_active_point_by_name(str(entry["point_name"]))
		return True

	def clear_active_point_on_frame(self) -> bool:
		"""Remove the active bodypart on the current frame (all camera views)."""
		dh = self.video_handler.data_handler
		cleared = False
		for video_index in range(len(dh.config.video_names)):
			if dh.point_is_labeled(video_index, self.frame_number, dh.active_point):
				dh.clear_point(video_index, self.frame_number, dh.active_point)
				cleared = True
		return cleared

	def undo(self) -> bool:
		"""Undo last placement, or clear active bodypart on this frame."""
		if self._undo_stack:
			return self.undo_last_label()
		return self.clear_active_point_on_frame()

	def _frame_label_status(self) -> tuple[list[str], list[str]]:
		"""Bodyparts placed vs still available on the primary video for the current frame."""
		dh = self.video_handler.data_handler
		tracked = dh.config.tracked_point_names
		click_data = dh.get_data_by_video_frame(0, self.frame_number)
		placed = [name for name in tracked if name in click_data]
		available = [name for name in tracked if name not in click_data]
		return placed, available

	def _nav_video_name(self) -> str | None:
		"""Basename used to scope left-panel nav; None = synced grid (all open videos)."""
		if self.labeling_mode == LabelingMode.synced:
			return None
		if self.active_video_path:
			return Path(self.active_video_path).name
		names = self.video_handler.data_handler.config.video_names
		return names[0] if names else None

	def state_dict(self) -> dict:
		handler = self.video_handler
		active = handler.data_handler.active_point
		# Corpus / single: left panel shows only the video selected in the right HUD.
		nav_video = self._nav_video_name()
		labeled_frame_list = handler.data_handler.get_nonempty_frames(nav_video)
		labeled = len(labeled_frame_list)
		# Human frames only — predictions are reviewed via live scrub / Full Analysis.
		nav_frame_list = build_nav_frame_list(labeled_frame_list)
		placed_points, available_points = self._frame_label_status()
		tracked = handler.data_handler.config.tracked_point_names
		point_colors = {
			name: list(rgb)
			for name, rgb in get_colors_for_css(tracked).items()
		}
		return {
			"session_id": self.session_id,
			"frame_number": self.frame_number,
			"frame_count": handler.frame_count,
			"active_point": active,
			"tracked_points": tracked,
			"point_colors": point_colors,
			"placed_points": placed_points,
			"available_points": available_points,
			"labeled_frames": labeled,
			"labeled_frame_list": labeled_frame_list,
			"nav_frame_list": nav_frame_list,
			"show_machine_labels": self.show_machine_labels,
			"show_help": self.show_help,
			"show_names": self.show_names,
			"has_machine_labels": bool(
				handler.machine_labels_path
				or (self._live_inference is not None and self._live_inference.ready)
			),
			"live_inference_ready": bool(
				self._live_inference is not None and self._live_inference.ready
			),
			"auto_next_point": self.auto_next_point,
			"grid_width": handler.grid_parameters.total_width,
			"grid_height": handler.grid_parameters.total_height,
			"labeling_mode": self.labeling_mode.value,
			"session_videos": [
				{"path": p, "name": Path(p).name} for p in self.session_video_paths
			],
			"active_video_path": self.active_video_path,
			"active_video_name": (
				Path(self.active_video_path).name if self.active_video_path else None
			),
		}

	def close(self, save: bool, save_path: str | None = None) -> str | None:
		# Corpus mode must merge via save_labels before releasing captures.
		if save and self.labeling_mode != LabelingMode.synced and self.active_video_path:
			path = self.save_labels(save_path)
			self.video_handler.close(save_data=False)
			return path
		return self.video_handler.close(save_data=save, save_path=save_path)
