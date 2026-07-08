"""Headless labeling session — wraps VideoHandler without OpenCV windows."""

import threading
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
from skellyclicker.services.label_nav_frames import build_nav_frame_list


class LabelingEngine(BaseModel):
	"""Server-side labeler for one open labeling session."""
	model_config = ConfigDict(arbitrary_types_allowed=True)

	session_id: str = Field(default_factory=lambda: str(uuid4()))
	video_handler: VideoHandler
	frame_number: int = 0
	auto_next_point: bool = True
	show_machine_labels: bool = False
	show_help: bool = False
	show_names: bool = True
	# OpenCV VideoCapture is not thread-safe; scrub previews hit this concurrently.
	_render_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
	_undo_stack: list[dict[str, Any]] = PrivateAttr(default_factory=list)
	# Snapshot at frame navigation — machine overlay must not flip us into review mode mid-click.
	_frame_had_human_on_entry: bool = PrivateAttr(default=False)

	@classmethod
	def open(
		cls,
		video_paths: list[str],
		human_labels_path: str | None,
		machine_labels_path: str | None,
		train_on_machine_labels: bool,
		tracked_point_names: list[str],
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
		engine = cls(
			video_handler=handler,
		)
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
		self.frame_number = frame_number
		self._frame_had_human_on_entry = self._frame_has_human_labels()
		self._sync_auto_next_point_for_frame()
		if self.auto_next_point:
			self.video_handler.data_handler.reset_active_point_for_frame(frame_number)

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
		"""Write human labels to CSV without closing the labeler session."""
		return self.video_handler.save_labels(save_path)

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
			try:
				if preview:
					# Scrub preview: show machine predictions when available, skip human overlays.
					self.video_handler.show_machine_labels = bool(
						self.video_handler.machine_labels_path
					)
					if self.video_handler.show_machine_labels:
						self.video_handler.ensure_machine_labels_loaded()
				else:
					self.video_handler.show_machine_labels = self.show_machine_labels
				self._sync_annotator_overlay_flags()
				image = self.video_handler.create_grid_image(
					render_at,
					annotate_images=not preview,
					preview=preview,
				)
			finally:
				if preview:
					self.video_handler.show_machine_labels = prev_show
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

	def _machine_nonempty_frames(self) -> list[int] | None:
		"""Frame indices with machine predictions, or None when overlay is unavailable."""
		handler = self.video_handler
		if not handler.machine_labels_path:
			return None
		handler.ensure_machine_labels_loaded()
		mlh = handler.machine_labels_handler
		if mlh is None:
			return None
		return mlh.get_nonempty_frames()

	def state_dict(self) -> dict:
		handler = self.video_handler
		active = handler.data_handler.active_point
		labeled_frame_list = handler.data_handler.get_nonempty_frames()
		labeled = len(labeled_frame_list)
		machine_frame_list = self._machine_nonempty_frames()
		nav_frame_list = build_nav_frame_list(labeled_frame_list, machine_frame_list)
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
			"has_machine_labels": bool(handler.machine_labels_path),
			"auto_next_point": self.auto_next_point,
			"grid_width": handler.grid_parameters.total_width,
			"grid_height": handler.grid_parameters.total_height,
		}

	def close(self, save: bool, save_path: str | None = None) -> str | None:
		return self.video_handler.close(save_data=save, save_path=save_path)
