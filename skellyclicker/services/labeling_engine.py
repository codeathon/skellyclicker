"""Headless labeling session — wraps VideoHandler without OpenCV windows."""

import threading
from uuid import uuid4

import cv2
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from skellyclicker import (
	LABELER_JPEG_QUALITY_COMMITTED,
	LABELER_JPEG_QUALITY_PREVIEW,
)
from skellyclicker.core.video_handler.video_handler import VideoHandler


class LabelingEngine(BaseModel):
	"""Server-side labeler for one open labeling session."""
	model_config = ConfigDict(arbitrary_types_allowed=True)

	session_id: str = Field(default_factory=lambda: str(uuid4()))
	video_handler: VideoHandler
	frame_number: int = 0
	auto_next_point: bool = True
	show_machine_labels: bool = False
	show_help: bool = False
	# OpenCV VideoCapture is not thread-safe; scrub previews hit this concurrently.
	_render_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

	@classmethod
	def open(
		cls,
		video_paths: list[str],
		human_labels_path: str | None,
		machine_labels_path: str | None,
		train_on_machine_labels: bool,
		tracked_point_names: list[str],
	) -> "LabelingEngine":
		# Machine-only mode uses machine CSV as primary label source.
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
		engine = cls(video_handler=handler)
		engine.video_handler.image_annotator.config.web_help = True
		engine.sync_active_point()
		return engine

	def sync_active_point(self) -> None:
		"""Align active bodypart with the current frame before labeling."""
		self.video_handler.data_handler.reset_active_point_for_frame(
			self.frame_number,
		)

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
			if frame_number is not None and not preview:
				self.frame_number = frame_number
				self.sync_active_point()

			prev_show = self.video_handler.show_machine_labels
			prev_help = self.video_handler.image_annotator.config.show_help
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
				self.video_handler.image_annotator.config.show_help = self.show_help
				image = self.video_handler.create_grid_image(
					render_at,
					annotate_images=not preview,
					preview=preview,
				)
			finally:
				if preview:
					self.video_handler.show_machine_labels = prev_show
				self.video_handler.image_annotator.config.show_help = prev_help

			quality = (
				LABELER_JPEG_QUALITY_PREVIEW if preview else LABELER_JPEG_QUALITY_COMMITTED
			)
			ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
			if not ok:
				raise RuntimeError("Failed to encode frame as JPEG")
			return encoded.tobytes()

	def handle_click(self, x: int, y: int) -> None:
		self.video_handler.handle_clicks(
			x, y, self.frame_number, auto_next_point=self.auto_next_point
		)

	def state_dict(self) -> dict:
		handler = self.video_handler
		active = handler.data_handler.active_point
		labeled = len(handler.data_handler.get_nonempty_frames())
		return {
			"session_id": self.session_id,
			"frame_number": self.frame_number,
			"frame_count": handler.frame_count,
			"active_point": active,
			"tracked_points": handler.data_handler.config.tracked_point_names,
			"labeled_frames": labeled,
			"show_machine_labels": self.show_machine_labels,
			"show_help": self.show_help,
			"has_machine_labels": bool(handler.machine_labels_path),
			"auto_next_point": self.auto_next_point,
			"grid_width": handler.grid_parameters.total_width,
			"grid_height": handler.grid_parameters.total_height,
		}

	def close(self, save: bool, save_path: str | None = None) -> str | None:
		return self.video_handler.close(save_data=save, save_path=save_path)
