"""Headless labeling session — wraps VideoHandler without OpenCV windows."""

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from skellyclicker import MAX_WINDOW_SIZE
from skellyclicker.core.video_handler.video_handler import VideoHandler

TRACKED_POINTS_JSON = (
	Path(__file__).resolve().parents[2] / "tracked_points.json"
)


class LabelingEngine(BaseModel):
	"""Server-side labeler for one open labeling session."""
	model_config = ConfigDict(arbitrary_types_allowed=True)

	session_id: str = Field(default_factory=lambda: str(uuid4()))
	video_handler: VideoHandler
	frame_number: int = 0
	auto_next_point: bool = True
	show_machine_labels: bool = False

	@classmethod
	def open(
		cls,
		video_paths: list[str],
		human_labels_path: str | None,
		machine_labels_path: str | None,
		train_on_machine_labels: bool,
	) -> "LabelingEngine":
		# Machine-only mode uses machine CSV as primary label source.
		if train_on_machine_labels and machine_labels_path:
			primary = machine_labels_path
			overlay = None
		elif human_labels_path:
			primary = human_labels_path
			overlay = machine_labels_path
		else:
			primary = str(TRACKED_POINTS_JSON)
			overlay = machine_labels_path

		handler = VideoHandler.from_videos(
			video_paths=video_paths,
			max_window_size=MAX_WINDOW_SIZE,
			data_handler_path=primary,
			machine_labels_path=overlay,
		)
		return cls(video_handler=handler)

	def render_frame_jpeg(self, frame_number: int | None = None) -> bytes:
		if frame_number is not None:
			self.frame_number = frame_number
		self.video_handler.show_machine_labels = self.show_machine_labels
		image = self.video_handler.create_grid_image(self.frame_number)
		ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
			"auto_next_point": self.auto_next_point,
			"grid_width": handler.grid_parameters.total_width,
			"grid_height": handler.grid_parameters.total_height,
		}

	def close(self, save: bool, save_path: str | None = None) -> str | None:
		return self.video_handler.close(save_data=save, save_path=save_path)
