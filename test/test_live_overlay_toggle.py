"""m toggles both live on-the-fly overlays and saved machine CSV overlays."""

from unittest.mock import MagicMock

from skellyclicker.services.labeling_engine import LabelingEngine
from skellyclicker.services.models import LabelingMode


def _engine_with_live(*, ready: bool = True) -> tuple[LabelingEngine, MagicMock]:
	handler = MagicMock()
	handler.machine_labels_path = None
	handler.machine_labels_handler = None
	handler.machine_labels_annotator = None
	handler.live_points_lookup = None
	handler.frame_count = 10
	handler.videos = {"/tmp/cam.mp4": MagicMock()}
	handler.data_handler.config.tracked_point_names = ["nose"]
	handler.data_handler.active_point = "nose"
	handler.image_annotator.config.show_help = False
	handler.image_annotator.config.show_names = False
	handler.grid_parameters.total_width = 100
	handler.grid_parameters.total_height = 100

	engine = LabelingEngine(
		video_handler=handler,
		labeling_mode=LabelingMode.single,
		session_video_paths=["/tmp/cam.mp4"],
		active_video_path="/tmp/cam.mp4",
		show_machine_labels=False,
	)
	live = MagicMock()
	live.ready = ready
	live.load_error = None
	live.bodyparts = ["nose"]
	live.get_cached.return_value = None
	live.get_overlay_points = MagicMock(return_value=None)
	return engine, live


def test_attach_live_inference_turns_overlay_on_when_ready():
	engine, live = _engine_with_live(ready=True)
	engine.attach_live_inference(live)
	assert engine.show_machine_labels is True
	assert engine.video_handler.live_points_lookup is live.get_overlay_points


def test_maybe_request_live_infer_skips_when_overlay_hidden():
	engine, live = _engine_with_live(ready=True)
	engine.attach_live_inference(live)
	engine.show_machine_labels = False
	engine._maybe_request_live_infer(3)
	live.request_infer.assert_not_called()


def test_maybe_request_live_infer_runs_when_overlay_shown():
	engine, live = _engine_with_live(ready=True)
	engine.attach_live_inference(live)
	engine.show_machine_labels = True
	engine._maybe_request_live_infer(3)
	live.request_infer.assert_called_once_with("/tmp/cam.mp4", 3)
