"""Block leaving a frame once human labeling has started but is incomplete."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skellyclicker.core.click_data_handler.data_handler import DataHandler, DataHandlerConfig
from skellyclicker.core.video_handler.video_models import ClickData
from skellyclicker.services.errors import SessionError
from skellyclicker.services.labeling_engine import LabelingEngine


def _handler(
	*,
	videos: list[str] | None = None,
	points: list[str] | None = None,
) -> DataHandler:
	config = DataHandlerConfig(
		num_frames=10,
		video_names=videos or ["cam.mp4"],
		tracked_point_names=points or ["nose", "tail"],
	)
	return DataHandler.from_config(config)


def _place(dh: DataHandler, *, frame: int, point: str, xy: int = 1) -> None:
	dh.update_dataframe(
		ClickData(
			window_x=xy,
			window_y=xy,
			video_x=xy,
			video_y=xy,
			frame_number=frame,
			video_index=0,
		),
		point_name=point,
	)


def test_empty_frame_is_not_incomplete():
	dh = _handler()
	assert dh.incomplete_labeling_message(0) is None


def test_partial_frame_reports_missing_bodyparts():
	dh = _handler()
	_place(dh, frame=0, point="nose")
	msg = dh.incomplete_labeling_message(0)
	assert msg is not None
	assert "Still missing: tail" in msg


def test_complete_frame_is_not_incomplete():
	dh = _handler()
	_place(dh, frame=0, point="nose", xy=1)
	_place(dh, frame=0, point="tail", xy=2)
	assert dh.incomplete_labeling_message(0) is None


def test_set_frame_blocks_when_partially_labeled():
	dh = _handler()
	_place(dh, frame=3, point="nose")
	handler = MagicMock()
	handler.data_handler = dh
	handler.machine_labels_path = None
	engine = LabelingEngine(video_handler=handler)
	engine.frame_number = 3
	engine._maybe_request_live_infer = MagicMock()  # type: ignore[method-assign]

	with pytest.raises(SessionError, match="Finish labeling"):
		engine.set_frame(4)
	assert engine.frame_number == 3


def test_set_frame_allows_skip_from_empty_frame():
	dh = _handler()
	handler = MagicMock()
	handler.data_handler = dh
	handler.machine_labels_path = None
	engine = LabelingEngine(video_handler=handler)
	engine.frame_number = 0
	engine._maybe_request_live_infer = MagicMock()  # type: ignore[method-assign]
	engine._frame_has_human_labels = MagicMock(return_value=False)  # type: ignore[method-assign]

	engine.set_frame(2)
	assert engine.frame_number == 2
