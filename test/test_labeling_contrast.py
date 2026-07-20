"""Display-only contrast for the web labeler."""

from __future__ import annotations

from unittest.mock import MagicMock

from skellyclicker.services.labeling_engine import (
	CONTRAST_DEFAULT,
	CONTRAST_MAX,
	CONTRAST_MIN,
	LabelingEngine,
)


def test_set_contrast_clamps_and_applies_to_all_videos():
	v0 = MagicMock()
	v0.contrast = CONTRAST_DEFAULT
	v1 = MagicMock()
	v1.contrast = CONTRAST_DEFAULT
	handler = MagicMock()
	handler.videos = {"a.mp4": v0, "b.mp4": v1}
	engine = LabelingEngine(video_handler=handler)

	assert engine.set_contrast(10.0) == CONTRAST_MAX
	assert v0.contrast == CONTRAST_MAX
	assert v1.contrast == CONTRAST_MAX

	assert engine.set_contrast(0.0) == CONTRAST_MIN
	assert engine.contrast == CONTRAST_MIN


def test_set_contrast_default_roundtrip():
	video = MagicMock()
	video.contrast = 2.0
	handler = MagicMock()
	handler.videos = {"cam.mp4": video}
	engine = LabelingEngine(video_handler=handler)
	assert engine.contrast == 2.0
	assert engine.set_contrast(CONTRAST_DEFAULT) == CONTRAST_DEFAULT
	assert video.contrast == CONTRAST_DEFAULT
