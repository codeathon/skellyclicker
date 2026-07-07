"""Overlay scale helper for native-resolution labeler."""

from skellyclicker.core.video_handler.image_annotator import overlay_scale


def test_overlay_scale_1080p_unchanged():
	assert overlay_scale(1080) == 1.0


def test_overlay_scale_4k_halved():
	assert overlay_scale(2160) == 0.5


def test_overlay_scale_small_video_capped_at_one():
	assert overlay_scale(720) == 1.0
