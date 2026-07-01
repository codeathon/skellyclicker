"""Tests for human-label cursor PNG generation."""

from skellyclicker.services.human_label_cursor import human_label_cursor_png


def test_human_label_cursor_png_is_valid_png():
	png = human_label_cursor_png(255, 128, 0)
	assert png.startswith(b"\x89PNG\r\n\x1a\n")
	assert len(png) > 100
