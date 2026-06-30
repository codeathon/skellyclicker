"""Web labeler external HUD skips grid text overlays."""

import numpy as np

from skellyclicker.core.video_handler.image_annotator import ImageAnnotator, ImageAnnotatorConfig


def test_external_hud_skips_grid_overlays():
	img = np.zeros((1080, 1920, 3), dtype=np.uint8)
	annotator = ImageAnnotator(
		config=ImageAnnotatorConfig(
			external_hud=True,
			show_help=True,
			show_legend=True,
		),
	)
	out = annotator.annotate_image_grid(img, active_point="nose", frame_number=42)
	assert np.array_equal(out, img)


def test_external_hud_disables_click_status_overlay():
	img = np.zeros((480, 640, 3), dtype=np.uint8)
	annotator = ImageAnnotator(
		config=ImageAnnotatorConfig(
			external_hud=True,
			show_clicks=False,
			tracked_points=["nose"],
		),
	)
	out = annotator.annotate_single_image(img, active_point="nose", click_data={})
	assert np.array_equal(out, img)
