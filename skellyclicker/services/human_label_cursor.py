"""PNG cursor images for the web labeler (human-label diamond marker)."""

import cv2
import numpy as np

CURSOR_SIZE = 32
CURSOR_HOTSPOT = 16


def human_label_cursor_png(r: int, g: int, b: int) -> bytes:
	"""Return a 32x32 RGBA PNG: hollow diamond with white halo, tinted to bodypart RGB."""
	r = max(0, min(255, int(r)))
	g = max(0, min(255, int(g)))
	b = max(0, min(255, int(b)))

	img = np.zeros((CURSOR_SIZE, CURSOR_SIZE, 4), dtype=np.uint8)
	center = CURSOR_HOTSPOT
	radius = 10
	pts = np.array(
		[
			[center, center - radius],
			[center + radius, center],
			[center, center + radius],
			[center - radius, center],
		],
		dtype=np.int32,
	)
	# BGRA for OpenCV; white halo then bodypart color (matches legend / on-video markers).
	cv2.polylines(img, [pts], True, (255, 255, 255, 255), 3, cv2.LINE_AA)
	cv2.polylines(img, [pts], True, (b, g, r, 255), 2, cv2.LINE_AA)
	# GTK/Firefox on Linux ignore cursors when the hotspot pixel is fully transparent.
	img[center, center] = (b, g, r, 255)

	ok, encoded = cv2.imencode(".png", img)
	if not ok:
		raise RuntimeError("Failed to encode human label cursor PNG")
	return encoded.tobytes()
