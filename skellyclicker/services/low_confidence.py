"""Find machine-label predictions below a likelihood threshold."""

from __future__ import annotations

import pandas as pd

from skellyclicker.core.click_data_handler.data_handler import DataHandler

# DLC default pcutoff used across the project.
DEFAULT_LIKELIHOOD_THRESHOLD = 0.6


def find_low_confidence_items(
	handler: DataHandler,
	*,
	threshold: float = DEFAULT_LIKELIHOOD_THRESHOLD,
) -> list[dict]:
	"""Return sorted low-confidence (frame, bodypart) items from machine overlay CSV."""
	likelihood_cols = [
		c for c in handler.dataframe.columns if c.endswith("_likelihood")
	]
	if not likelihood_cols:
		return []

	video_names = handler.config.video_names
	items: list[dict] = []

	for (video_name, frame_number), row in handler.dataframe.iterrows():
		try:
			video_index = video_names.index(str(video_name))
		except ValueError:
			continue
		frame_int = int(frame_number)
		for col in likelihood_cols:
			likelihood = row[col]
			if pd.isna(likelihood) or float(likelihood) >= threshold:
				continue
			bodypart = col[: -len("_likelihood")]
			x_col = f"{bodypart}_x"
			y_col = f"{bodypart}_y"
			if x_col not in row.index or y_col not in row.index:
				continue
			x, y = row[x_col], row[y_col]
			if pd.isna(x) or pd.isna(y):
				continue
			items.append(
				{
					"frame_number": frame_int,
					"video_index": video_index,
					"bodypart": bodypart,
					"likelihood": float(likelihood),
					"x": float(x),
					"y": float(y),
				}
			)

	items.sort(key=lambda item: (item["frame_number"], item["likelihood"], item["bodypart"]))
	return items
