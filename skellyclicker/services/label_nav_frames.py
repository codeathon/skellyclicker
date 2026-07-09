"""Build color-coded frame navigation list for the web labeler."""

from __future__ import annotations

from typing import Literal

NavFrameKind = Literal["human", "machine", "both"]


def build_nav_frame_list(
	human_frames: list[int],
	machine_frames: list[int] | None = None,
	*,
	sample_frames: list[int] | None = None,
) -> list[dict[str, int | str]]:
	"""Left-panel nav: human-labeled frames only.

	Predicted/machine frames are reviewed via live scrub (or Full Analysis CSV
	overlay with `m`). `machine_frames` / `sample_frames` are ignored for nav.
	"""
	_ = machine_frames, sample_frames  # kept for call-site compatibility
	return [{"frame": frame, "kind": "human"} for frame in sorted(set(human_frames))]
