"""Build color-coded frame navigation list for the web labeler."""

from __future__ import annotations

from typing import Literal

from skellyclicker import MAX_NAV_MACHINE_FRAMES

NavFrameKind = Literal["human", "machine", "both"]


def build_nav_frame_list(
	human_frames: list[int],
	machine_frames: list[int] | None,
) -> list[dict[str, int | str]]:
	"""Merge human and machine frame indices with kind tags for the left panel."""
	human_set = set(human_frames)
	machine_set: set[int] = set()
	if machine_frames is not None and len(machine_frames) <= MAX_NAV_MACHINE_FRAMES:
		machine_set = set(machine_frames)

	all_frames = sorted(human_set | machine_set)
	nav: list[dict[str, int | str]] = []
	for frame in all_frames:
		in_human = frame in human_set
		in_machine = frame in machine_set
		if in_human and in_machine:
			kind: NavFrameKind = "both"
		elif in_human:
			kind = "human"
		else:
			kind = "machine"
		nav.append({"frame": frame, "kind": kind})
	return nav
