"""Build color-coded frame navigation list for the web labeler."""

from __future__ import annotations

from typing import Literal

from skellyclicker import MAX_NAV_MACHINE_FRAMES

NavFrameKind = Literal["human", "machine", "both"]


def build_nav_frame_list(
	human_frames: list[int],
	machine_frames: list[int] | None,
	*,
	sample_frames: list[int] | None = None,
) -> list[dict[str, int | str]]:
	"""Merge human and predicted frame indices with kind tags for the left panel.

	When `sample_frames` is provided (from the partial-analyze sidecar) it is the
	authoritative "predicted" set and is always shown, even if the machine CSV is
	dense. Otherwise fall back to machine_frames, guarded so a dense full-analysis
	CSV does not flood the panel.
	"""
	human_set = set(human_frames)
	if sample_frames is not None:
		predicted_set = set(sample_frames)
	elif machine_frames is not None and len(machine_frames) <= MAX_NAV_MACHINE_FRAMES:
		predicted_set = set(machine_frames)
	else:
		predicted_set = set()

	all_frames = sorted(human_set | predicted_set)
	nav: list[dict[str, int | str]] = []
	for frame in all_frames:
		in_human = frame in human_set
		in_machine = frame in predicted_set
		if in_human and in_machine:
			kind: NavFrameKind = "both"
		elif in_human:
			kind = "human"
		else:
			kind = "machine"
		nav.append({"frame": frame, "kind": kind})
	return nav
