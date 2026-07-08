"""Evenly-spread frame sampling for partial-analysis performance estimates."""

from __future__ import annotations


def evenly_spread_sample(
	n_frames: int,
	exclude: set[int],
	*,
	fraction: float,
	min_frames: int,
	max_frames: int,
) -> list[int]:
	"""Deterministic evenly-spaced unlabeled frame indices for a perf estimate."""
	if n_frames <= 0:
		return []

	target = round(fraction * n_frames)
	target = max(min_frames, min(max_frames, target))
	target = min(target, n_frames)

	if target <= 0:
		return []

	if target == 1:
		candidates = [0]
	elif n_frames == 1:
		candidates = [0]
	else:
		# Evenly spaced endpoints inclusive — same count as target after rounding.
		step = (n_frames - 1) / (target - 1)
		candidates = [round(i * step) for i in range(target)]

	seen: set[int] = set()
	result: list[int] = []
	for frame in candidates:
		idx = int(max(0, min(n_frames - 1, frame)))
		if idx in exclude or idx in seen:
			continue
		seen.add(idx)
		result.append(idx)

	return sorted(result)
