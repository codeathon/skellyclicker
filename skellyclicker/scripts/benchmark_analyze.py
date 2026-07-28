"""Benchmark Full Analysis timing (inference + per-phase) for 1-GPU vs 2-GPU runs.

Why this exists: to actually see the multi-GPU speedup. Run once with
``CUDA_VISIBLE_DEVICES=0`` (device_count()==1 -> sequential) and once with
``CUDA_VISIBLE_DEVICES=0,1`` (device_count()==2 -> one video per GPU), then
compare the ``inference`` numbers. With N similar-length videos on 2 GPUs the
inference wall time should drop to ~1/2.

Zero overhead / no core changes: per-phase durations are derived purely from the
progress-callback boundary messages that ``analyze_videos`` already reports
(inference/filter/merge/plot/annotate). We only stamp perf_counter() when those
messages first appear, so the pipeline itself is untouched.

Usage:
	CUDA_VISIBLE_DEVICES=0 python -m skellyclicker.scripts.benchmark_analyze \
		--config /path/to/dlc_project/config.yaml \
		--video-folder /path/to/four_videos \
		--output /tmp/bench_1gpu --label "1 GPU"

	CUDA_VISIBLE_DEVICES=0,1 python -m skellyclicker.scripts.benchmark_analyze \
		--config /path/to/dlc_project/config.yaml \
		--video-folder /path/to/four_videos \
		--output /tmp/bench_2gpu --label "2 GPU"
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from skellyclicker.core.deeplabcut_handler.deeplabcut_handler import DeeplabcutHandler
from skellyclicker.core.deeplabcut_handler.parallel_analyze import resolve_worker_count

# Boundary message substrings -> phase name. These match the report(...) calls in
# analyze_videos; first sighting of each marks that phase's start. "Merging
# machine labels CSV" (not the repeated "Merging machine labels (x/y)") pins the
# merge start; "Analyzing" would be ambiguous so inference start is just t0.
_POST_INFERENCE_PHASES = ("filter", "merge", "plot", "annotate")
_PHASE_MARKS = {
	"inference_done": "Inference complete",
	"filter": "Filtering predictions",
	"merge": "Merging machine labels CSV",
	"plot": "Plotting trajectories",
	"annotate": "Annotating videos",
}


class PhaseRecorder:
	"""Stamp perf_counter() the first time each boundary message appears."""

	def __init__(self) -> None:
		self.marks: dict[str, float] = {}

	def __call__(self, fraction: float | None, message: str) -> None:
		# Called by analyze_videos on every progress update; record first hit only.
		if not message:
			return
		now = time.perf_counter()
		for name, needle in _PHASE_MARKS.items():
			if name not in self.marks and needle in message:
				self.marks[name] = now


def compute_durations(marks: dict[str, float], t0: float, t_end: float) -> dict[str, float]:
	"""Turn boundary timestamps into per-phase seconds.

	inference = start -> "Inference complete" (includes model load, which is the
	real inference cost). Each post-inference phase runs to the next phase that
	actually happened, and the last one to t_end.
	"""
	durations = {"inference": marks.get("inference_done", t_end) - t0}
	present = [p for p in _POST_INFERENCE_PHASES if p in marks]
	boundaries = [marks[p] for p in present] + [t_end]
	for i, phase in enumerate(present):
		durations[phase] = boundaries[i + 1] - boundaries[i]
	return durations


def describe_environment(num_videos: int) -> tuple[str, int, list[str], int]:
	"""Report visible GPUs + the worker count analyze_videos will auto-resolve."""
	visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
	count, names = 0, []
	try:
		import torch

		if torch.cuda.is_available():
			count = torch.cuda.device_count()
			names = [torch.cuda.get_device_name(i) for i in range(count)]
	except Exception as exc:  # torch/driver issue -> treat as CPU-only.
		print(f"  (GPU probe failed: {exc})")
	# None => auto (one worker per GPU); mirrors the real analyze path.
	workers = resolve_worker_count(num_videos, None)
	return visible, count, names, workers


def resolve_config(config: str) -> Path:
	"""Accept either a config.yaml or a project dir containing one."""
	path = Path(config)
	return path / "config.yaml" if path.is_dir() else path


def resolve_videos(args: argparse.Namespace) -> list[Path]:
	"""Explicit --videos wins; otherwise glob *.mp4 in --video-folder (sorted)."""
	if args.videos:
		return [Path(v) for v in args.videos]
	if args.video_folder:
		return sorted(Path(args.video_folder).glob("*.mp4"))
	raise SystemExit("Provide --videos or --video-folder")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", required=True, help="DLC config.yaml or project dir")
	parser.add_argument("--video-folder", help="folder of *.mp4 to analyze")
	parser.add_argument("--videos", nargs="+", help="explicit video paths")
	parser.add_argument("--output", required=True, help="destination folder for DLC output")
	parser.add_argument("--label", default="run", help="label printed in the summary")
	# Default both off so a run isolates inference; enable to time all steps.
	parser.add_argument("--filter", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=False)
	return parser.parse_args()


def print_summary(
	label: str,
	num_videos: int,
	visible: str,
	gpu_count: int,
	workers: int,
	durations: dict[str, float],
	total: float,
) -> None:
	print("\n" + "=" * 52)
	print(f"BENCHMARK: {label}")
	print(f"  videos={num_videos}  CUDA_VISIBLE_DEVICES={visible}  "
	      f"visible_gpus={gpu_count}  workers={workers}")
	mode = "parallel (1 video/GPU)" if workers > 1 else "sequential"
	print(f"  mode={mode}")
	print("-" * 52)
	for phase, secs in durations.items():
		print(f"  {phase:<12} {secs:9.2f}s")
	print("-" * 52)
	print(f"  {'total':<12} {total:9.2f}s")
	print("=" * 52)


def main() -> None:
	args = parse_args()
	config = resolve_config(args.config)
	videos = resolve_videos(args)
	if not videos:
		raise SystemExit("No videos found to analyze")
	output = Path(args.output)
	output.mkdir(parents=True, exist_ok=True)

	visible, gpu_count, names, workers = describe_environment(len(videos))
	print(f"Config: {config}")
	print(f"Videos ({len(videos)}): {[v.name for v in videos]}")
	print(f"GPUs: {names or 'none'} (CUDA_VISIBLE_DEVICES={visible})")
	if len(videos) > 1 and workers == 1:
		print("  NOTE: 1 worker -> sequential. Set CUDA_VISIBLE_DEVICES=0,1 for the 2-GPU run.")

	handler = DeeplabcutHandler.load_deeplabcut_project(project_config_path=str(config))
	recorder = PhaseRecorder()
	t0 = time.perf_counter()
	handler.analyze_videos(
		video_paths=[str(v) for v in videos],
		output_folder=str(output),
		filter_videos=args.filter,
		annotate_videos=args.annotate,
		progress_callback=recorder,
	)
	t_end = time.perf_counter()

	durations = compute_durations(recorder.marks, t0, t_end)
	print_summary(args.label, len(videos), visible, gpu_count, workers, durations, t_end - t0)


if __name__ == "__main__":
	main()
