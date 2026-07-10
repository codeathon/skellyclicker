"""Read-only debug snapshot for /api/health — never mutates session state."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


# Bump when health payload shape or session fixes change (proves process restarted).
CODE_STAMP = "2026-07-10-health-debug-v1"


def _path_info(path: str | None) -> dict[str, Any]:
	"""Cheap exists/size check for one path (no directory walks)."""
	if not path:
		return {"path": None, "exists": False}
	p = Path(path).expanduser()
	info: dict[str, Any] = {
		"path": str(p),
		"exists": p.exists(),
		"is_file": p.is_file() if p.exists() else False,
		"is_dir": p.is_dir() if p.exists() else False,
	}
	if p.is_file():
		try:
			info["size_bytes"] = p.stat().st_size
		except OSError as exc:
			info["stat_error"] = str(exc)
	return info


def _scan_machine_csv_leftovers(
	*,
	dlc_project_path: str | None,
	video_paths: list[str] | None,
) -> list[dict[str, Any]]:
	"""List leftover machine-label CSVs on disk without attaching them to the session.

	Why: explains Loaded Assets ghosts — old servers auto-attached these; we only report.
	"""
	from skellyclicker.services.dlc_paths import (
		iter_machine_labels_csvs,
		resolve_dlc_project_input,
	)

	found: list[dict[str, Any]] = []
	project_dir: Path | None = None
	if dlc_project_path:
		try:
			project_dir, _ = resolve_dlc_project_input(dlc_project_path)
		except ValueError as exc:
			return [{"error": f"dlc_project_unresolved: {exc}"}]
	# Without a project, still scan video parents for beside-video leftovers.
	if project_dir is not None:
		csvs = iter_machine_labels_csvs(project_dir, video_paths)
	elif video_paths:
		# Same glob patterns as iter_machine_labels_csvs, but no project root.
		csvs = []
		seen: set[Path] = set()
		patterns = (
			"model_outputs/**/skellyclicker_machine_labels_iteration_*.csv",
			"*_model_outputs_iteration_*/skellyclicker_machine_labels_iteration_*.csv",
		)
		for raw in video_paths:
			parent = Path(raw).expanduser().resolve().parent
			for pattern in patterns:
				for path in sorted(parent.glob(pattern)):
					resolved = path.resolve()
					if resolved.is_file() and resolved not in seen:
						seen.add(resolved)
						csvs.append(resolved)
	else:
		return []
	for csv in csvs:
		found.append(
			{
				"path": str(csv),
				"exists": csv.is_file(),
				"under_project": (
					project_dir is not None
					and _is_relative_to(csv, project_dir)
				),
			}
		)
	return found


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve().relative_to(root.resolve())
		return True
	except (ValueError, OSError):
		return False


def _git_head(repo_root: Path) -> str | None:
	"""Best-effort HEAD sha from .git (no subprocess) — None if unavailable."""
	head = repo_root / ".git" / "HEAD"
	try:
		text = head.read_text().strip()
		if text.startswith("ref:"):
			ref = text.split(" ", 1)[1].strip()
			ref_path = repo_root / ".git" / ref
			return ref_path.read_text().strip()[:12]
		return text[:12]
	except OSError:
		return None


def build_health_debug(store: Any, *, repo_root: Path) -> dict[str, Any]:
	"""Assemble a full debug dump from in-memory store + cheap path checks.

	Only called from GET /api/health — does not run on other API routes.
	"""
	session = store.session
	videos = list(session.videos or [])
	live = store.live_inference
	engine = store.labeling_engine
	dlc = store.dlc_handler

	# Detected mode from current videos (may differ from session.labeling_mode if stale).
	detected_mode = None
	detect_error = None
	try:
		from skellyclicker.services.labeling_mode import detect_labeling_mode

		detected_mode = detect_labeling_mode(videos).value if videos else "single"
	except Exception as exc:  # noqa: BLE001 — health must never 500
		detect_error = f"{type(exc).__name__}: {exc}"

	jobs = []
	for job in store.jobs.values():
		jobs.append(
			{
				"job_id": job.job_id,
				"name": job.name,
				"status": job.status.value,
				"message": job.message,
				"progress_percent": job.progress_percent,
				"log_tail": (job.log_lines or [])[-5:],
			}
		)

	labeler: dict[str, Any] | None = None
	if engine is not None:
		labeler = {
			"session_id": engine.session_id,
			"labeling_mode": engine.labeling_mode.value,
			"active_video_path": engine.active_video_path,
			"frame_number": engine.frame_number,
			"frame_count": engine.video_handler.frame_count,
			"show_machine_labels": engine.show_machine_labels,
			"open_video_paths": list(engine.video_handler.videos.keys()),
			"machine_labels_path_on_handler": engine.video_handler.machine_labels_path,
		}

	live_info: dict[str, Any] = {
		"loaded": live is not None,
		"ready": bool(live and getattr(live, "ready", False)),
		"load_error": getattr(live, "load_error", None) if live else None,
		"config_path": getattr(live, "_config_path", None) if live else None,
		"bodyparts": list(getattr(live, "bodyparts", []) or []) if live else [],
	}

	dlc_info: dict[str, Any] = {
		"loaded": dlc is not None,
		"project_config_path": getattr(dlc, "project_config_path", None) if dlc else None,
		"iteration": getattr(dlc, "iteration", None) if dlc else None,
		"tracked_point_names": list(getattr(dlc, "tracked_point_names", []) or [])
		if dlc
		else [],
	}

	return {
		"ok": True,
		"code_stamp": CODE_STAMP,
		"process": {
			"pid": os.getpid(),
			"cwd": os.getcwd(),
			"python": sys.executable,
			"python_version": sys.version.split()[0],
			"argv": list(sys.argv),
			"git_head": _git_head(repo_root),
			"repo_root": str(repo_root),
			"frontend_dist_exists": (repo_root / "frontend" / "dist").is_dir(),
		},
		"session": session.model_dump(mode="json"),
		"paths": {
			"videos": [_path_info(v) for v in videos],
			"human_labels": _path_info(session.human_labels_path),
			"machine_labels": _path_info(session.machine_labels_path),
			"dlc_project": _path_info(session.dlc_project_path),
			"session_saved": _path_info(session.session_saved_path),
			"active_video": _path_info(session.active_video_path),
		},
		"labeling": {
			"session_mode": session.labeling_mode.value,
			"detected_mode": detected_mode,
			"detect_error": detect_error,
			"mode_mismatch": (
				detected_mode is not None
				and detected_mode != session.labeling_mode.value
			),
			"can_open_labeler": store._can_open_labeler(),
			"labeler_open": engine is not None,
			"labeler": labeler,
		},
		"live_inference": live_info,
		"dlc_handler": dlc_info,
		"jobs": jobs,
		# Disk leftovers that old servers would auto-attach — never written to session here.
		"machine_csv_leftovers_on_disk": _scan_machine_csv_leftovers(
			dlc_project_path=session.dlc_project_path,
			video_paths=videos or None,
		),
	}
