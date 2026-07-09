"""Resolve CSV video basenames to absolute session paths (cross-folder training)."""

from __future__ import annotations

from pathlib import Path


def build_video_path_registry(video_paths: list[str]) -> dict[str, str]:
	"""Map basename → absolute path. Duplicate basenames raise (ambiguous)."""
	registry: dict[str, str] = {}
	for raw in video_paths:
		path = str(Path(raw).expanduser().resolve())
		name = Path(path).name
		if name in registry and registry[name] != path:
			raise ValueError(
				f"Duplicate video basename {name!r} from different folders: "
				f"{registry[name]} and {path}. Rename one file or remove a duplicate."
			)
		registry[name] = path
	return registry


def resolve_video_path(video_name: str, video_paths: list[str]) -> str:
	"""Resolve a CSV `video` column value to an absolute path in the session list."""
	registry = build_video_path_registry(video_paths)
	name = Path(video_name).name
	if name not in registry:
		raise FileNotFoundError(
			f"Video {video_name!r} from labels CSV not found in session videos. "
			f"Known: {sorted(registry)}"
		)
	return registry[name]


def labeled_data_prefix(video_path: str) -> str:
	"""DLC labeled-data folder prefix from a video's parent path (session_* or folder name)."""
	parent = Path(video_path).parent
	for part in parent.parts:
		if part.startswith("session") or part.startswith("Session"):
			return part
	return parent.name or "videos"
