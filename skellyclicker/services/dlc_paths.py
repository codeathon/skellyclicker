"""Resolve DLC project paths consistently for load, train, and analyze."""

from pathlib import Path

DEEPLABCUT_CONFIG = "config.yaml"


def resolve_dlc_project_input(path: str) -> tuple[Path, Path]:
	"""Return (project_dir, config_path) from a project folder or config.yaml file."""
	raw = Path(path).expanduser()
	if raw.is_file() and raw.name == DEEPLABCUT_CONFIG:
		config_path = raw.resolve()
		return config_path.parent, config_path
	if raw.is_dir() and (raw / DEEPLABCUT_CONFIG).is_file():
		project_dir = raw.resolve()
		return project_dir, project_dir / DEEPLABCUT_CONFIG
	raise ValueError(
		f"Not a DLC project: expected a folder containing {DEEPLABCUT_CONFIG} "
		f"or a path to {DEEPLABCUT_CONFIG}"
	)


def dlc_project_dir(project_config_path: str) -> Path:
	"""Project root from the loaded config.yaml path (authoritative for SkellyClicker)."""
	return Path(project_config_path).expanduser().resolve().parent


def analyze_output_folder(
	project_config_path: str,
	use_training_videos: bool,
	video_paths: list[str],
	iteration: int | None = None,
) -> Path:
	"""Output folder for analyze — always anchored to the loaded config file location."""
	from deeplabcut.utils import auxiliaryfunctions

	project_dir = dlc_project_dir(project_config_path)
	cfg = auxiliaryfunctions.read_config(project_config_path)
	iter_num = iteration if iteration is not None else int(cfg["iteration"])
	if use_training_videos:
		return project_dir / "model_outputs" / f"model_outputs_iteration_{iter_num}"
	project_name = cfg.get("Task", "project")
	return (
		Path(video_paths[0]).resolve().parent
		/ f"{project_name}_model_outputs_iteration_{iter_num}"
	)
