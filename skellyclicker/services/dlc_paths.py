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


DEEPLABCUT_CONFIG = "config.yaml"
PYTORCH_MODELS_DIR = "dlc-models-pytorch"
PYTORCH_TRAIN_CONFIG = "pytorch_config.yaml"


def iteration_has_pytorch_model(project_dir: Path, iteration: int) -> bool:
	"""True if iteration-N contains a shuffle folder with train/pytorch_config.yaml."""
	iter_root = project_dir / PYTORCH_MODELS_DIR / f"iteration-{iteration}"
	if not iter_root.is_dir():
		return False
	return any(
		(shuffle_dir / "train" / PYTORCH_TRAIN_CONFIG).is_file()
		for shuffle_dir in iter_root.iterdir()
		if shuffle_dir.is_dir()
	)


def latest_iteration_with_pytorch_model(project_dir: Path) -> int | None:
	"""Highest iteration-N on disk that has a usable PyTorch train config."""
	root = project_dir / PYTORCH_MODELS_DIR
	if not root.is_dir():
		return None
	found: list[int] = []
	for iter_dir in root.glob("iteration-*"):
		try:
			n = int(iter_dir.name.split("-", 1)[1])
		except (IndexError, ValueError):
			continue
		if iteration_has_pytorch_model(project_dir, n):
			found.append(n)
	return max(found) if found else None


def resolve_analyze_iteration(project_dir: Path, cfg: dict) -> int:
	"""Pick iteration for analyze: config value if model exists, else latest on disk."""
	cfg_iter = int(cfg["iteration"])
	if iteration_has_pytorch_model(project_dir, cfg_iter):
		return cfg_iter
	latest = latest_iteration_with_pytorch_model(project_dir)
	if latest is not None:
		return latest
	raise FileNotFoundError(
		f"No trained PyTorch model under {project_dir / PYTORCH_MODELS_DIR}. "
		f"config.yaml iteration={cfg_iter} but no train/{PYTORCH_TRAIN_CONFIG} was found. "
		"Train the network before analyzing."
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
