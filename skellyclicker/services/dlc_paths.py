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


def iter_machine_labels_csvs(
	project_dir: Path,
	video_paths: list[str] | None = None,
) -> list[Path]:
	"""All skellyclicker machine-label CSVs under project and video output folders."""
	seen: set[Path] = set()
	found: list[Path] = []
	roots: list[Path] = [project_dir]
	if video_paths:
		for raw in video_paths:
			parent = Path(raw).expanduser().resolve().parent
			if parent not in roots:
				roots.append(parent)
	patterns = (
		"model_outputs/**/skellyclicker_machine_labels_iteration_*.csv",
		"*_model_outputs_iteration_*/skellyclicker_machine_labels_iteration_*.csv",
	)
	for root in roots:
		for pattern in patterns:
			for path in sorted(root.glob(pattern)):
				resolved = path.expanduser().resolve()
				if resolved.is_file() and resolved not in seen:
					seen.add(resolved)
					found.append(resolved)
	return found


def machine_labels_iteration(path: Path) -> int | None:
	"""Parse iteration-N from skellyclicker_machine_labels_iteration_N.csv."""
	stem = path.name
	prefix = "skellyclicker_machine_labels_iteration_"
	if not stem.startswith(prefix) or not stem.endswith(".csv"):
		return None
	try:
		return int(stem[len(prefix) : -len(".csv")])
	except ValueError:
		return None


def latest_machine_labels_csv(
	project_dir: Path,
	video_paths: list[str] | None = None,
) -> Path | None:
	"""Newest machine-label CSV by iteration number, then file mtime."""
	candidates = iter_machine_labels_csvs(project_dir, video_paths)
	if not candidates:
		return None
	return max(
		candidates,
		key=lambda path: (
			machine_labels_iteration(path) or -1,
			path.stat().st_mtime,
		),
	)


def resolve_latest_machine_labels_path(
	project_config_path: str,
	video_paths: list[str] | None = None,
) -> Path | None:
	"""Absolute path to the latest skellyclicker machine-labels CSV, if any."""
	project_dir = dlc_project_dir(project_config_path)
	latest = latest_machine_labels_csv(project_dir, video_paths)
	return latest.resolve() if latest is not None else None


def _densest_machine_labels_csv(paths: list[Path]) -> Path | None:
	"""Prefer the largest CSV — full analyze is dense; partial-only files are small."""
	if not paths:
		return None
	return max(paths, key=lambda path: path.stat().st_size)


def resolve_partial_machine_labels_path(
	project_config_path: str,
	analyze_iter: int,
	use_training_videos: bool,
	video_paths: list[str],
	session_machine_labels_path: str | None,
) -> Path:
	"""Patch target for the current analyze iteration, seeded from the densest prior CSV."""
	import shutil

	output_folder = analyze_output_folder(
		project_config_path,
		use_training_videos,
		video_paths,
		iteration=analyze_iter,
	)
	target = output_folder / f"skellyclicker_machine_labels_iteration_{analyze_iter}.csv"
	project_dir = dlc_project_dir(project_config_path)
	candidates = iter_machine_labels_csvs(project_dir, video_paths)
	dense_base = _densest_machine_labels_csv(candidates)

	seed_source: Path | None = dense_base
	if session_machine_labels_path:
		session_path = Path(session_machine_labels_path).expanduser().resolve()
		if session_path.is_file():
			if seed_source is None or session_path.stat().st_size >= seed_source.stat().st_size:
				seed_source = session_path

	def should_seed() -> bool:
		if seed_source is None:
			return False
		if not target.is_file():
			return True
		# Full analyze CSVs are much larger than human-frame-only partial outputs.
		return target.stat().st_size < seed_source.stat().st_size * 0.5

	if should_seed():
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(seed_source, target)

	return target
