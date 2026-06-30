"""Tests for DLC model iteration resolution."""

import tempfile
from pathlib import Path

import pytest

from skellyclicker.services.dlc_paths import (
	PYTORCH_TRAIN_CONFIG,
	latest_iteration_with_pytorch_model,
	resolve_analyze_iteration,
)


def _make_model_tree(project: Path, iteration: int) -> None:
	shuffle = project / "dlc-models-pytorch" / f"iteration-{iteration}" / "MyTask_trainset95shuffle1"
	(shuffle / "train").mkdir(parents=True)
	(shuffle / "train" / PYTORCH_TRAIN_CONFIG).write_text("method: bu\n")


def test_resolve_uses_config_iteration_when_model_exists():
	with tempfile.TemporaryDirectory() as tmp:
		project = Path(tmp)
		_make_model_tree(project, 3)
		cfg = {"iteration": 3}
		assert resolve_analyze_iteration(project, cfg) == 3


def test_resolve_falls_back_to_latest_on_disk():
	with tempfile.TemporaryDirectory() as tmp:
		project = Path(tmp)
		_make_model_tree(project, 2)
		_make_model_tree(project, 5)
		cfg = {"iteration": 9}
		assert resolve_analyze_iteration(project, cfg) == 5


def test_resolve_raises_when_no_models():
	with tempfile.TemporaryDirectory() as tmp:
		project = Path(tmp)
		with pytest.raises(FileNotFoundError):
			resolve_analyze_iteration(project, {"iteration": 9})


def test_latest_iteration_scan():
	with tempfile.TemporaryDirectory() as tmp:
		project = Path(tmp)
		_make_model_tree(project, 1)
		_make_model_tree(project, 4)
		assert latest_iteration_with_pytorch_model(project) == 4
