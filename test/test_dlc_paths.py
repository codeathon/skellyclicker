"""Tests for DLC path resolution."""

import tempfile
from pathlib import Path

import pytest

from skellyclicker.services.dlc_paths import resolve_dlc_project_input


def test_resolve_project_directory():
	with tempfile.TemporaryDirectory() as tmp:
		root = Path(tmp)
		(root / "config.yaml").write_text("Task: test\n")
		project_dir, config_path = resolve_dlc_project_input(str(root))
		assert project_dir == root.resolve()
		assert config_path == (root / "config.yaml").resolve()


def test_resolve_config_file_path():
	with tempfile.TemporaryDirectory() as tmp:
		root = Path(tmp)
		cfg = root / "config.yaml"
		cfg.write_text("Task: test\n")
		project_dir, config_path = resolve_dlc_project_input(str(cfg))
		assert project_dir == root.resolve()
		assert config_path == cfg.resolve()


def test_resolve_rejects_missing_config():
	with tempfile.TemporaryDirectory() as tmp:
		with pytest.raises(ValueError):
			resolve_dlc_project_input(tmp)
