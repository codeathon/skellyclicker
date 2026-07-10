"""Tests for /api/health debug snapshot (read-only, no session mutation)."""

from pathlib import Path

from skellyclicker.services.health_debug import CODE_STAMP, build_health_debug
from skellyclicker.services.session_store import SessionStore


def test_health_debug_includes_code_stamp_and_session(tmp_path: Path):
	store = SessionStore()
	store.session.tracked_point_names = ["nose"]
	payload = build_health_debug(store, repo_root=tmp_path)
	assert payload["ok"] is True
	assert payload["code_stamp"] == CODE_STAMP
	assert payload["session"]["tracked_point_names"] == ["nose"]
	assert payload["machine_csv_leftovers_on_disk"] == []
	assert "process" in payload
	assert payload["labeling"]["labeler_open"] is False


def test_health_debug_lists_leftover_csvs_without_attaching(tmp_path: Path):
	"""Leftovers are reported for diagnosis; session.machine_labels_path stays unset."""
	project = tmp_path / "proj"
	project.mkdir()
	(project / "config.yaml").write_text("Task: t\niteration: 0\n")
	csv = (
		project
		/ "model_outputs"
		/ "model_outputs_iteration_0"
		/ "skellyclicker_machine_labels_iteration_0.csv"
	)
	csv.parent.mkdir(parents=True)
	csv.write_text("video,frame,nose_x,nose_y\n")

	store = SessionStore()
	store.session.dlc_project_path = str(project)
	assert store.session.machine_labels_path is None

	payload = build_health_debug(store, repo_root=tmp_path)
	assert store.session.machine_labels_path is None
	paths = [row["path"] for row in payload["machine_csv_leftovers_on_disk"]]
	assert str(csv.resolve()) in paths or str(csv) in paths
