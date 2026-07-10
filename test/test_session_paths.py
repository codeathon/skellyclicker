"""Session asset path existence checks."""

import json

import pytest

from skellyclicker.services.models import AppSession
from skellyclicker.services.session_paths import collect_asset_path_checks
from skellyclicker.services.session_store import SessionStore


@pytest.fixture
def fresh_store():
	return SessionStore()


def test_collect_asset_path_checks_mixed_existence(tmp_path):
	video_ok = tmp_path / "cam.mp4"
	video_ok.write_bytes(b"v")
	video_missing = str(tmp_path / "gone.mp4")
	labeled = tmp_path / "proj" / "labeled-data"
	labeled.mkdir(parents=True)

	session = AppSession(
		videos=[str(video_ok), video_missing],
		human_labels_path=str(labeled),
		machine_labels_path=str(tmp_path / "missing.csv"),
		dlc_project_path=str(tmp_path / "no_project"),
	)
	checks = {c.path: c.exists for c in collect_asset_path_checks(session)}

	assert checks[str(video_ok)] is True
	assert checks[video_missing] is False
	assert checks[str(labeled)] is True
	assert checks[str(tmp_path / "missing.csv")] is False
	assert checks[str(tmp_path / "no_project")] is False


def test_load_session_populates_asset_path_checks(fresh_store, tmp_path):
	video = tmp_path / "cam.mp4"
	video.write_bytes(b"v")
	session_data = {
		"videos": [str(video)],
		"human_labels_path": None,
		"machine_labels_path": None,
		"dlc_project_path": None,
		"tracked_point_names": ["nose"],
	}
	path = tmp_path / "session.json"
	path.write_text(json.dumps(session_data))

	fresh_store.load_session_json(str(path))
	by_path = {c.path: c.exists for c in fresh_store.session.asset_path_checks}

	assert by_path[str(video)] is True
