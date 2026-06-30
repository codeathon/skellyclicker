"""Tests for browser upload staging on the server."""

from __future__ import annotations

from skellyclicker.services.upload_store import UPLOAD_ROOT, save_upload


def test_save_upload_writes_under_session_dir(tmp_path, monkeypatch):
	monkeypatch.setattr(
		"skellyclicker.services.upload_store.UPLOAD_ROOT",
		tmp_path,
	)
	path = save_upload("sess-1", "labels.csv", b"x,y\n1,2")
	assert path == str((tmp_path / "sess-1" / "labels.csv").resolve())
	assert (tmp_path / "sess-1" / "labels.csv").read_bytes() == b"x,y\n1,2"
