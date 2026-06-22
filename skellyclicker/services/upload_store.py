"""Save browser-uploaded files to server disk for the active session."""

from __future__ import annotations

from pathlib import Path

UPLOAD_ROOT = Path.home() / "skellyclicker_uploads"


def save_upload(session_id: str, filename: str, content: bytes) -> str:
	"""Write upload under ~/skellyclicker_uploads/<session_id>/; return absolute path."""
	safe_name = Path(filename).name
	if not safe_name:
		raise ValueError("Upload filename is empty")

	dest_dir = UPLOAD_ROOT / session_id
	dest_dir.mkdir(parents=True, exist_ok=True)
	dest = dest_dir / safe_name
	dest.write_bytes(content)
	return str(dest.resolve())
