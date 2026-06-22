"""Native OS file dialogs for the local web UI (server-side paths)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

# One dialog at a time — concurrent picks confuse the window server.
_dialog_lock = threading.Lock()

_RUNNER = Path(__file__).with_name("tk_dialog_runner.py")


class DialogCancelled(Exception):
	"""User closed the file dialog without selecting."""


class DialogUnavailable(Exception):
	"""No display / tkinter available for native dialogs."""


def _spawn_dialog(
	kind: str,
	title: str,
	extensions: list[str] | None = None,
	default_name: str = "",
) -> list[str]:
	ext_arg = ",".join(extensions) if extensions else ""
	cmd = [
		sys.executable,
		str(_RUNNER),
		kind,
		"--title",
		title,
		"--extensions",
		ext_arg,
	]
	if default_name:
		cmd.extend(["--default-name", default_name])

	try:
		completed = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			check=False,
			timeout=600,
		)
	except (OSError, subprocess.TimeoutExpired) as exc:
		raise DialogUnavailable(
			"Native file dialog is not available on this machine."
		) from exc

	if completed.returncode != 0:
		err = (completed.stderr or completed.stdout or "").strip()
		raise DialogUnavailable(err or "Native file dialog failed")

	try:
		paths = json.loads(completed.stdout.strip() or "[]")
	except json.JSONDecodeError as exc:
		raise DialogUnavailable("Native file dialog returned invalid output") from exc

	if not isinstance(paths, list):
		raise DialogUnavailable("Native file dialog returned unexpected output")
	return [str(p) for p in paths if p]


def pick_file(title: str, extensions: list[str]) -> str:
	with _dialog_lock:
		paths = _spawn_dialog("file", title, extensions)
	if not paths:
		raise DialogCancelled()
	return paths[0]


def pick_files(title: str, extensions: list[str]) -> list[str]:
	with _dialog_lock:
		paths = _spawn_dialog("files", title, extensions)
	if not paths:
		raise DialogCancelled()
	return paths


def pick_directory(title: str) -> str:
	with _dialog_lock:
		paths = _spawn_dialog("directory", title, None)
	if not paths:
		raise DialogCancelled()
	return paths[0]


def save_file(title: str, extensions: list[str], default_name: str = "") -> str:
	with _dialog_lock:
		paths = _spawn_dialog("save", title, extensions, default_name=default_name)
	if not paths:
		raise DialogCancelled()
	return paths[0]
