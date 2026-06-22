"""Native OS file dialogs for the local web UI (server-side paths)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from skellyclicker.services.dialog_errors import DialogCancelled, DialogUnavailable

# One dialog at a time — concurrent picks confuse the window server.
_dialog_lock = threading.Lock()

_RUNNER = Path(__file__).with_name("tk_dialog_runner.py")


def check_dialog_availability() -> tuple[bool, str]:
	"""Probe whether a native picker can open (zenity on Linux, else tkinter)."""
	if sys.platform.startswith("linux"):
		has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
		if not has_display:
			return (
				False,
				"No DISPLAY or WAYLAND_DISPLAY — file dialogs need a graphical session "
				"(local Ubuntu desktop or SSH with X11 forwarding).",
			)

		from skellyclicker.services.zenity_dialog import zenity_available

		if zenity_available():
			return True, "Native file dialogs are available (zenity)."

	try:
		import tkinter as tk
	except ImportError:
		return (
			False,
			"Install dialog support on Ubuntu: sudo apt install zenity python3-tk",
		)

	try:
		root = tk.Tk()
		root.withdraw()
		root.update_idletasks()
		root.destroy()
	except tk.TclError as exc:
		return False, f"tkinter cannot connect to a display: {exc}"

	return True, "Native file dialogs are available (tkinter)."


def dialog_startup_warning() -> str | None:
	"""Human-readable warning for server logs when dialogs are unavailable."""
	available, detail = check_dialog_availability()
	if available:
		return None
	return (
		"SkellyClicker native file dialogs are unavailable on the server. "
		"The web UI will use the browser file picker instead. "
		f"Reason: {detail}"
	)


def _spawn_tk_dialog(
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
			env=os.environ,
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


def _spawn_dialog(
	kind: str,
	title: str,
	extensions: list[str] | None = None,
	default_name: str = "",
) -> list[str]:
	# Prefer zenity on Linux — more reliable than tkinter under uvicorn.
	if sys.platform.startswith("linux"):
		from skellyclicker.services.zenity_dialog import zenity_available, zenity_spawn

		if zenity_available():
			return zenity_spawn(kind, title, extensions, default_name)

	return _spawn_tk_dialog(kind, title, extensions, default_name)


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
