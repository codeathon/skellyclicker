"""GTK file dialogs via zenity (preferred on Ubuntu desktop)."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Literal

from skellyclicker.services.dialog_errors import DialogUnavailable

DialogKind = Literal["file", "files", "directory", "save"]


def zenity_available() -> bool:
	"""True when the zenity binary is on PATH."""
	return shutil.which("zenity") is not None


def _file_filters(extensions: list[str]) -> list[str]:
	args: list[str] = []
	for ext in extensions:
		if ext == "*":
			continue
		clean = ext.lstrip(".")
		args.extend(["--file-filter", f"{clean.upper()} files | *.{clean}"])
	args.extend(["--file-filter", "All files | *"])
	return args


def _run_zenity(args: list[str]) -> str:
	try:
		completed = subprocess.run(
			["zenity", *args],
			capture_output=True,
			text=True,
			check=False,
			timeout=600,
			env=os.environ,
		)
	except (OSError, subprocess.TimeoutExpired) as exc:
		raise DialogUnavailable("zenity could not run") from exc

	# 1 = user cancelled; 0 = OK.
	if completed.returncode == 1:
		return ""
	if completed.returncode != 0:
		err = (completed.stderr or completed.stdout or "").strip()
		raise DialogUnavailable(err or "zenity file dialog failed")
	return completed.stdout.strip()


def zenity_spawn(
	kind: DialogKind,
	title: str,
	extensions: list[str] | None = None,
	default_name: str = "",
) -> list[str]:
	"""Show a zenity picker; return absolute path(s) or [] when cancelled."""
	extensions = extensions or ["*"]
	args = ["--title", title]

	if kind == "file":
		args.append("--file-selection")
		args.extend(_file_filters(extensions))
	elif kind == "files":
		args.extend(["--file-selection", "--multiple", "--separator", "|"])
		args.extend(_file_filters(extensions))
	elif kind == "directory":
		args.extend(["--file-selection", "--directory"])
	elif kind == "save":
		args.extend(["--file-selection", "--save", "--confirm-overwrite"])
		if default_name:
			args.extend(["--filename", default_name])
		args.extend(_file_filters(extensions))
	else:
		raise ValueError(f"Unknown dialog kind: {kind}")

	selected = _run_zenity(args)
	if not selected:
		return []
	if kind == "files":
		return [part for part in selected.split("|") if part]
	return [selected]
