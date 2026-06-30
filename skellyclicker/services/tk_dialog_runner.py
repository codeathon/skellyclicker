"""Run tkinter file dialogs in a subprocess (safe from FastAPI worker threads)."""

from __future__ import annotations

import argparse
import json
import sys


def _filetypes(extensions: list[str]) -> list[tuple[str, str]]:
	if not extensions or extensions == ["*"]:
		return [("All files", "*.*")]
	pattern = " ".join(f"*.{ext.lstrip('.')}" for ext in extensions)
	label = extensions[0].upper() + " files"
	return [(label, pattern), ("All files", "*.*")]


def run_dialog(
	kind: str,
	title: str,
	extensions: list[str] | None = None,
	default_name: str = "",
) -> list[str]:
	"""Show one native dialog; return selected path(s), or [] if cancelled."""
	import tkinter as tk
	from tkinter import filedialog

	extensions = extensions or ["*"]
	root = tk.Tk()
	root.withdraw()
	try:
		root.attributes("-topmost", True)
	except tk.TclError:
		pass

	filetypes = _filetypes(extensions)
	paths: list[str] = []
	if kind == "file":
		selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
		if selected:
			paths = [selected]
	elif kind == "files":
		selected = filedialog.askopenfilenames(title=title, filetypes=filetypes)
		if selected:
			paths = list(selected)
	elif kind == "directory":
		selected = filedialog.askdirectory(title=title, mustexist=True)
		if selected:
			paths = [selected]
	elif kind == "save":
		ext = extensions[0] if extensions and extensions[0] != "*" else ""
		selected = filedialog.asksaveasfilename(
			title=title,
			initialfile=default_name or None,
			defaultextension=f".{ext.lstrip('.')}" if ext else None,
			filetypes=filetypes,
		)
		if selected:
			paths = [selected]
	else:
		raise ValueError(f"Unknown dialog kind: {kind}")

	root.destroy()
	return paths


def main() -> None:
	parser = argparse.ArgumentParser(description="SkellyClicker native file dialog")
	parser.add_argument("kind", choices=["file", "files", "directory", "save"])
	parser.add_argument("--title", default="Select")
	parser.add_argument(
		"--extensions",
		default="",
		help="Comma-separated extensions, e.g. mp4,avi",
	)
	parser.add_argument("--default-name", default="")
	args = parser.parse_args()
	exts = [e.strip() for e in args.extensions.split(",") if e.strip()] or ["*"]
	paths = run_dialog(args.kind, args.title, exts, args.default_name)
	print(json.dumps(paths))


if __name__ == "__main__":
	main()
