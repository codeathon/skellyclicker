"""Validate session asset paths exist on the server filesystem."""

from pathlib import Path

from skellyclicker.services.dlc_paths import resolve_dlc_project_input
from skellyclicker.services.models import AppSession, AssetPathCheck


def _dlc_project_exists(path: str) -> bool:
	try:
		resolve_dlc_project_input(path)
		return True
	except ValueError:
		return False


def collect_asset_path_checks(session: AppSession) -> list[AssetPathCheck]:
	"""Build existence checks for every file/folder path stored on the session."""
	checks: list[AssetPathCheck] = []

	for video_path in session.videos or []:
		checks.append(
			AssetPathCheck(
				kind="video",
				path=video_path,
				exists=Path(video_path).expanduser().is_file(),
			)
		)

	if session.human_labels_path:
		checks.append(
			AssetPathCheck(
				kind="human_labels",
				path=session.human_labels_path,
				exists=Path(session.human_labels_path).expanduser().is_file(),
			)
		)

	if session.machine_labels_path:
		checks.append(
			AssetPathCheck(
				kind="machine_labels",
				path=session.machine_labels_path,
				exists=Path(session.machine_labels_path).expanduser().is_file(),
			)
		)

	if session.dlc_project_path:
		checks.append(
			AssetPathCheck(
				kind="dlc_project",
				path=session.dlc_project_path,
				exists=_dlc_project_exists(session.dlc_project_path),
			)
		)

	return checks
