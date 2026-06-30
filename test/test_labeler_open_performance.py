"""Labeler open should not parse dense machine-label CSVs up front."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from skellyclicker.core.click_data_handler.data_handler import DataHandler
from skellyclicker.core.video_handler.video_handler import VideoHandler


def test_from_videos_defers_machine_label_csv_parse(tmp_path):
	video = tmp_path / "cam0.mp4"
	video.write_bytes(b"\x00" * 64)
	human = tmp_path / "human.csv"
	human.write_text("video,frame,nose_x,nose_y\ncam0.mp4,0,1.0,2.0\n")
	machine = tmp_path / "machine.csv"
	machine.write_text("video,frame,nose_x,nose_y\ncam0.mp4,0,3.0,4.0\n")

	mock_cap = MagicMock()
	mock_cap.isOpened.return_value = True
	mock_cap.get.side_effect = lambda prop: {3: 64, 4: 48, 7: 10}.get(prop, 0)

	with patch("cv2.VideoCapture", return_value=mock_cap):
		handler = VideoHandler.from_videos(
			video_paths=[str(video)],
			max_window_size=(640, 480),
			data_handler_path=str(human),
			tracked_point_names=["nose"],
			machine_labels_path=str(machine),
		)

	assert handler.machine_labels_path == str(machine)
	assert handler.machine_labels_handler is None

	with patch.object(DataHandler, "from_csv_overlay") as overlay_mock:
		overlay_mock.return_value = MagicMock()
		handler.ensure_machine_labels_loaded()
		overlay_mock.assert_called_once()
