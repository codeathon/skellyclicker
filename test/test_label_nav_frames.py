"""Left-panel nav lists human-labeled frames only."""

from skellyclicker.services.label_nav_frames import build_nav_frame_list


def test_nav_human_only():
	nav = build_nav_frame_list([1, 5, 10], None)
	assert nav == [
		{"frame": 1, "kind": "human"},
		{"frame": 5, "kind": "human"},
		{"frame": 10, "kind": "human"},
	]


def test_nav_ignores_machine_and_sample_frames():
	# Predicted frames come from live scrub — not the left panel.
	nav = build_nav_frame_list(
		[5, 10],
		[10, 20, 40],
		sample_frames=[100, 200],
	)
	assert nav == [
		{"frame": 5, "kind": "human"},
		{"frame": 10, "kind": "human"},
	]


def test_nav_dedupes_and_sorts():
	nav = build_nav_frame_list([10, 1, 10, 5])
	assert [item["frame"] for item in nav] == [1, 5, 10]
