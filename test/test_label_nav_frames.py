"""Tests for labeler left-panel frame navigation list."""

from skellyclicker import MAX_NAV_MACHINE_FRAMES
from skellyclicker.services.label_nav_frames import build_nav_frame_list


def test_nav_human_only():
	nav = build_nav_frame_list([1, 5, 10], None)
	assert nav == [
		{"frame": 1, "kind": "human"},
		{"frame": 5, "kind": "human"},
		{"frame": 10, "kind": "human"},
	]


def test_nav_machine_only_sparse():
	nav = build_nav_frame_list([], [20, 40, 60])
	assert nav == [
		{"frame": 20, "kind": "machine"},
		{"frame": 40, "kind": "machine"},
		{"frame": 60, "kind": "machine"},
	]


def test_nav_merges_overlap_as_both():
	nav = build_nav_frame_list([5, 10], [10, 20])
	assert nav == [
		{"frame": 5, "kind": "human"},
		{"frame": 10, "kind": "both"},
		{"frame": 20, "kind": "machine"},
	]


def test_nav_ignores_dense_machine_csv():
	dense = list(range(MAX_NAV_MACHINE_FRAMES + 1))
	nav = build_nav_frame_list([1, 2], dense)
	assert nav == [
		{"frame": 1, "kind": "human"},
		{"frame": 2, "kind": "human"},
	]
