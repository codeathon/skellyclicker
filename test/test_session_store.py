"""Tests for web session store lifecycle (Phase A regressions)."""

import pytest

from skellyclicker.services.session_store import SessionStore


@pytest.fixture
def fresh_store():
	return SessionStore()


def test_clear_session_resets_dlc_handler(fresh_store):
	# Simulate a loaded handler without full DLC init.
	fresh_store.dlc_handler = object()  # type: ignore[assignment]
	fresh_store.clear_session()
	assert fresh_store.dlc_handler is None
	assert fresh_store.labeling_engine is None


def test_train_on_machine_requires_csv():
	from skellyclicker.services.session_store import SessionStore
	store = SessionStore()
	store.session.train_on_machine_labels = True
	assert store.session.machine_labels_path is None


def test_bump_generation_on_teardown(fresh_store):
	gen = fresh_store.session.generation
	fresh_store._teardown_all()
	assert fresh_store.session.generation == gen + 1
