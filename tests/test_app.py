"""Headless smoke tests of the Streamlit app (skipped if the `ui` extra is missing)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_runs_without_exception(app):
    assert not app.exception


def test_headline_shows_cost(app):
    labels = [m.label for m in app.metric]
    assert "Total cost (P50)" in labels
    assert app.metric[0].value.startswith("€")


def test_switching_design_changes_result(app):
    first = app.metric[0].value
    app.sidebar.selectbox[0].select("ref-cns-ph3-typical").run()
    assert not app.exception
    assert app.metric[0].value != first
