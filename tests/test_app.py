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


def test_results_tab_has_charts_and_dropout(app):
    assert len(app.get("vega_lite_chart")) >= 3  # two histograms + breakdown
    assert any(m.label == "Cost per completer (P50)" for m in app.metric)


def test_breakdown_switch_and_tornado(app):
    app.radio(key="breakdown_by").set_value("country").run()
    assert not app.exception
    app.button(key="tornado_button").click().run()
    assert not app.exception
    assert len(app.get("vega_lite_chart")) >= 4


def test_editing_n_changes_result(app):
    before = app.metric[0].value
    app.number_input(key="ed_rare-neuro-ph2-mri-scales_n").set_value(100).run()
    assert not app.exception
    assert app.metric[0].value != before
    assert any("Edited from" in c.value for c in app.caption)


def test_operations_toggle_and_preset(app):
    app.toggle(key="ed_rare-neuro-ph2-mri-scales_dmc").set_value(False).run()
    assert not app.exception
    app.selectbox(key="ed_rare-neuro-ph2-mri-scales_preset").select(
        "rare_neuro, phase=PHASE3"
    ).run()
    assert not app.exception


@pytest.mark.parametrize("name", ["rare-neuro-ph2-mri-scales", "ref-cns-ph2-typical"])
def test_untouched_editor_leaves_design_unchanged(name):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.selectbox[0].select(name).run()
    assert not at.exception
    assert not any("Edited from" in c.value for c in at.caption)
