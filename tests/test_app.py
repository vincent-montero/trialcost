"""Headless smoke tests of the Streamlit app (skipped if the `ui` extra is missing)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
TABS = {
    "results": ":material/insights: Results",
    "compare": ":material/compare_arrows: Compare scenarios",
    "method": ":material/menu_book: Method & caveats",
}


def open_tab(at, name):
    """Tabs track state (on_change="rerun"): select one through its key, then rerun."""
    at.session_state["main_tabs"] = TABS[name]
    return at.run()


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
    app.segmented_control(key="breakdown_by").set_value("country").run()
    assert not app.exception
    app.toggle(key="tornado_on").set_value(True).run()
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


def test_scenario_comparison(app):
    app.text_input(key="sc_name").input("as loaded").run()
    app.button(key="sc_save").click().run()
    app.toggle(key="ed_rare-neuro-ph2-mri-scales_dmc").set_value(False).run()
    app.text_input(key="sc_name").input("no DMC").run()
    app.button(key="sc_save").click().run()
    assert not app.exception
    open_tab(app, "compare")
    labels = [m.label for m in app.metric]
    assert "no DMC: cost P50" in labels
    delta = next(m for m in app.metric if m.label == "no DMC: cost P50").delta
    assert delta.startswith("-")
    assert any("operations.dmc: True → False" in c.value for c in app.caption)


def test_suggestion_flow(app):
    before = app.metric[0].value
    app.text_input(key="sg_reviewer").input("CRA tester").run()
    assert any(b.label == "Add my changes as suggestions" for b in app.button)
    # Editing a data_editor is not scriptable in AppTest: add the suggestion via session state
    from trialcost import suggestions as sg

    app.session_state["suggestions"] = sg.merge(
        sg.empty(),
        [
            sg.make(
                "study_items",
                "project_management",
                "mode",
                12000,
                20000,
                "PM effort higher",
                "CRA tester",
            )
        ],
    )
    app.run()
    assert not app.exception
    assert app.metric[0].value != before
    assert any("reviewer suggestion(s) active" in (w.proto.title or "") for w in app.warning)
    assert any(b.label == "Download suggestions (CSV)" for b in app.get("download_button"))


def test_method_tab_grade_shares_sum_to_100(app):
    open_tab(app, "method")
    shares = [m for m in app.metric if m.label[:2] in ("A ", "B ", "C ", "D ")]
    assert len(shares) == 4
    total = sum(float(m.value.rstrip(" %")) for m in shares)
    assert 98 <= total <= 102  # rounding


def test_closed_tabs_do_not_compute(app):
    """Compare and Method bodies only run when open (heavy work guarded by .open)."""
    assert not any(m.label.endswith("cost P50") for m in app.metric)
    assert not any(m.label.startswith("D ") for m in app.metric)
