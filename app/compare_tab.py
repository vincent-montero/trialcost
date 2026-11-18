"""Scenario comparison: save the current design as a named scenario, compare up to three.

Scenarios live in the browser session only. All scenarios are simulated with the
same seed and number of simulations, so differences come from the designs, not
from Monte Carlo noise (common random numbers, D32).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import charts
import streamlit as st

from trialcost import editing, report
from trialcost.design import Design
from trialcost.simulate import SimulationResult

MAX_SCENARIOS = 3  # palette: the first three categorical slots validate all-pairs


def scenarios() -> dict[str, tuple[str, str]]:
    """name -> (design JSON, suggestions JSON), in insertion order."""
    return st.session_state.setdefault("scenarios", {})


def save_control(design: Design, suggestions_json: str) -> None:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        name = st.text_input(
            "Save the current design as a scenario",
            placeholder="e.g. monthly full SDV",
            key="sc_name",
        )
        save = st.button(
            "Save scenario", key="sc_save", icon=":material/bookmark_add:", disabled=not name
        )
    if save:
        scenarios()[name.strip()] = (design.model_dump_json(), suggestions_json)
        st.toast(f"Saved scenario “{name.strip()}”", icon=":material/bookmark_added:")


def compare_tab(
    simulate: Callable[[str, str], SimulationResult], eur: Callable[[float], str]
) -> None:
    saved = scenarios()
    if len(saved) < 2:
        st.info(
            "Save at least two scenarios (box above the tabs), e.g. the design as loaded, then "
            "after a change in the Design tab. The first selected scenario is the baseline.",
            icon=":material/info:",
        )
        if saved:
            st.caption("Saved: " + ", ".join(saved))
        return
    chosen = st.multiselect(
        f"Scenarios to compare (first = baseline, up to {MAX_SCENARIOS})",
        list(saved),
        default=list(saved)[:MAX_SCENARIOS],
        max_selections=MAX_SCENARIOS,
        key="sc_chosen",
    )
    if len(chosen) < 2:
        st.warning("Choose at least two scenarios.")
        return
    results = {name: simulate(*saved[name]) for name in chosen}
    table = report.compare(results)

    base_name = chosen[0]
    cols = st.columns(len(chosen))
    for col, name in zip(cols, chosen, strict=True):
        row = table.loc[name]
        delta = None if name == base_name else f"{row.delta_cost_P50_pct:+.1f} % vs {base_name}"
        col.metric(
            f"{name}: cost P50",
            eur(row.cost_P50),
            delta=delta,
            delta_color="inverse",
            border=True,
            height="stretch",
        )
        col.caption(
            f"P10 {eur(row.cost_P10)} · P90 {eur(row.cost_P90)} · duration P50 "
            f"{row.duration_P50:.1f} mo ({row.delta_duration_P50:+.1f})"
        )

    st.markdown("**Cumulative distribution of total cost**")
    st.altair_chart(
        charts.ecdf({name: results[name].total for name in chosen}),
        alt="Cumulative distribution of total cost for each compared scenario",
    )
    st.caption("Axis stops at the largest P99; probabilities count every simulated trial.")

    for name in chosen[1:]:
        st.markdown(f"**{name}** vs **{base_name}**")
        changes = editing.diff(results[base_name].design, results[name].design)
        n_a, n_b = (len(json.loads(saved[x][1])) for x in (base_name, name))
        if saved[base_name][1] != saved[name][1]:
            changes.append(f"reviewer suggestions: {n_a} → {n_b}")
        st.caption("Design changes: " + ("; ".join(changes) if changes else "none"))
        by = st.segmented_control(
            "Δ by",
            ["category", "resource", "country"],
            default="category",
            required=True,
            key=f"sc_by_{name}",
            persist_state="page",
        )
        deltas = report.category_deltas(results[base_name], results[name], by)
        deltas = deltas[deltas.abs() >= 500]  # hide rounding-level changes
        if deltas.empty:
            st.caption("No group moves by more than €500.")
        else:
            st.altair_chart(
                charts.delta_bars(deltas, "Δ mean cost"),
                height=charts.element_height(len(deltas)),
                alt=f"Change in mean cost by {by}, {name} versus {base_name}",
            )

    with st.expander("Comparison table", icon=":material/table:"):
        st.dataframe(table.round(1), alt="Cost and duration quantiles per scenario, with changes")
    if st.button("Clear saved scenarios", key="sc_clear", icon=":material/delete_sweep:"):
        st.session_state["scenarios"] = {}
        st.rerun()
