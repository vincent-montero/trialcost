"""Streamlit MVP of the clinical trial cost estimator (README §8 roadmap #6).

    uv run --extra ui streamlit run app/streamlit_app.py

UI only: every number comes from the engine (``src/trialcost``) and the
catalogue (``data/``). Nothing here writes to ``data/``.
"""

from __future__ import annotations

from pathlib import Path

import charts
import numpy as np
import pandas as pd
import streamlit as st
from compare_tab import compare_tab, save_control
from design_editor import design_editor

from trialcost import report
from trialcost.catalogue import Catalogue, load_catalogue
from trialcost.design import Design, load_design
from trialcost.simulate import SimulationResult, run

ROOT = Path(__file__).resolve().parents[1]
DESIGNS_DIR = ROOT / "designs"

st.set_page_config(page_title="Clinical trial cost estimator", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------- #
# Data access (cached)
# --------------------------------------------------------------------------- #


@st.cache_resource
def base_catalogue() -> Catalogue:
    return load_catalogue()


def design_files() -> dict[str, Path]:
    """Archetypes first, then the reference designs used for benchmark validation."""
    files = sorted(DESIGNS_DIR.glob("*.yaml"), key=lambda p: (p.stem.startswith("ref-"), p.stem))
    return {p.stem: p for p in files}


@st.cache_data(show_spinner="Simulating…", max_entries=32)
def simulate(design_json: str, n_sims: int, seed: int) -> SimulationResult:
    """Cached on the design (as JSON), number of simulations and seed."""
    return run(Design.model_validate_json(design_json), base_catalogue(), n_sims, seed)


@st.cache_data(show_spinner="Comparing with the same design without dropout…", max_entries=16)
def dropout_table(design_json: str, n_sims: int, seed: int) -> pd.DataFrame:
    design = Design.model_validate_json(design_json)
    return report.dropout_report(design, base_catalogue(), n_sims, seed)


@st.cache_data(show_spinner="Running the tornado (about 1 s)…", max_entries=16)
def tornado_table(design_json: str, seed: int, n_sims: int = 2_000) -> pd.DataFrame:
    design = Design.model_validate_json(design_json)
    return report.tornado(design, base_catalogue(), n_sims, seed)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def eur(x: float) -> str:
    return f"€{x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"€{x / 1e3:,.0f}k"


def quantiles(x: np.ndarray) -> tuple[float, float, float]:
    p10, p50, p90 = np.quantile(x, [0.1, 0.5, 0.9])
    return float(p10), float(p50), float(p90)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def sidebar() -> tuple[Design, str, int, int, float | None]:
    st.sidebar.header("Design")
    files = design_files()
    name = st.sidebar.selectbox(
        "Start from",
        list(files),
        help="Archetypes first; `ref-*` are reference designs used for benchmark validation.",
    )
    design = load_design(files[name])
    st.sidebar.header("Simulation")
    n_sims = st.sidebar.select_slider("Simulations", [1_000, 2_000, 5_000, 10_000], value=5_000)
    seed = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)
    budget = st.sidebar.number_input(
        "Budget to test (€, 0 = none)", min_value=0.0, value=0.0, step=250_000.0, format="%.0f"
    )
    return design, name, int(n_sims), int(seed), (budget or None)


def headline(result: SimulationResult, budget: float | None) -> None:
    d = result.design
    st.subheader(f"{d.name}: N = {d.n_randomised}, {d.n_sites} sites, {len(d.countries)} countries")
    cost, dur, lpi = (
        quantiles(x) for x in (result.total, result.duration, result.quantities.t_lpi)
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cost (P50)", eur(cost[1]), help=f"P10 {eur(cost[0])} · P90 {eur(cost[2])}")
    c2.metric("Duration (P50)", f"{dur[1]:.1f} months", help=f"P10 {dur[0]:.1f} · P90 {dur[2]:.1f}")
    c3.metric(
        "Recruitment LPI (P50)", f"{lpi[1]:.1f} months", help=f"P10 {lpi[0]:.1f} · P90 {lpi[2]:.1f}"
    )
    c4.metric("Cost per randomised patient (P50)", eur(cost[1] / d.n_randomised))
    st.caption(
        f"Total cost P10 {eur(cost[0])} · P50 {eur(cost[1])} · P90 {eur(cost[2])} — "
        f"EUR 2026, {result.n_sims:,} simulated trials."
    )
    if budget:
        st.info(f"P(total cost ≤ {eur(budget)}) = {np.mean(result.total <= budget):.0%}")


def results_tab(result: SimulationResult, design_json: str, n_sims: int, seed: int) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Total cost** (EUR 2026)")
        st.altair_chart(charts.histogram(result.total, "Total cost"), width="stretch")
    with c2:
        st.markdown("**Duration**, study start to CSR (months)")
        st.altair_chart(
            charts.histogram(result.duration, "Duration (months)", money=False),
            width="stretch",
        )

    st.markdown("#### Where the money goes")
    by = st.radio(
        "Break down by", ["category", "resource", "country"], horizontal=True, key="breakdown_by"
    )
    table = report.breakdown(result, by)
    st.caption("Bar = mean cost; whisker = P10-P90 across simulated trials.")
    st.altair_chart(charts.breakdown(table, by), width="stretch")
    with st.expander("Table"):
        st.dataframe(
            table.assign(share=100 * table["share"]).round(0),
            column_config={"share": st.column_config.NumberColumn("share %", format="%.1f")},
        )

    st.markdown("#### Cost of dropout")
    dp = dropout_table(design_json, n_sims, seed)
    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Completers (P50)",
        f"{dp.loc['completers', 'P50']:.1f} of {result.design.n_randomised}",
    )
    d2.metric(
        "Cost per completer (P50)",
        eur(dp.loc["cost_per_completer_eur", "P50"]),
        delta=f"+{dp.loc['completer_cost_premium_pct', 'P50']:.1f} % vs no dropout",
        delta_color="inverse",
    )
    d3.metric(
        "Total cost change due to dropout (P50)",
        eur(dp.loc["dropout_delta_cost_eur", "P50"]),
        help="Dropouts stop generating visit costs; fixed and time-driven costs stay.",
    )

    st.markdown("#### Which assumptions matter most")
    st.caption(
        "One-way sensitivity on the mean total cost: each input moves to its P10 then P90 "
        "(0.5x / 1.5x if it is fixed) while the others keep their uncertainty."
    )
    if st.button("Compute tornado", key="tornado_button") or st.session_state.get("tornado_on"):
        st.session_state["tornado_on"] = True
        tt = tornado_table(design_json, seed)
        st.altair_chart(charts.tornado(tt), width="stretch")
        with st.expander("Table"):
            st.dataframe(tt.drop(columns=["base"]).round(0))


def main() -> None:
    st.title("Clinical trial cost estimator")
    st.caption(
        "Design-driven Monte Carlo estimate of a trial's total sponsor budget. Personal project, "
        "public data only. Many unit costs are expert-judgement placeholders: read results as "
        "orders of magnitude with honest uncertainty, not quotes."
    )
    base, base_name, n_sims, seed, budget = sidebar()
    header = st.container()  # filled after the editor has run
    tab_results, tab_design, tab_compare = st.tabs(["Results", "Design", "Compare scenarios"])
    with tab_design:
        design = design_editor(base, base_name, base_catalogue())
    design_json = design.model_dump_json()
    result = simulate(design_json, n_sims, seed)
    with header:
        headline(result, budget)
        if design != base:
            st.caption(f"Edited from `{base_name}` (see the Design tab).")
        save_control(design)
    with tab_results:
        results_tab(result, design_json, n_sims, seed)
    with tab_compare:
        compare_tab(lambda dj: simulate(dj, n_sims, seed), eur)


main()
