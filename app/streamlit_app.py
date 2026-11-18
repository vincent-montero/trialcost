"""Streamlit MVP of the clinical trial cost estimator (README §8 roadmap #6).

    uv run --extra ui streamlit run app/streamlit_app.py

UI only: every number comes from the engine (``src/trialcost``) and the
catalogue (``data/``). Nothing here writes to ``data/``.
"""

from __future__ import annotations

import io
from pathlib import Path

import charts
import numpy as np
import pandas as pd
import streamlit as st
from assumptions_tab import active, active_json, assumptions_tab
from compare_tab import compare_tab, save_control
from design_editor import design_editor
from method_tab import method_tab

from trialcost import report
from trialcost import suggestions as sg
from trialcost.catalogue import Catalogue, load_catalogue
from trialcost.design import Design, load_design
from trialcost.simulate import SimulationResult, run

ROOT = Path(__file__).resolve().parents[1]
DESIGNS_DIR = ROOT / "designs"

st.set_page_config(
    page_title="Clinical trial cost estimator", page_icon=":material/science:", layout="wide"
)


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


@st.cache_resource(max_entries=32)
def session_catalogue(suggestions_json: str) -> Catalogue:
    """Published catalogue with this session's reviewer suggestions applied (D40)."""
    df = pd.read_json(io.StringIO(suggestions_json), orient="records", dtype={"row_id": str})
    if df.empty:
        return base_catalogue()
    return sg.apply_to_catalogue(base_catalogue(), df)


@st.cache_data(show_spinner="Simulating…", max_entries=32)
def simulate(design_json: str, suggestions_json: str, n_sims: int, seed: int) -> SimulationResult:
    """Cached on design and suggestions (as JSON), number of simulations and seed."""
    design = Design.model_validate_json(design_json)
    return run(design, session_catalogue(suggestions_json), n_sims, seed)


@st.cache_data(show_spinner="Comparing with the same design without dropout…", max_entries=16)
def dropout_table(design_json: str, suggestions_json: str, n_sims: int, seed: int) -> pd.DataFrame:
    design = Design.model_validate_json(design_json)
    return report.dropout_report(design, session_catalogue(suggestions_json), n_sims, seed)


@st.cache_data(show_spinner="Running the tornado (about 1 s)…", max_entries=16)
def tornado_table(
    design_json: str, suggestions_json: str, seed: int, n_sims: int = 2_000
) -> pd.DataFrame:
    design = Design.model_validate_json(design_json)
    return report.tornado(design, session_catalogue(suggestions_json), n_sims, seed)


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
    with st.sidebar:
        st.header("Design", anchor=False)
        files = design_files()
        name = st.selectbox(
            "Start from",
            list(files),
            key="design_file",
            help="Archetypes first; `ref-*` are reference designs used for benchmark validation.",
        )
        st.header("Simulation", anchor=False)
        n_sims = st.select_slider(
            "Simulated trials", [1_000, 2_000, 5_000, 10_000], value=5_000, key="n_sims"
        )
        seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="seed")
        budget = st.number_input(
            "Budget to test (€, 0 = none)",
            min_value=0.0,
            value=0.0,
            step=250_000.0,
            format="%.0f",
            key="budget",
        )
    return load_design(files[name]), name, int(n_sims), int(seed), (budget or None)


def headline(result: SimulationResult, budget: float | None) -> None:
    d = result.design
    st.subheader(
        f"{d.name}: N = {d.n_randomised}, {d.n_sites} sites, {len(d.countries)} countries",
        anchor=False,
    )
    cost, dur, lpi = (
        quantiles(x) for x in (result.total, result.duration, result.quantities.t_lpi)
    )
    cards = [
        ("Total cost (P50)", eur(cost[1]), f"P10 {eur(cost[0])} · P90 {eur(cost[2])}"),
        ("Duration (P50)", f"{dur[1]:.1f} months", f"P10 {dur[0]:.1f} · P90 {dur[2]:.1f}"),
        ("Recruitment LPI (P50)", f"{lpi[1]:.1f} months", f"P10 {lpi[0]:.1f} · P90 {lpi[2]:.1f}"),
        ("Cost per randomised patient (P50)", eur(cost[1] / d.n_randomised), None),
    ]
    if budget:
        cards.append((f"P(cost ≤ {eur(budget)})", f"{np.mean(result.total <= budget):.0%}", None))
    for col, (label, value, help_text) in zip(st.columns(len(cards)), cards, strict=True):
        col.metric(label, value, help=help_text, border=True, height="stretch")
    st.caption(
        f"Total cost P10 {eur(cost[0])} · P50 {eur(cost[1])} · P90 {eur(cost[2])}. "
        f"EUR 2026, {result.n_sims:,} simulated trials."
    )


def results_tab(
    result: SimulationResult, design_json: str, suggestions_json: str, n_sims: int, seed: int
) -> None:
    cost_col, duration_col = st.columns(2)
    with cost_col:
        st.markdown("**Total cost** (EUR 2026)")
        st.altair_chart(
            charts.histogram(result.total, "Total cost"),
            alt="Distribution of simulated total trial cost with P10, P50 and P90 markers",
        )
    with duration_col:
        st.markdown("**Duration**, study start to CSR (months)")
        st.altair_chart(
            charts.histogram(result.duration, "Duration (months)", money=False),
            alt="Distribution of simulated trial duration with P10, P50 and P90 markers",
        )

    st.subheader("Where the money goes", anchor=False)
    by = st.segmented_control(
        "Break down by",
        ["category", "resource", "country"],
        default="category",
        required=True,
        key="breakdown_by",
        persist_state="page",
    )
    table = report.breakdown(result, by)
    st.caption("Bar = mean cost; whisker = P10-P90 across simulated trials.")
    st.altair_chart(
        charts.breakdown(table, by), alt=f"Mean cost by {by}, largest first, with P10-P90 range"
    )
    with st.expander("Table", icon=":material/table:"):
        st.dataframe(
            table.assign(share=100 * table["share"]).round(0),
            column_config={"share": st.column_config.NumberColumn("share %", format="%.1f")},
            alt=f"Cost by {by}: P10, P50, P90, mean and share of the total",
        )

    st.subheader("Cost of dropout", anchor=False)
    dp = dropout_table(design_json, suggestions_json, n_sims, seed)
    premium = dp.loc["completer_cost_premium_pct", "P50"]
    cards = [
        (
            "Completers (P50)",
            f"{dp.loc['completers', 'P50']:.1f} of {result.design.n_randomised}",
            None,
        ),
        (
            "Cost per completer (P50)",
            eur(dp.loc["cost_per_completer_eur", "P50"]),
            f"+{premium:.1f} % vs the same design without dropout",
        ),
        (
            "Total cost change due to dropout (P50)",
            eur(dp.loc["dropout_delta_cost_eur", "P50"]),
            "Dropouts stop generating visit costs; fixed and time-driven costs stay.",
        ),
    ]
    for col, (label, value, help_text) in zip(st.columns(3), cards, strict=True):
        col.metric(label, value, help=help_text, border=True, height="stretch")

    st.subheader("Which assumptions matter most", anchor=False)
    st.caption(
        "One-way sensitivity on the mean total cost: each input moves to its P10 then P90 "
        "(0.5x / 1.5x if it is fixed) while the others keep their uncertainty. About 1 s."
    )
    if st.toggle("Show tornado", key="tornado_on", persist_state="page"):
        tt = tornado_table(design_json, suggestions_json, seed)
        st.altair_chart(
            charts.tornado(tt),
            alt=f"Tornado: {tt['parameter'].iloc[0]} moves the mean total cost most",
        )
        with st.expander("Table", icon=":material/table:"):
            st.dataframe(
                tt.drop(columns=["base"]).round(0),
                alt="Tornado inputs, mean total cost at low and high values, and swing",
            )


def main() -> None:
    st.title("Clinical trial cost estimator", anchor=False)
    st.caption(
        "Design-driven Monte Carlo estimate of a trial's total sponsor budget. Personal project, "
        "public data only. Many unit costs are expert-judgement placeholders: read results as "
        "orders of magnitude with honest uncertainty, not quotes."
    )
    base, base_name, n_sims, seed, budget = sidebar()
    header = st.container()  # filled once the design editor has run
    tab_results, tab_design, tab_compare, tab_assumptions, tab_method = st.tabs(
        [
            ":material/insights: Results",
            ":material/tune: Design",
            ":material/compare_arrows: Compare scenarios",
            ":material/rate_review: Assumptions & suggestions",
            ":material/menu_book: Method & caveats",
        ],
        key="main_tabs",
        on_change="rerun",
    )
    # Always rendered: the editors hold the user's edits (data editors cannot persist
    # their state while hidden) and are cheap.
    with tab_design:
        design = design_editor(base, base_name, base_catalogue())
    design_json, suggestions_json = design.model_dump_json(), active_json()
    result = simulate(design_json, suggestions_json, n_sims, seed)
    with header:
        headline(result, budget)
        if design != base:
            st.caption(f"Edited from `{base_name}` (see the Design tab).")
        if n := len(active()):
            st.warning(
                "Results use them instead of the published catalogue; see the Assumptions & "
                "suggestions tab.",
                title=f"{n} reviewer suggestion(s) active",
                icon=":material/rate_review:",
            )
        save_control(design, suggestions_json)
    with tab_assumptions:
        share = result.costs.by("item").mean(axis=1) / result.total.mean()
        assumptions_tab(base_catalogue(), session_catalogue(suggestions_json), share)
    # Only computed when open: extra simulations (dropout comparison, tornado, scenarios).
    if tab_results.open:
        with tab_results:
            results_tab(result, design_json, suggestions_json, n_sims, seed)
    if tab_compare.open:
        with tab_compare:
            compare_tab(lambda dj, sj: simulate(dj, sj, n_sims, seed), eur)
    if tab_method.open:
        with tab_method:
            method_tab(result, session_catalogue(suggestions_json))


main()
