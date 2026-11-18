"""Streamlit MVP of the clinical trial cost estimator (README §8 roadmap #6).

    uv run --extra ui streamlit run app/streamlit_app.py

UI only: every number comes from the engine (``src/trialcost``) and the
catalogue (``data/``). Nothing here writes to ``data/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

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


def sidebar() -> tuple[Design, int, int, float | None]:
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
    return design, int(n_sims), int(seed), (budget or None)


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


def main() -> None:
    st.title("Clinical trial cost estimator")
    st.caption(
        "Design-driven Monte Carlo estimate of a trial's total sponsor budget. Personal project, "
        "public data only. Many unit costs are expert-judgement placeholders: read results as "
        "orders of magnitude with honest uncertainty, not quotes."
    )
    design, n_sims, seed, budget = sidebar()
    result = simulate(design.model_dump_json(), n_sims, seed)
    headline(result, budget)


main()
