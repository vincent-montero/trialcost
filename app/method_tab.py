"""Method & caveats tab: how the estimate is built and how far to trust it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from trialcost.catalogue import Catalogue
from trialcost.simulate import SimulationResult

ROOT = Path(__file__).resolve().parents[1]


def evidence_shares(result: SimulationResult, catalogue: Catalogue) -> pd.Series:
    """Share of the design's mean cost by evidence grade of the line's catalogue row.

    Mixed grades (e.g. "C/D") count as their weakest grade; CTA fees take the
    country row's grade.
    """
    meta = result.costs.meta
    mean = pd.Series(result.costs.values.mean(axis=1), index=meta.index)
    grades = pd.concat(
        [catalogue.procedures["evidence"], catalogue.study_items["evidence"]]
    ).to_dict()
    country_grade = catalogue.countries["evidence"].to_dict()

    def weakest(g: str) -> str:
        return max(str(g).split("/"))

    g = [
        weakest(country_grade.get(c, "D")) if item == "cta_fee" else weakest(grades.get(item, "D"))
        for item, c in zip(meta["item"], meta["country"], strict=True)
    ]
    return mean.groupby(g).sum() / mean.sum()


def method_tab(result: SimulationResult, catalogue: Catalogue) -> None:
    st.markdown(
        """
**What it estimates.** The total **sponsor budget** of one clinical trial (CRO, sites,
vendors, fees; EUR, price year 2026) and its duration, from the trial's **design**: patients,
visit schedule and procedures, countries and sites, operational choices. Not a site budget,
not drug manufacturing, not sponsor internal staff (except under the in-house model).

**How.** Each simulated trial draws site activation and recruitment (Poisson-gamma model),
screen failure and dropout, and every uncertain unit price; it then counts what the design
consumes (screened and randomised patients, visits and procedures, site-months, monitoring
visits, months of project management...) and prices it. 10,000 simulations give the P10 / P50
/ P90 shown. Scenarios share the same random numbers, so differences between them are exact.

**Where numbers come from.** Every number lives in a CSV of the public repository with an
evidence grade: **A** read from an official tariff, **B** secondary source, **C** derived,
**D** expert-judgement placeholder. Recruitment rates and dropout come from ClinicalTrials.gov
(rare neurological trials, 2010 onwards).
"""
    )
    shares = evidence_shares(result, catalogue)
    st.markdown("**How much of this design's cost rests on each evidence grade**")
    cols = st.columns(4)
    names = {"A": "A sourced", "B": "B secondary", "C": "C derived", "D": "D assumption"}
    for col, grade in zip(cols, "ABCD", strict=True):
        col.metric(names[grade], f"{100 * shares.get(grade, 0):.0f} %")

    bench = ROOT / "reports" / "benchmark-validation.csv"
    if bench.exists():
        st.markdown("**Against published benchmarks** (reference designs; README §7.2)")
        df = pd.read_csv(bench)
        df = df[df["comparability"] != "poor"]
        st.dataframe(
            df[["description", "design", "ratio_model_to_benchmark", "comparability"]].rename(
                columns={"ratio_model_to_benchmark": "model / benchmark"}
            ),
            hide_index=True,
        )
        st.caption(
            "Reading: mid-size phase 2 in range; large, long trials ~15-30 % above benchmarks; "
            "small trials ~20-25 % below. The model is probably too flat across trial size."
        )
    st.markdown(
        """
**Known limits.** Screen-failure rate and between-site recruitment spread are placeholders.
CRO labour rates rest on salary times an assumed bill-rate multiplier and are the single largest
source of uncertainty. Benchmarks are US-centric averages over many designs. Read results as
orders of magnitude with honest uncertainty, not quotes.

**Suggest better numbers** in the *Assumptions & suggestions* tab: your changes apply to your
session only and can be downloaded and sent to the author for review.
"""
    )
