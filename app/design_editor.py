"""Design tab: edit the selected design; returns the edited, validated Design.

Edits are applied in order (basics -> countries -> population -> operations ->
visits). An invalid edit shows an error and keeps the last valid design, so the
results never run on an inconsistent design. Widget keys are prefixed with the
starting design's name: choosing another design resets the editor.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st
from pydantic import ValidationError

from trialcost import editing
from trialcost.catalogue import Catalogue
from trialcost.design import Design, Operations
from trialcost.distributions import Beta, LogNormal

KEEP = "As in the design file"
CUSTOM = "Custom"


def _try(design: Design, label: str, change: Callable[[Design], Design]) -> Design:
    try:
        return change(design)
    except (ValidationError, ValueError, StopIteration) as e:
        st.error(str(e), title=f"{label}: edit not applied (the design would be invalid)")
        return design


def _choices(field: str) -> list[str]:
    from typing import get_args

    return list(get_args(Operations.model_fields[field].annotation))


def basics(d: Design, k: str) -> Design:
    n = st.number_input("Randomised patients (N)", min_value=1, value=d.n_randomised, key=f"{k}_n")
    return _try(d, "N", lambda x: editing.apply_fields(x, n_randomised=int(n)))


def countries(d: Design, k: str, cat: Catalogue) -> Design:
    st.markdown("**Countries and sites**")
    table = st.data_editor(
        editing.countries_table(d),
        alt="Countries and number of sites",
        num_rows="dynamic",
        key=f"{k}_countries",
        column_config={
            "country": st.column_config.SelectboxColumn(
                "Country", options=list(cat.countries.index), required=True
            ),
            "n_sites": st.column_config.NumberColumn("Sites", min_value=0, step=1, required=True),
        },
        hide_index=True,
    )
    st.caption("Activation delays come from `countries.csv` unless the design file sets them.")
    return _try(d, "Countries", lambda x: editing.apply_countries(x, table))


def population(d: Design, k: str) -> Design:
    st.markdown("**Population and recruitment**")
    presets = editing.prior_presets()
    choice = st.selectbox(
        "Recruitment rate and dropout",
        [KEEP, *presets, CUSTOM],
        key=f"{k}_preset",
        help="ClinicalTrials.gov-derived priors (README §2.4, data/recruitment_priors.csv).",
    )
    fields: dict = {}
    if choice in presets:
        p = presets[choice]
        fields = {key: p[key] for key in ("rate_mean", "annual_dropout_rate") if key in p}
        st.caption(f"Prior fitted on {p['n_trials']} trials.")
    elif choice == CUSTOM:
        c1, c2, c3, c4 = st.columns(4)
        rate = c1.number_input(
            "Rate median (pts / open site / month)", 0.01, 10.0, 0.30, key=f"{k}_rm"
        )
        gsd = c2.number_input("Rate gsd (spread, > 1)", 1.01, 10.0, 2.0, key=f"{k}_rg")
        drop = c3.number_input("Annual dropout, mean", 0.0, 0.95, 0.20, key=f"{k}_dm")
        kappa = c4.number_input("Dropout κ (concentration)", 0.5, 200.0, 5.0, key=f"{k}_dk")
        fields = {
            "rate_mean": LogNormal(median=rate, gsd=gsd),
            "annual_dropout_rate": Beta(mean=max(drop, 1e-4), kappa=kappa),
        }
    c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
    sf_now = d.population.screen_failure_rate.central()
    sf = c1.slider(
        "Screen-failure rate (placeholder)", 0.0, 0.9, float(sf_now), 0.05, key=f"{k}_sf"
    )
    shape = c2.number_input(
        "Between-site spread (gamma shape)",
        0.2,
        50.0,
        float(d.population.rate_shape),
        key=f"{k}_shape",
    )
    if sf != sf_now:
        fields["screen_failure_rate"] = Beta(mean=max(sf, 1e-4), kappa=20)
    fields["rate_shape"] = shape
    c3.caption(f"Rate: {d.population.rate_mean.model_dump(exclude={'kind'})}")
    return _try(d, "Population", lambda x: editing.apply_population(x, **fields))


def operations(d: Design, k: str) -> Design:
    st.markdown("**Operations**")
    o = d.operations
    c1, c2, c3 = st.columns(3)
    monitoring, models = _choices("monitoring"), _choices("sponsor_model")
    ops: dict = {
        "monitoring": c1.selectbox(
            "Monitoring", monitoring, index=monitoring.index(o.monitoring), key=f"{k}_mon"
        ),
        "sponsor_model": c3.selectbox(
            "Sponsor model", models, index=models.index(o.sponsor_model), key=f"{k}_spons"
        ),
    }
    interval_now = float(o.monitoring_interval_months.central())
    interval = c2.number_input(
        "Monitoring interval (months)", 0.25, 12.0, interval_now, 0.25, key=f"{k}_int"
    )
    if interval != interval_now:  # keep a distribution from the file unless edited
        ops["monitoring_interval_months"] = interval
    c1, c2, c3, c4, c5 = st.columns(5, vertical_alignment="bottom")
    ops["dmc"] = c1.toggle("DMC", o.dmc, key=f"{k}_dmc")
    ops["central_imaging_reading"] = c2.toggle(
        "Central imaging", o.central_imaging_reading, key=f"{k}_img"
    )
    ops["irt"] = c3.toggle("IRT", o.irt, key=f"{k}_irt")
    ops["patient_travel_support"] = c4.toggle(
        "Travel support", o.patient_travel_support, key=f"{k}_trav"
    )
    ops["n_interim_analyses"] = int(
        c5.number_input("Interim analyses", 0, 5, o.n_interim_analyses, key=f"{k}_ia")
    )
    if ops["monitoring"] == "full_sdv" and interval >= 2:
        st.warning(
            "Full SDV with a ≥ 2-month interval: ADAMON saw 1.8-3.5x more visits with extensive "
            "on-site monitoring; consider ~1 month (README §2.9)."
        )
    return _try(d, "Operations", lambda x: editing.apply_fields(x, operations=ops))


def visits(d: Design, k: str, cat: Catalogue) -> Design:
    st.markdown("**Visit schedule**")
    gen_key = f"{k}_generated"
    with st.expander("Quick mode: regular visits", icon=":material/event_repeat:"):
        c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
        every = c1.number_input("Every (weeks)", 1, 26, 4, key=f"{k}_every")
        until = c2.number_input("Until week", 1, 260, 52, key=f"{k}_until")
        post = [v.id for v in d.visits if v.day > 0]
        template = c3.selectbox("Procedures as in visit", post, key=f"{k}_tmpl")
        if st.button("Replace follow-up visits", key=f"{k}_gen", icon=":material/autorenew:"):
            st.session_state[gen_key] = editing.regular_visits(
                d, int(every), int(until), template
            ).model_dump_json()
    source = (
        Design.model_validate_json(st.session_state[gen_key]) if gen_key in st.session_state else d
    )
    ids = list(cat.procedures.index)
    labels = cat.procedures["label"]
    grid = st.data_editor(
        editing.visits_grid(source, ids),
        alt="Visit schedule: one row per visit, one tick box per procedure",
        num_rows="dynamic",
        hide_index=True,
        key=f"{k}_visits_{hash(st.session_state.get(gen_key, ''))}",
        column_config={
            "id": st.column_config.TextColumn("Visit", required=True),
            "day": st.column_config.NumberColumn("Day", step=1, required=True),
            **{p: st.column_config.CheckboxColumn(p, help=labels[p]) for p in ids},
        },
    )
    st.caption(
        "Screening visits must have day < 0; every on-site visit also gets coordinator time, "
        "logistics, vital signs and (if enabled) patient travel automatically."
    )
    return _try(d, "Visits", lambda x: editing.apply_visits(x, grid))


def design_editor(base: Design, base_name: str, cat: Catalogue) -> Design:
    k = f"ed_{base_name}"
    with st.container(border=True):
        d = basics(base, k)
        d = countries(d, k, cat)
    with st.container(border=True):
        d = population(d, k)
    with st.container(border=True):
        d = operations(d, k)
    with st.container(border=True):
        d = visits(d, k, cat)
    st.download_button(
        "Download design (YAML)",
        editing.to_yaml(d),
        file_name=f"{base_name}-edited.yaml",
        mime="text/yaml",
        key=f"{k}_download",
        icon=":material/download:",
    )
    return d
