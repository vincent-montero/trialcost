"""Assumptions tab: browse the catalogue and turn edits into reviewer suggestions (D40).

Edits never touch data/. A changed value plus a rationale becomes a suggestion
that applies to this browser session only (every result on the page reflects
it). Suggestions are downloaded as CSV and sent to Vincent, who reviews and
applies them with ``python -m trialcost.suggestions``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from trialcost import suggestions as sg
from trialcost.catalogue import Catalogue

LABELS = {
    "study_items": "Study items (CRO, vendors, fees)",
    "procedures": "Site procedures (France reference prices)",
    "countries": "Countries (price index, overhead, fees, activation)",
    "procedure_overrides": "Country-specific procedure prices",
    "sponsor_model_factors": "Sponsor model factors",
    "constants": "Constants",
}
KEY = "suggestions"


def active() -> pd.DataFrame:
    return st.session_state.setdefault(KEY, sg.empty())


def active_json() -> str:
    return active().to_json(orient="records")


def _display_table(table: str, catalogue: Catalogue, item_share: pd.Series) -> pd.DataFrame:
    attr, _, _, fields = sg.TABLES[table]
    df = getattr(catalogue, attr).reset_index()
    df.insert(0, "row_id", [sg.row_key(table, k) for k in getattr(catalogue, attr).index])
    keep = ["row_id"] + [c for c in ("label",) if c in df] + list(fields)
    keep += [c for c in ("evidence", "source", "notes") if c in df]
    out = df[keep].astype(dict.fromkeys(fields, float))  # allow decimals in the editor
    if table in ("study_items", "procedures"):
        out.insert(1, "share_of_total_pct", out["row_id"].map(item_share).fillna(0) * 100)
    out["rationale"] = ""
    return out


def assumptions_tab(base: Catalogue, session: Catalogue, item_share: pd.Series) -> None:
    st.markdown(
        "Every number comes from `data/` with an evidence grade (A sourced … D assumption). "
        "**Edit a value and give a rationale** to see its effect: it becomes a *suggestion* for "
        "this session only. Download your suggestions and send them to the author, who reviews "
        "them before anything changes in the catalogue."
    )
    reviewer = st.text_input(
        "Your name or role (optional, kept in the suggestions file)", key="sg_reviewer"
    )
    c1, c2 = st.columns([2, 1])
    table = c1.selectbox("Table", list(LABELS), format_func=LABELS.get, key="sg_table")
    only_d = c2.toggle("Assumptions only (grade D)", value=False, key="sg_only_d")

    shown = _display_table(table, session, item_share)
    if only_d and "evidence" in shown:
        shown = shown[shown["evidence"].str.contains("D")]
    if "share_of_total_pct" in shown:
        shown = shown.sort_values("share_of_total_pct", ascending=False)
    shown = shown.reset_index(drop=True)
    editable = set(sg.TABLES[table][3]) | {"rationale"}
    edited = st.data_editor(
        shown,
        key=f"sg_editor_{table}_{hash(active_json())}_{only_d}",
        hide_index=True,
        disabled=[c for c in shown.columns if c not in editable],
        column_config={
            "share_of_total_pct": st.column_config.NumberColumn(
                "% of total", format="%.1f", help="Share of the current design's mean total cost"
            ),
            "rationale": st.column_config.TextColumn(
                "rationale (required to suggest)", width="large"
            ),
        },
    )
    if st.button("Add my changes as suggestions", key="sg_add", type="primary"):
        new, missing = sg.from_edits(shown, edited, table, reviewer)
        if missing:
            st.error("Add a rationale for: " + ", ".join(missing))
        elif not new:
            st.info("No value changed.")
        else:
            merged = sg.merge(active(), new)
            if problems := sg.validate(merged, base):
                st.error("Not added:\n\n" + "\n".join(f"- {p}" for p in problems))
            else:
                st.session_state[KEY] = merged
                st.rerun()

    st.markdown("#### Suggestions in this session")
    current = active()
    if current.empty:
        st.caption("None yet. Results on every tab use the catalogue as published.")
    else:
        st.dataframe(current.drop(columns=["created"]), hide_index=True)
        c1, c2, c3 = st.columns(3)
        drop = c1.multiselect("Remove", current["suggestion_id"].tolist(), key="sg_drop")
        if c1.button("Remove selected", key="sg_remove", disabled=not drop):
            kept = current[~current["suggestion_id"].isin(drop)].to_dict("records")
            st.session_state[KEY] = sg.merge(
                sg.empty(), [{k: r[k] for k in sg.COLUMNS[1:]} for r in kept]
            )
            st.rerun()
        who = "-".join(reviewer.lower().split()) or "reviewer"
        c2.download_button(
            "Download suggestions (CSV)",
            current.to_csv(index=False),
            file_name=f"suggestions-{date.today():%Y%m%d}-{who}.csv",
            mime="text/csv",
            key="sg_download",
        )
        if c3.button("Clear all", key="sg_clear"):
            st.session_state[KEY] = sg.empty()
            st.rerun()

    with st.expander("Load a suggestions file"):
        upload = st.file_uploader("Suggestions CSV", type="csv", key="sg_upload")
        if upload is not None and st.button("Load into this session", key="sg_load"):
            try:
                loaded = sg.read(upload)
                merged = sg.merge(active(), loaded[sg.COLUMNS[1:]].to_dict("records"))
                if problems := sg.validate(merged, base):
                    st.error("Not loaded:\n\n" + "\n".join(f"- {p}" for p in problems))
                else:
                    st.session_state[KEY] = merged
                    st.rerun()
            except ValueError as e:
                st.error(str(e))
