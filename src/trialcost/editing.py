"""Design <-> editable tables, for the Streamlit app (pure functions, no UI).

Every ``apply_*`` function returns a **new** validated ``Design`` (Pydantic
raises ``ValidationError`` on inconsistent edits) and never mutates its input.

Visit grid: one row per visit (``id``, ``day``, ``screening``, ``remote``)
plus one boolean column per catalogue procedure. A procedure listed twice at a
visit becomes a single tick (no current design does this).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from trialcost import DATA_DIR
from trialcost.design import CountryAllocation, Design, Population, Visit
from trialcost.distributions import Beta, LogNormal

VISIT_COLUMNS = ["id", "day", "screening", "remote"]


# --------------------------------------------------------------------------- #
# Countries
# --------------------------------------------------------------------------- #


def countries_table(design: Design) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": [c.country for c in design.countries],
            "n_sites": [c.n_sites for c in design.countries],
        }
    )


def apply_countries(design: Design, table: pd.DataFrame) -> Design:
    """Replace country allocations; a country already in the design keeps its activation."""
    previous = {c.country: c.activation_months for c in design.countries}
    rows = table.dropna(subset=["country"])
    rows = rows[rows["n_sites"].fillna(0).astype(int) > 0]
    countries = [
        CountryAllocation(
            country=str(r.country).upper(),
            n_sites=int(r.n_sites),
            activation_months=previous.get(str(r.country).upper()),
        )
        for r in rows.itertuples()
    ]
    return Design.model_validate(
        {**design.model_dump(), "countries": [c.model_dump() for c in countries]}
    )


# --------------------------------------------------------------------------- #
# Visits
# --------------------------------------------------------------------------- #


def visits_grid(design: Design, procedure_ids: list[str]) -> pd.DataFrame:
    rows = []
    for v in design.visits:
        row = {"id": v.id, "day": v.day, "screening": v.screening, "remote": v.remote}
        row.update({p: p in v.procedures for p in procedure_ids})
        rows.append(row)
    return pd.DataFrame(rows, columns=VISIT_COLUMNS + list(procedure_ids))


def apply_visits(design: Design, grid: pd.DataFrame) -> Design:
    """Rebuild the visit list from the grid (rows without an id are ignored).

    A visit that already existed keeps its procedure order; newly ticked
    procedures are appended in catalogue order (tidy YAML diffs).
    """
    procedures = [c for c in grid.columns if c not in VISIT_COLUMNS]
    previous = {v.id: v.procedures for v in design.visits}
    visits = []
    for r in grid.dropna(subset=["id", "day"]).to_dict("records"):
        vid = str(r["id"]).strip()
        if not vid:
            continue
        ticked = [p for p in procedures if bool(r.get(p) or False)]
        old = [p for p in previous.get(vid, []) if p in ticked]
        visits.append(
            Visit(
                id=vid,
                day=int(r["day"]),
                screening=bool(r.get("screening") or False),
                remote=bool(r.get("remote") or False),
                procedures=old + [p for p in ticked if p not in old],
            )
        )
    visits.sort(key=lambda v: v.day)
    return Design.model_validate(
        {**design.model_dump(), "visits": [v.model_dump() for v in visits]}
    )


def regular_visits(
    design: Design, every_weeks: int, until_week: int, template_visit_id: str
) -> Design:
    """Keep visits up to day 0; add visits every N weeks with a template's procedures.

    Quick-mode helper (D15): the follow-up schedule becomes W{n}, W{2n}, ... up to
    ``until_week``, each with the procedures of ``template_visit_id``.
    """
    from trialcost.design import visits_every

    template = next(v for v in design.visits if v.id == template_visit_id)
    kept = [v for v in design.visits if v.day <= 0]
    new = visits_every(every_weeks, until_week, template.procedures, remote=template.remote)
    taken = {v.id for v in kept}
    new = [v for v in new if v.id not in taken]
    return Design.model_validate(
        {**design.model_dump(), "visits": [v.model_dump() for v in kept + new]}
    )


# --------------------------------------------------------------------------- #
# Population presets from ClinicalTrials.gov priors
# --------------------------------------------------------------------------- #


def prior_presets(path: Path = DATA_DIR / "recruitment_priors.csv") -> dict[str, dict]:
    """``{"rare_neuro, phase=PHASE2": {"rate_mean": LogNormal, "annual_dropout_rate": Beta}}``."""
    df = pd.read_csv(path)
    presets: dict[str, dict] = {}
    for (segment, subset), rows in df.groupby(["segment", "subset"], sort=False):
        entry = {}
        for r in rows.itertuples():
            if r.parameter == "rate_mean":
                entry["rate_mean"] = LogNormal(median=r.p1, gsd=r.p2)
            elif r.parameter == "annual_dropout_rate":
                entry["annual_dropout_rate"] = Beta(mean=r.p1, kappa=r.p2)
        entry["n_trials"] = int(rows["n_trials"].max())
        presets[f"{segment}, {subset}"] = entry
    return presets


def apply_population(design: Design, **fields) -> Design:
    """Replace some population fields (distributions or numbers)."""
    population = Population.model_validate({**design.population.model_dump(), **fields})
    return design.model_copy(update={"population": population}, deep=True)


def apply_fields(design: Design, **fields) -> Design:
    """Replace top-level fields (e.g. ``n_randomised``) or ``operations`` sub-fields."""
    data = design.model_dump()
    ops = fields.pop("operations", None)
    data.update(fields)
    if ops:
        data["operations"] = {**data["operations"], **ops}
    return Design.model_validate(data)


def to_yaml(design: Design) -> str:
    """YAML text of a design (plain numbers for Fixed values), loadable by load_design."""

    def simplify(obj):
        if isinstance(obj, dict):
            if obj.get("kind") == "fixed":
                return obj["value"]
            return {k: simplify(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [simplify(v) for v in obj]
        return obj

    return yaml.safe_dump(simplify(design.model_dump()), sort_keys=False, allow_unicode=True)
