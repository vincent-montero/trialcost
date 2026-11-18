"""Quantities x unit prices -> cost lines (EUR 2026).

Unit price of a site procedure in country ``c`` (D16, D25)::

    override exists        -> override price x (1 + overhead_c if apply_overhead)
    country_scaling = site -> FR price x index_c x (1 + overhead_c)
    country_scaling = none -> FR price (vendor price, same everywhere)

Study items follow the same ``country_scaling`` rule (D23), multiplied by the
quantity of their driver; country fees (CTA) come from ``countries.csv``.
Under a ``functional`` or ``in_house`` sponsor model, study items delivered by
CRO labour (``country_scaling = none`` and resource listed in
``sponsor_model_factors.csv``) are multiplied by that model's factor (D31).

Random draws and their sharing
------------------------------
- A country's site price index and overhead: one draw per simulation, shared
  by every procedure and item in that country.
- A procedure / item / override price: one draw per simulation, shared by all
  countries (the uncertainty is about the price level, not per country).
- Every catalogue row is drawn on every run, whether or not it applies, each
  from its own child generator (``rng.spawn``). Changing a design option or
  pinning one price therefore does not shift any other random number
  (common random numbers between scenarios, D29).

Output: ``CostLines``, i.e. a metadata table with one row per cost line
(item, country, category, resource, driver, source) plus a
``(n_lines, n_sims)`` array of costs. Any breakdown is a groupby on the
metadata; ``to_frame()`` gives the tidy long table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trialcost.catalogue import STUDY_LEVEL_DRIVERS, Catalogue
from trialcost.design import Design
from trialcost.distributions import Triangular, draw
from trialcost.quantities import GLOBAL, Quantities

META_COLUMNS = ["item", "source", "country", "category", "resource", "driver"]


@dataclass(frozen=True)
class CostLines:
    """Cost lines of ``n_sims`` simulated trials."""

    meta: pd.DataFrame  # one row per line, columns META_COLUMNS
    values: np.ndarray  # (n_lines, n_sims), EUR 2026

    @property
    def n_sims(self) -> int:
        return self.values.shape[1]

    def total(self) -> np.ndarray:
        """Total cost per simulation, shape ``(n_sims,)``."""
        return self.values.sum(axis=0)

    def by(self, *columns: str) -> pd.DataFrame:
        """Sum lines by metadata columns: one row per group, one column per simulation."""
        groups = self.meta.groupby(list(columns), sort=True).indices
        return pd.DataFrame(
            np.array([self.values[idx].sum(axis=0) for idx in groups.values()]),
            index=pd.Index(list(groups.keys()), name=columns[0])
            if len(columns) == 1
            else pd.MultiIndex.from_tuples(list(groups.keys()), names=list(columns)),
        )

    def select(self, mask: np.ndarray | pd.Series) -> CostLines:
        mask = np.asarray(mask, dtype=bool)
        return CostLines(self.meta[mask].reset_index(drop=True), self.values[mask])

    def to_frame(self) -> pd.DataFrame:
        """Tidy table: one row per (sim, line) with its metadata and ``cost``."""
        n_lines = len(self.meta)
        frame = self.meta.loc[np.tile(np.arange(n_lines), self.n_sims)].reset_index(drop=True)
        frame.insert(0, "sim", np.repeat(np.arange(self.n_sims), n_lines))
        frame["cost"] = self.values.T.ravel()
        return frame


@dataclass(frozen=True)
class CountryFactors:
    index: np.ndarray  # (n_sims, n_c) site price index vs FR
    overhead: np.ndarray  # (n_sims, n_c) site overhead rate


def draw_country_factors(
    catalogue: Catalogue,
    countries: tuple[str, ...],
    rng: np.random.Generator,
    n_sims: int,
    deterministic: bool = False,
) -> CountryFactors:
    s_index, s_overhead = rng.spawn(2)
    return CountryFactors(
        index=_draw_country_range(
            catalogue, countries, "site_price", s_index, n_sims, deterministic
        ),
        overhead=_draw_country_range(
            catalogue, countries, "site_overhead", s_overhead, n_sims, deterministic
        ),
    )


def _draw_country_range(
    catalogue: Catalogue,
    countries: tuple[str, ...],
    name: str,
    rng: np.random.Generator,
    n_sims: int,
    deterministic: bool,
) -> np.ndarray:
    """One draw per simulation and country, shape ``(n_sims, n_c)``."""
    streams = rng.spawn(len(countries))
    return np.column_stack(
        [
            draw(catalogue.country_range(c, name), s, n_sims, deterministic)
            for c, s in zip(countries, streams, strict=True)
        ]
    )


def _draw_rows(
    table: pd.DataFrame, rng: np.random.Generator, n_sims: int, deterministic: bool
) -> dict:
    """One price draw per row of a catalogue table (index -> (n_sims,) array)."""
    streams = rng.spawn(len(table))
    return {
        key: draw(Triangular(low=r.low, mode=r.mode, high=r.high), s, n_sims, deterministic)
        for key, r, s in zip(table.index, table.itertuples(index=False), streams, strict=True)
    }


def compute_costs(
    design: Design,
    catalogue: Catalogue,
    q: Quantities,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> CostLines:
    """Price every quantity; return the cost lines."""
    n_sims, countries = q.n_sims, q.countries

    # --- draws (fixed order, every row, see module docstring) ---
    s_factors, s_proc, s_item, s_override, s_sponsor, s_cta = rng.spawn(6)
    factors = draw_country_factors(catalogue, countries, s_factors, n_sims, deterministic)
    site_scale = factors.index * (1 + factors.overhead)  # (n_sims, n_c)
    proc_price = _draw_rows(catalogue.procedures, s_proc, n_sims, deterministic)
    item_price = _draw_rows(catalogue.study_items, s_item, n_sims, deterministic)
    override_price = _draw_rows(catalogue.overrides, s_override, n_sims, deterministic)
    sponsor_factor_draws = _draw_rows(catalogue.sponsor_factors, s_sponsor, n_sims, deterministic)
    cta_fee = _draw_country_range(catalogue, countries, "cta_fee_eur", s_cta, n_sims, deterministic)

    meta: list[dict] = []
    values: list[np.ndarray] = []

    def add(item, source, country, category, resource, driver, value):
        meta.append(
            dict(
                zip(META_COLUMNS, [item, source, country, category, resource, driver], strict=True)
            )
        )
        values.append(value)

    # --- site procedures ---
    procs = catalogue.procedures
    for j, pid in enumerate(q.procedure_ids):
        row = procs.loc[pid]
        for k, c in enumerate(countries):
            count = q.procedure_counts[:, k, j]
            if not count.any():
                continue
            if (pid, c) in override_price:
                unit = override_price[(pid, c)]
                if bool(catalogue.overrides.loc[(pid, c), "apply_overhead"]):
                    unit = unit * (1 + factors.overhead[:, k])
            elif row.country_scaling == "site_price":
                unit = proc_price[pid] * site_scale[:, k]
            else:
                unit = proc_price[pid]
            add(pid, "procedure", c, row.category, row.resource, "per_occurrence", count * unit)

    # --- study items ---
    items = catalogue.study_items
    model = design.operations.sponsor_model
    labour = catalogue.sponsor_model_factor(model)

    def sponsor_factor(row) -> np.ndarray | float:
        if labour and row.country_scaling == "none" and row.resource in labour.resources:
            return sponsor_factor_draws[model]
        return 1.0

    for item_id, cond in catalogue.conditions.items():
        row = items.loc[item_id]
        qty = q.driver(row.driver)
        if row.driver in STUDY_LEVEL_DRIVERS:
            if cond.holds(design) and np.any(qty):
                value = item_price[item_id] * qty * sponsor_factor(row)
                add(item_id, "study_item", GLOBAL, row.category, row.resource, row.driver, value)
            continue
        for k, c in enumerate(countries):
            if not cond.holds(design, c):
                continue
            unit = item_price[item_id] * sponsor_factor(row)
            if row.country_scaling == "site_price":
                unit = unit * site_scale[:, k]
            add(item_id, "study_item", c, row.category, row.resource, row.driver, unit * qty[:, k])

    # --- country fees (CTA / ethics / IRB central review) ---
    for k, c in enumerate(countries):
        add(
            "cta_fee", "country_fee", c, "regulatory", "authority_fee", "per_country", cta_fee[:, k]
        )

    return CostLines(meta=pd.DataFrame(meta, columns=META_COLUMNS), values=np.vstack(values))
