"""Summaries, breakdowns, dropout-cost report and tornado.

All functions take a ``SimulationResult`` (or run the simulations they need)
and return pandas DataFrames; ``format_report`` turns them into text and
``write_outputs`` into CSV files.

Dropout cost (README §2.5, D14)
-------------------------------
The same design is simulated twice with the same seed: as given, and with
``annual_dropout_rate = 0``. Thanks to common random numbers (D29) the two
runs differ only through dropout, so the per-simulation difference is the
cost impact of dropout. Completers = N x exp(-h x patient duration), the
expected number of randomised patients attending the last visit.

Tornado (one-way sensitivity)
-----------------------------
Each input is set to a low then a high value while every other input keeps
its full uncertainty; the bar is the change in **mean** total cost (same seed
for every run, so differences are not Monte Carlo noise). Low / high are:

- the input's P10 / P90 when it is uncertain;
- 0.5x / 1.5x its value when it is fixed (no stated uncertainty);
- for a group (e.g. all countries' site price index, all CRO labour prices),
  every member at its own P10 / P90 together; members without spread stay put.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trialcost.catalogue import Catalogue
from trialcost.design import Design
from trialcost.distributions import AnyDist, Fixed, Triangular
from trialcost.quantities import dropout_hazard
from trialcost.simulate import SimulationResult, run

QUANTILES = {"P10": 0.1, "P50": 0.5, "P90": 0.9}
#: Proportions swung by +50 % are capped below 1.
_MAX_PROPORTION = 0.95


# --------------------------------------------------------------------------- #
# Summary and breakdowns
# --------------------------------------------------------------------------- #


def describe(x: np.ndarray) -> dict[str, float]:
    """Mean and P10 / P50 / P90 of a per-simulation array."""
    out = {"mean": float(np.mean(x))}
    out.update({k: float(np.quantile(x, q)) for k, q in QUANTILES.items()})
    return out


def summary(result: SimulationResult) -> pd.DataFrame:
    """Headline distributions: cost, duration, LPI, cost per randomised patient."""
    q = result.quantities
    rows = {
        "total_cost_eur": result.total,
        "duration_months": result.duration,
        "lpi_months": q.t_lpi,
        "cost_per_randomised_eur": result.total / result.design.n_randomised,
    }
    return pd.DataFrame({k: describe(v) for k, v in rows.items()}).T[["P10", "P50", "P90", "mean"]]


def probabilities(
    result: SimulationResult, budget: float | None = None, recruit_by: float | None = None
) -> dict[str, float]:
    out = {}
    if budget is not None:
        out[f"P(total cost <= {budget:,.0f} EUR)"] = float(np.mean(result.total <= budget))
    if recruit_by is not None:
        out[f"P(LPI <= {recruit_by:g} months)"] = float(
            np.mean(result.quantities.t_lpi <= recruit_by)
        )
    return out


def breakdown(result: SimulationResult, by: str) -> pd.DataFrame:
    """Cost by ``category`` / ``resource`` / ``country`` / ``item``: mean, P10-P90, share.

    ``share`` = group mean / total mean, so shares add up to 1 (quantiles of
    groups do not add up to the quantiles of the total).
    """
    groups = result.costs.by(by)
    stats = pd.DataFrame([describe(row) for row in groups.to_numpy()], index=groups.index)
    stats["share"] = stats["mean"] / result.total.mean()
    return stats[["P10", "P50", "P90", "mean", "share"]].sort_values("mean", ascending=False)


# --------------------------------------------------------------------------- #
# Dropout cost
# --------------------------------------------------------------------------- #


def completers(result: SimulationResult) -> np.ndarray:
    """Expected patients attending the last visit, per simulation."""
    q, d = result.quantities, result.design
    h = dropout_hazard(q.annual_dropout)
    return d.n_randomised * np.exp(-h * d.patient_duration_months)


def dropout_report(
    design: Design, catalogue: Catalogue, n_sims: int, seed: int, deterministic: bool = False
) -> pd.DataFrame:
    """Cost impact of dropout: per completer, and difference vs the same design without dropout."""
    base = run(design, catalogue, n_sims, seed, deterministic)
    no_dropout_design = design.model_copy(deep=True)
    no_dropout_design.population.annual_dropout_rate = Fixed(value=0.0)
    zero = run(no_dropout_design, catalogue, n_sims, seed, deterministic)

    n = design.n_randomised
    done = completers(base)
    rows = {
        "completers": done,
        "cost_per_randomised_eur": base.total / n,
        "cost_per_completer_eur": base.total / done,
        "total_cost_no_dropout_eur": zero.total,
        "cost_per_completer_no_dropout_eur": zero.total / n,
        "dropout_delta_cost_eur": base.total - zero.total,
        "dropout_delta_cost_pct": 100 * (base.total / zero.total - 1),
        "completer_cost_premium_pct": 100 * ((base.total / done) / (zero.total / n) - 1),
    }
    return pd.DataFrame({k: describe(v) for k, v in rows.items()}).T[["P10", "P50", "P90", "mean"]]


# --------------------------------------------------------------------------- #
# Tornado
# --------------------------------------------------------------------------- #


def _swing_value(dist: AnyDist, q: float) -> float:
    """P10 / P90 of an uncertain input, or 0.5x / 1.5x of a fixed one."""
    if dist.is_point_mass:
        return dist.central() * (0.5 if q < 0.5 else 1.5)
    return dist.quantile(q)


def _pin_rows(
    df: pd.DataFrame, mask: pd.Series | np.ndarray, q: float, prefix: str = ""
) -> pd.DataFrame:
    """Copy of ``df`` with the selected triangular rows fixed at their quantile ``q``."""
    cols = [f"{prefix}low", f"{prefix}mode", f"{prefix}high"]
    df = df.astype(dict.fromkeys(cols, float))  # copy; CSV integer columns become float
    for idx in df.index[np.asarray(mask, dtype=bool)]:
        tri = Triangular(**dict(zip(("low", "mode", "high"), df.loc[idx, cols], strict=True)))
        if not tri.is_point_mass:
            df.loc[idx, cols] = tri.quantile(q)
    return df


def _scale_row(df: pd.DataFrame, key: str, q: float) -> pd.DataFrame:
    """Copy of ``df`` with one triangular row set to its swing value (point mass: x0.5/1.5)."""
    df = df.astype(dict.fromkeys(["low", "mode", "high"], float))
    tri = Triangular(low=df.loc[key, "low"], mode=df.loc[key, "mode"], high=df.loc[key, "high"])
    df.loc[key, ["low", "mode", "high"]] = _swing_value(tri, q)
    return df


Apply = Callable[[Design, Catalogue, float], tuple[Design, Catalogue, str]]


@dataclass(frozen=True)
class Swing:
    name: str
    apply: Apply  # (design copy, catalogue, q) -> (design, catalogue, value label)


def _population(field: str, proportion: bool = False) -> Apply:
    def apply(d: Design, c: Catalogue, q: float):
        v = _swing_value(getattr(d.population, field), q)
        if proportion:
            v = min(v, _MAX_PROPORTION)
        setattr(d.population, field, Fixed(value=v))
        return d, c, f"{v:.3g}"

    return apply


def _rate_shape(d: Design, c: Catalogue, q: float):
    d.population.rate_shape *= 0.5 if q < 0.5 else 1.5
    return d, c, f"{d.population.rate_shape:.3g}"


def _monitoring_interval(d: Design, c: Catalogue, q: float):
    v = _swing_value(d.operations.monitoring_interval_months, q)
    d.operations.monitoring_interval_months = Fixed(value=v)
    return d, c, f"{v:.3g}"


def _activation(d: Design, c: Catalogue, q: float):
    for alloc in d.countries:
        if alloc.activation_months is not None and not alloc.activation_months.is_point_mass:
            alloc.activation_months = Fixed(value=alloc.activation_months.quantile(q))
    in_design = c.countries.index.isin([a.country for a in d.countries])
    countries = _pin_rows(c.countries, in_design, q, prefix="activation_months_")
    return d, dataclasses.replace(c, countries=countries), _pct_label(q)


def _country_column(prefix: str) -> Apply:
    def apply(d: Design, c: Catalogue, q: float):
        countries = _pin_rows(c.countries, np.ones(len(c.countries)), q, prefix=prefix)
        return d, dataclasses.replace(c, countries=countries), _pct_label(q)

    return apply


def cro_labour_mask(catalogue: Catalogue) -> pd.Series:
    """Study items delivered by CRO labour: same definition as the sponsor-model factors."""
    resources = set().union(*(r.split(";") for r in catalogue.sponsor_factors["resources"]))
    items = catalogue.study_items
    return (items["country_scaling"] == "none") & items["resource"].isin(resources)


def _cro_labour(d: Design, c: Catalogue, q: float):
    items = _pin_rows(c.study_items, cro_labour_mask(c), q)
    return d, dataclasses.replace(c, study_items=items), _pct_label(q)


def _closeout(d: Design, c: Catalogue, q: float):
    constants = _scale_row(c.constants, "closeout_months", q)
    return (
        d,
        dataclasses.replace(c, constants=constants),
        f"{constants.loc['closeout_months', 'mode']:.3g}",
    )


def _pct_label(q: float) -> str:
    return f"all at P{round(100 * q)}"


TORNADO_SWINGS: tuple[Swing, ...] = (
    Swing("Recruitment rate (rate_mean)", _population("rate_mean")),
    Swing("Between-site rate spread (rate_shape)", _rate_shape),
    Swing("Site activation delays", _activation),
    Swing("Screen-failure rate", _population("screen_failure_rate", proportion=True)),
    Swing("Annual dropout rate", _population("annual_dropout_rate", proportion=True)),
    Swing("Monitoring interval (months)", _monitoring_interval),
    Swing("Site price index", _country_column("site_price_")),
    Swing("Site overhead", _country_column("site_overhead_")),
    Swing("CRO labour prices", _cro_labour),
    Swing("Close-out duration (months)", _closeout),
)


def tornado(
    design: Design,
    catalogue: Catalogue,
    n_sims: int = 2_000,
    seed: int = 42,
    swings: tuple[Swing, ...] = TORNADO_SWINGS,
) -> pd.DataFrame:
    """One-way sensitivity of the mean total cost, sorted by swing (largest first)."""
    base = run(design, catalogue, n_sims, seed).total.mean()
    rows = []
    for swing in swings:
        out = {"parameter": swing.name}
        for side, q in (("low", 0.1), ("high", 0.9)):
            d, c, label = swing.apply(design.model_copy(deep=True), catalogue, q)
            out[f"{side}_input"] = label
            out[f"cost_at_{side}"] = run(d, c, n_sims, seed).total.mean()
        rows.append(out)
    df = pd.DataFrame(rows)
    df["base"] = base
    df["swing"] = (df["cost_at_high"] - df["cost_at_low"]).abs()
    return df.sort_values("swing", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Text and files
# --------------------------------------------------------------------------- #


def _eur(x: float) -> str:
    return f"{x / 1e6:,.2f} M€" if abs(x) >= 1e6 else f"{x / 1e3:,.0f} k€"


def _months(x: float) -> str:
    return f"{x:.1f} mo"


def _quantile_line(label: str, row: pd.Series, fmt: Callable[[float], str]) -> str:
    return f"  {label:<16} " + "   ".join(f"{k} {fmt(row[k])}" for k in QUANTILES)


def _money_table(df: pd.DataFrame, cols=("P10", "P50", "P90", "mean")) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].map(_eur)
    if "share" in out:
        out["share"] = out["share"].map(lambda s: f"{100 * s:.1f} %")
    return out


def format_report(
    result: SimulationResult,
    dropout: pd.DataFrame | None = None,
    tornado: pd.DataFrame | None = None,
    budget: float | None = None,
    recruit_by: float | None = None,
) -> str:
    d = result.design
    s = summary(result)
    lines = [
        f"Design: {d.name} — N = {d.n_randomised}, {d.n_sites} sites, "
        f"{len(d.countries)} countries, {result.n_sims:,} simulations",
        "",
        "Headline (EUR 2026)",
        _quantile_line("Total cost", s.loc["total_cost_eur"], _eur),
        _quantile_line("Duration", s.loc["duration_months"], _months),
        _quantile_line("Recruitment LPI", s.loc["lpi_months"], _months),
        f"  Per randomised   P50 {_eur(s.loc['cost_per_randomised_eur', 'P50'])}",
    ]
    for label, p in probabilities(result, budget, recruit_by).items():
        lines.append(f"  {label}: {100 * p:.0f} %")
    for by in ("category", "resource", "country"):
        lines += ["", f"By {by}", _money_table(breakdown(result, by)).to_string()]
    if dropout is not None:
        dp = dropout
        lines += [
            "",
            "Dropout cost (vs same design with no dropout, same random numbers)",
            f"  Completers               P50 {dp.loc['completers', 'P50']:.1f} of {d.n_randomised}",
            f"  Cost per completer       P50 {_eur(dp.loc['cost_per_completer_eur', 'P50'])} "
            f"(no dropout: {_eur(dp.loc['cost_per_completer_no_dropout_eur', 'P50'])}, "
            f"+{dp.loc['completer_cost_premium_pct', 'P50']:.1f} %)",
            f"  Total cost change        P50 {_eur(dp.loc['dropout_delta_cost_eur', 'P50'])} "
            f"({dp.loc['dropout_delta_cost_pct', 'P50']:+.1f} %): fewer visits, same fixed costs",
        ]
    if tornado is not None:
        t = tornado.copy()
        t["at low"] = t["cost_at_low"].map(_eur) + " (" + t["low_input"] + ")"
        t["at high"] = t["cost_at_high"].map(_eur) + " (" + t["high_input"] + ")"
        t["swing"] = t["swing"].map(_eur)
        lines += [
            "",
            f"Tornado: mean total cost, base {_eur(tornado['base'].iloc[0])} "
            "(input at P10 / P90, or 0.5x / 1.5x if fixed)",
            t[["parameter", "at low", "at high", "swing"]].to_string(index=False),
        ]
    return "\n".join(lines)


def write_outputs(
    out: Path,
    result: SimulationResult,
    dropout: pd.DataFrame | None = None,
    tornado: pd.DataFrame | None = None,
    budget: float | None = None,
    recruit_by: float | None = None,
    lines: bool = False,
) -> list[Path]:
    """Write the report tables as CSV files; return the paths written."""
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {"summary": summary(result)}
    probs = probabilities(result, budget, recruit_by)
    if probs:
        tables["probabilities"] = pd.Series(probs, name="probability").to_frame()
    for by in ("category", "resource", "country", "item"):
        tables[f"breakdown_{by}"] = breakdown(result, by)
    if dropout is not None:
        tables["dropout"] = dropout
    if tornado is not None:
        tables["tornado"] = tornado
    written = []
    for name, df in tables.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=name != "tornado")
        written.append(path)
    if lines:
        path = out / "cost_lines.csv.gz"
        result.costs.to_frame().to_csv(path, index=False)
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Scenario comparison
# --------------------------------------------------------------------------- #


def compare(results: dict[str, SimulationResult]) -> pd.DataFrame:
    """One row per scenario: P50 / P10 / P90 cost and duration, and the change vs the first.

    Run all scenarios with the same seed: with common random numbers (D32) the
    differences reflect the design changes, not Monte Carlo noise.
    """
    rows = []
    for name, r in results.items():
        p10, p50, p90 = np.quantile(r.total, [0.1, 0.5, 0.9])
        rows.append(
            {
                "scenario": name,
                "cost_P10": p10,
                "cost_P50": p50,
                "cost_P90": p90,
                "cost_mean": r.total.mean(),
                "duration_P50": float(np.median(r.duration)),
                "lpi_P50": float(np.median(r.quantities.t_lpi)),
                "cost_per_patient_P50": float(np.median(r.total / r.design.n_randomised)),
            }
        )
    df = pd.DataFrame(rows).set_index("scenario")
    base = df.iloc[0]
    df["delta_cost_P50"] = df["cost_P50"] - base["cost_P50"]
    df["delta_cost_P50_pct"] = 100 * (df["cost_P50"] / base["cost_P50"] - 1)
    df["delta_duration_P50"] = df["duration_P50"] - base["duration_P50"]
    return df


def category_deltas(
    base: SimulationResult, other: SimulationResult, by: str = "category"
) -> pd.Series:
    """Change in mean cost per group (other - base), groups present in either, sorted by |Δ|."""
    a = base.costs.by(by).mean(axis=1)
    b = other.costs.by(by).mean(axis=1)
    delta = b.reindex(a.index.union(b.index), fill_value=0) - a.reindex(
        a.index.union(b.index), fill_value=0
    )
    return delta.reindex(delta.abs().sort_values(ascending=False).index)
