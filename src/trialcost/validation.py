"""Benchmark validation report (README §7.2). Never asserted in tests.

    uv run python -m trialcost.validation [--n-sims 5000] [--seed 42] [--out reports/]

Why reference designs
---------------------
Published figures ("a CNS phase 2 trial costs $X per patient") are averages
over a mixture of designs (README §2.1). Comparing them with our 50-patient
rare-disease archetype would mostly measure the difference in trial size. Each
benchmark is therefore compared with a design that resembles the trials it
summarises (``data/benchmark_checks.csv`` says which design, which metric and
how comparable the two are).

Converting benchmarks to EUR 2026
---------------------------------
USD values are inflated with the US CPI-U Medical Care index (as ASPE did for
its own figures) from the benchmark's price year to 2026, then converted at
the fixed FX rate (``constants.csv``)::

    EUR_2026 = USD_year x CPI_2026 / CPI_year / fx_usd_per_eur

What is compared
----------------
The model gives a distribution of outcomes for one design (parameter
uncertainty); a benchmark is a mean or median over many real trials. They are
not the same object. The report shows the model's P10 / P50 / P90, the model
statistic named in the check (mean or median of the simulated distribution)
and the ratio model / benchmark. "Inside" means the benchmark falls inside the
model's P10-P90 band. Comparability (good / fair / poor) is a judgement
recorded with each check.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trialcost import DATA_DIR
from trialcost.catalogue import Catalogue, load_catalogue
from trialcost.design import DAYS_PER_MONTH, load_design
from trialcost.simulate import SimulationResult, run

DESIGNS_DIR = DATA_DIR.parent / "designs"
TARGET_PRICE_YEAR = 2026  # D12: every model value is EUR, price year 2026
CPI_INDEX = "us_cpi_medical_care"

#: Model metric name -> function of a simulation result (per-simulation array).
METRICS = {
    "total_cost": lambda r: r.total,
    "cost_per_patient": lambda r: r.total / r.design.n_randomised,
    "cost_per_patient_visit": lambda r: r.total / r.quantities.visit_counts.sum(axis=(1, 2)),
    "cost_per_day": lambda r: r.total / (r.duration * DAYS_PER_MONTH),
    "duration_months": lambda r: r.duration,
}
#: Benchmark unit -> unit of the model metric after conversion.
MONEY_UNITS = {"USD/trial", "USD/patient", "USD/patient-visit", "USD/day"}
CHECK_COLUMNS = [
    "benchmark_id", "design", "metric", "statistic", "price_year", "comparability", "notes",
]  # fmt: skip


@dataclass(frozen=True)
class ValidationInputs:
    checks: pd.DataFrame  # benchmark_checks.csv joined with benchmarks.csv
    cpi: pd.Series  # year -> index value


def load_inputs(catalogue: Catalogue, data_dir: Path = DATA_DIR) -> ValidationInputs:
    """Read and check ``benchmark_checks.csv`` and ``price_indices.csv``."""
    checks = pd.read_csv(data_dir / "benchmark_checks.csv")
    indices = pd.read_csv(data_dir / "price_indices.csv")
    problems = []
    if list(checks.columns) != CHECK_COLUMNS:
        problems.append(f"benchmark_checks: columns {list(checks.columns)} != {CHECK_COLUMNS}")
    cpi = indices[indices["index_id"] == CPI_INDEX].set_index("year")["value"]
    if TARGET_PRICE_YEAR not in cpi.index:
        problems.append(f"price_indices: {CPI_INDEX} has no {TARGET_PRICE_YEAR} value")
    for _, c in checks.iterrows():
        tag = f"benchmark_checks: {c.benchmark_id}"
        if c.benchmark_id not in catalogue.benchmarks.index:
            problems.append(f"{tag}: unknown benchmark")
            continue
        unit = catalogue.benchmarks.loc[c.benchmark_id, "unit"]
        if c.metric not in METRICS:
            problems.append(f"{tag}: unknown metric {c.metric!r}")
        if c.statistic not in ("mean", "median"):
            problems.append(f"{tag}: statistic must be mean or median")
        if c.comparability not in ("good", "fair", "poor"):
            problems.append(f"{tag}: comparability must be good / fair / poor")
        if not (DESIGNS_DIR / f"{c.design}.yaml").exists():
            problems.append(f"{tag}: design file {c.design}.yaml not found")
        if unit in MONEY_UNITS and (pd.isna(c.price_year) or int(c.price_year) not in cpi.index):
            problems.append(f"{tag}: no {CPI_INDEX} value for price year {c.price_year}")
    if problems:
        raise ValueError("validation inputs:\n" + "\n".join(f"- {p}" for p in problems))
    joined = checks.join(
        catalogue.benchmarks[["description", "value", "unit", "source"]], on="benchmark_id"
    )
    return ValidationInputs(checks=joined, cpi=cpi)


def to_eur_2026(value: float, unit: str, price_year, cpi: pd.Series, fx: float) -> float:
    """Inflate a USD benchmark to 2026 and convert to EUR; non-money units pass through."""
    if unit not in MONEY_UNITS:
        return value
    return value * cpi[TARGET_PRICE_YEAR] / cpi[int(price_year)] / fx


def run_validation(
    catalogue: Catalogue, n_sims: int = 5_000, seed: int = 42, data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """One row per benchmark check: converted benchmark, model quantiles, ratio."""
    inputs = load_inputs(catalogue, data_dir)
    fx = catalogue.constant("fx_usd_per_eur").central()
    results: dict[str, SimulationResult] = {}
    rows = []
    for _, c in inputs.checks.iterrows():
        if c.design not in results:
            design = load_design(DESIGNS_DIR / f"{c.design}.yaml")
            results[c.design] = run(design, catalogue, n_sims, seed)
        x = METRICS[c.metric](results[c.design])
        bench = to_eur_2026(c.value, c.unit, c.price_year, inputs.cpi, fx)
        model_stat = float(np.mean(x) if c.statistic == "mean" else np.median(x))
        p10, p50, p90 = np.quantile(x, [0.1, 0.5, 0.9])
        rows.append(
            {
                "benchmark_id": c.benchmark_id,
                "description": c.description,
                "design": c.design,
                "metric": c.metric,
                "statistic": c.statistic,
                "benchmark_original": f"{_number(c.value)} {c.unit} ({_year_label(c.price_year)})",
                "benchmark_2026": bench,
                "model_P10": p10,
                "model_P50": p50,
                "model_P90": p90,
                "model_stat": model_stat,
                "ratio_model_to_benchmark": model_stat / bench,
                "inside_P10_P90": bool(p10 <= bench <= p90),
                "comparability": c.comparability,
                "notes": c.notes,
            }
        )
    return pd.DataFrame(rows)


def _number(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"


def _year_label(price_year) -> str:
    return "no price year" if pd.isna(price_year) else f"{int(price_year)} USD"


def _fmt(value: float, metric: str) -> str:
    if metric == "duration_months":
        return f"{value:.1f} mo"
    return f"€{value / 1e6:,.2f}M" if abs(value) >= 1e6 else f"€{value / 1e3:,.1f}k"


def format_markdown(df: pd.DataFrame, n_sims: int, seed: int, fx: float) -> str:
    """Markdown report: one table plus the caveats that come with it."""
    lines = [
        "# Benchmark validation report",
        "",
        f"Engine v0, {n_sims:,} simulations per design, seed {seed}. Benchmarks inflated to "
        f"2026 with the US CPI-U Medical Care index and converted at 1 EUR = {fx} USD. "
        "Generated by `python -m trialcost.validation`; method in `src/trialcost/validation.py`.",
        "",
        "| Benchmark | Design | Metric | Benchmark (EUR 2026) | Model P10 / P50 / P90 | "
        "Model stat | Ratio | Inside | Comparability |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        m = r.metric
        band = " / ".join(_fmt(r[k], m) for k in ("model_P10", "model_P50", "model_P90"))
        lines.append(
            f"| {r.description} | `{r.design}` | {m} ({r.statistic}) | "
            f"{_fmt(r.benchmark_2026, m)} | {band} | "
            f"{_fmt(r.model_stat, m)} | {r.ratio_model_to_benchmark:.2f} | "
            f"{'yes' if r.inside_P10_P90 else 'no'} | {r.comparability} |"
        )
    lines += [
        "",
        "Ratio = model statistic / benchmark. Inside = benchmark within the model's P10-P90 band.",
        "",
        "## Notes per check",
        "",
    ]
    lines += [
        f"- **{r.benchmark_id}** ({r.benchmark_original}): {r.notes}" for _, r in df.iterrows()
    ]
    lines += [
        "",
        "## Caveats",
        "",
        "- A benchmark is a mean or median over many real trials with different designs; the model "
        "band is parameter uncertainty for one design. Agreement within ~±30 % is the most that "
        "can be expected; a systematic gap in one direction is the useful signal.",
        "- Reference designs (`designs/ref-*.yaml`) are assumptions (README §6 #19).",
        "- Benchmarks are US-centric; reference designs mix US and EU sites.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m trialcost.validation", description="Benchmark validation report."
    )
    p.add_argument("--n-sims", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, help="directory for benchmark-validation.md / .csv")
    args = p.parse_args(argv)
    catalogue = load_catalogue()
    df = run_validation(catalogue, args.n_sims, args.seed)
    text = format_markdown(
        df, args.n_sims, args.seed, catalogue.constant("fx_usd_per_eur").central()
    )
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "benchmark-validation.md").write_text(text + "\n", encoding="utf-8")
        df.to_csv(args.out / "benchmark-validation.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
