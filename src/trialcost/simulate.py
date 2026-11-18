"""Monte Carlo orchestration and command-line entry point.

    uv run python -m trialcost.simulate designs/rare-neuro-ph2-mri-scales.yaml \\
        --n-sims 10000 --seed 42 [--budget 5e6] [--recruit-by 24] [--out results/]

One run = recruitment -> quantities -> costs for ``n_sims`` simulated trials.
The seed feeds one ``numpy.random.Generator``; each stage gets its own child
stream (``rng.spawn``), and each random input inside a stage its own child
too, so two runs with the same seed share every random number that the
scenario change does not touch (common random numbers, D29). This is what
makes the dropout comparison and the tornado precise with few simulations.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trialcost.catalogue import Catalogue, load_catalogue
from trialcost.costing import CostLines, compute_costs
from trialcost.design import Design, load_design
from trialcost.quantities import Quantities, compute_quantities
from trialcost.recruitment import Recruitment, simulate_recruitment


@dataclass(frozen=True)
class SimulationResult:
    design: Design
    recruitment: Recruitment
    quantities: Quantities
    costs: CostLines

    @property
    def n_sims(self) -> int:
        return self.costs.n_sims

    @property
    def total(self) -> np.ndarray:
        """Total sponsor cost per simulated trial, EUR 2026."""
        return self.costs.total()

    @property
    def duration(self) -> np.ndarray:
        """Study start to CSR, months."""
        return self.quantities.t_total


def run(
    design: Design,
    catalogue: Catalogue,
    n_sims: int,
    seed: int,
    deterministic: bool = False,
) -> SimulationResult:
    """Simulate ``n_sims`` trials of ``design``."""
    catalogue.check_design(design)
    rng = np.random.default_rng(seed)
    s_recruit, s_quantities, s_costs = rng.spawn(3)
    rec = simulate_recruitment(design, catalogue, s_recruit, n_sims, deterministic)
    q = compute_quantities(design, catalogue, rec, s_quantities, deterministic)
    costs = compute_costs(design, catalogue, q, s_costs, deterministic)
    return SimulationResult(design=design, recruitment=rec, quantities=q, costs=costs)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m trialcost.simulate",
        description="Monte Carlo cost and duration estimate of a clinical trial design.",
    )
    p.add_argument("design", type=Path, help="design YAML file")
    p.add_argument("--n-sims", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", help="every input at its central value")
    p.add_argument("--budget", type=float, help="EUR; report P(total cost <= budget)")
    p.add_argument("--recruit-by", type=float, help="months; report P(LPI <= this)")
    p.add_argument("--tornado-sims", type=int, default=2_000, help="sims per tornado run")
    p.add_argument("--no-tornado", action="store_true")
    p.add_argument("--out", type=Path, help="directory for CSV outputs")
    p.add_argument(
        "--lines", action="store_true", help="also write the tidy cost lines (large file)"
    )
    p.add_argument("--data-dir", type=Path, help="catalogue directory (default: data/)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Imported here: report.py imports run() from this module.
    from trialcost import report

    args = _parse_args(argv)
    catalogue = load_catalogue(args.data_dir) if args.data_dir else load_catalogue()
    design = load_design(args.design)
    result = run(design, catalogue, args.n_sims, args.seed, args.deterministic)

    dropout = report.dropout_report(design, catalogue, args.n_sims, args.seed, args.deterministic)
    tornado = None
    if not (args.no_tornado or args.deterministic):
        tornado = report.tornado(design, catalogue, args.tornado_sims, args.seed)

    print(
        report.format_report(
            result, dropout=dropout, tornado=tornado, budget=args.budget, recruit_by=args.recruit_by
        )
    )
    if args.out:
        written = report.write_outputs(
            args.out,
            result,
            dropout=dropout,
            tornado=tornado,
            budget=args.budget,
            recruit_by=args.recruit_by,
            lines=args.lines,
        )
        print(f"\nWrote {len(written)} files to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
