"""Design-driven Monte Carlo estimator of a clinical trial's total sponsor budget."""

from pathlib import Path

__version__ = "0.1.0"

#: Repository data directory (CSV catalogue, source of truth for every number).
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
