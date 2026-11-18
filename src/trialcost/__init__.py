"""Design-driven Monte Carlo estimator of a clinical trial's total sponsor budget."""

import os
from pathlib import Path

__version__ = "0.1.0"


def _find_data_dir() -> Path:
    """Repository ``data/`` (CSV catalogue, source of truth for every number).

    Order: ``TRIALCOST_DATA_DIR`` if set; the repository checkout this package
    lives in (editable install, the normal case); ``./data`` from the working
    directory (e.g. a hosted app started at the repository root).
    """
    candidates = []
    if env := os.environ.get("TRIALCOST_DATA_DIR"):
        candidates.append(Path(env))
    candidates += [Path(__file__).resolve().parents[2] / "data", Path.cwd() / "data"]
    for path in candidates:
        if (path / "study_items.csv").exists():
            return path
    raise FileNotFoundError(
        "Cannot find the data/ catalogue; set TRIALCOST_DATA_DIR. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


#: Repository data directory (CSV catalogue, source of truth for every number).
DATA_DIR = _find_data_dir()
