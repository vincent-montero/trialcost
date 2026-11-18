from pathlib import Path

import numpy as np
import pytest

from trialcost.catalogue import load_catalogue
from trialcost.costing import compute_costs
from trialcost.design import load_design
from trialcost.quantities import compute_quantities
from trialcost.recruitment import simulate_recruitment

ARCHETYPE = Path(__file__).resolve().parents[1] / "designs" / "rare-neuro-ph2-mri-scales.yaml"


@pytest.fixture(scope="session")
def cat():
    return load_catalogue()


@pytest.fixture
def design():
    """Fresh archetype for each test (tests may mutate it)."""
    return load_design(ARCHETYPE)


def run_pipeline(design, cat, n_sims=1, seed=0, deterministic=True):
    """Recruitment -> quantities -> costs with one generator, as simulate.py will."""
    rng = np.random.default_rng(seed)
    rec = simulate_recruitment(design, cat, rng, n_sims, deterministic)
    q = compute_quantities(design, cat, rec, rng, deterministic)
    return q, compute_costs(design, cat, q, rng, deterministic)
