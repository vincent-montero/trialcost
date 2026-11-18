import time
from pathlib import Path

import numpy as np
import pytest

from trialcost.catalogue import load_catalogue
from trialcost.design import load_design
from trialcost.distributions import Fixed
from trialcost.recruitment import (
    draw_activation,
    draw_site_rates,
    expected_patients_per_site,
    simulate_recruitment,
    site_countries,
    time_to_intensity,
)

ARCHETYPE = Path(__file__).resolve().parents[1] / "designs" / "rare-neuro-ph2-mri-scales.yaml"


@pytest.fixture(scope="module")
def cat():
    return load_catalogue()


@pytest.fixture
def design():
    return load_design(ARCHETYPE)


def brute_force(activation, rates, n, reps, rng):
    """Reference: simulate each site's arrival times explicitly, take the N-th overall.

    The N-th arrival overall is among the first N arrivals of each site, so N
    inter-arrival exponentials per site are enough. Returns (t_lpi, counts).
    """
    gaps = rng.exponential(1 / rates[None, :, None], (reps, len(rates), n))
    times = activation[None, :, None] + np.cumsum(gaps, axis=2)  # (reps, sites, n)
    t_lpi = np.partition(times.reshape(reps, -1), n - 1, axis=1)[:, n - 1]
    counts = (times <= t_lpi[:, None, None]).sum(axis=2)
    return t_lpi, counts


# --- time_to_intensity: exact maths ------------------------------------------ #


def test_single_site_is_linear():
    t = time_to_intensity(np.array([[2.0]]), np.array([[0.5]]), np.array([10.0]))
    assert t == pytest.approx([2 + 10 / 0.5])


def test_piecewise_hand_example():
    # sites open at 0 (rate 1) and 4 (rate 2): M(4) = 4; then slope 3.
    act, rates = np.array([[4.0, 0.0]]), np.array([[2.0, 1.0]])  # unsorted on purpose
    assert time_to_intensity(act, rates, np.array([3.0])) == pytest.approx([3.0])
    assert time_to_intensity(act, rates, np.array([10.0])) == pytest.approx([4 + 6 / 3])


def test_matches_brute_force_poisson():
    rng = np.random.default_rng(1)
    act = np.array([5.0, 5.0, 6.0, 8.0, 12.0])
    rates = np.array([0.3, 0.1, 0.25, 0.5, 0.2])
    n, reps = 30, 20_000
    ref, _ = brute_force(act, rates, n, reps, rng)
    target = rng.gamma(n, 1.0, reps)
    ours = time_to_intensity(np.tile(act, (reps, 1)), np.tile(rates, (reps, 1)), target)
    assert ours.mean() == pytest.approx(ref.mean(), rel=0.01)
    for q in (0.1, 0.5, 0.9):
        assert np.quantile(ours, q) == pytest.approx(np.quantile(ref, q), rel=0.02)


# --- patients per site ------------------------------------------------------- #


def test_expected_patients_match_brute_force():
    rng = np.random.default_rng(2)
    act = np.array([0.0, 2.0, 6.0, 30.0])  # last site usually opens too late
    rates = np.array([0.5, 0.2, 0.8, 1.0])
    n, reps = 20, 20_000
    t_ref, counts = brute_force(act, rates, n, reps, rng)
    ours = expected_patients_per_site(np.tile(act, (reps, 1)), np.tile(rates, (reps, 1)), t_ref, n)
    np.testing.assert_allclose(ours.mean(axis=0), counts.mean(axis=0), atol=0.1)


def test_patients_sum_to_n_and_skip_unopened_sites():
    act = np.array([[0.0, 1.0, 50.0]])
    rates = np.array([[1.0, 1.0, 1.0]])
    t = time_to_intensity(act, rates, np.array([10.0]))
    p = expected_patients_per_site(act, rates, t, 10)
    assert p.sum() == pytest.approx(10)
    assert p[0, 2] == 0


# --- archetype ---------------------------------------------------------------- #


def test_site_countries(design):
    sc = site_countries(design)
    assert len(sc) == 15
    assert list(sc[:5]) == ["FR"] * 4 + ["DE"]


def test_deterministic_archetype_by_hand(design, cat):
    """Modes: FR/ES/US open at 5 (10 sites), DE at 6 (3), IT at 8 (2); rate 0.25 each.

    M(6) = 10 * 0.25 * 1 = 2.5; M(8) = 2.5 + 13 * 0.25 * 2 = 9;
    then slope 15 * 0.25 = 3.75, so T_LPI = 8 + (50 - 9) / 3.75 = 18.933 months.
    """
    rec = simulate_recruitment(design, cat, np.random.default_rng(0), 3, deterministic=True)
    np.testing.assert_allclose(rec.t_lpi, 8 + 41 / 3.75)
    np.testing.assert_allclose(rec.patients.sum(axis=1), 50)


def test_activation_uses_catalogue_and_design_override(design, cat):
    rng = np.random.default_rng(3)
    act = draw_activation(design, cat, rng, 5_000)
    it = site_countries(design) == "IT"
    assert act[:, it].min() >= 5 and act[:, it].max() <= 12  # countries.csv: 5-8-12
    design.countries[0].activation_months = Fixed(value=1.0)
    act = draw_activation(design, cat, rng, 10)
    assert np.all(act[:, :4] == 1.0)  # the 4 FR sites


def test_site_rates_mean_and_shared_rate_mean(design):
    rng = np.random.default_rng(4)
    pop = design.population
    rates = draw_site_rates(pop, rng, 100_000, 15)
    assert rates.mean() == pytest.approx(pop.rate_mean.expectation(), rel=0.02)
    # sites of one trial share rate_mean: rates are positively correlated across sites
    assert np.corrcoef(rates[:, 0], rates[:, 1])[0, 1] > 0.1


@pytest.mark.parametrize("change", ["more_sites", "faster_sites", "fewer_patients"])
def test_recruitment_gets_faster(design, cat, change):
    det = dict(rng=np.random.default_rng(0), n_sims=1, deterministic=True)
    base = simulate_recruitment(design, cat, **det).t_lpi[0]
    if change == "more_sites":
        design.countries[0].n_sites += 4
    elif change == "faster_sites":
        design.population.rate_mean = Fixed(value=0.5)
    else:
        design.n_randomised = 30
    assert simulate_recruitment(design, cat, **det).t_lpi[0] < base


def test_reproducible_with_seed(design, cat):
    a = simulate_recruitment(design, cat, np.random.default_rng(42), 100)
    b = simulate_recruitment(design, cat, np.random.default_rng(42), 100)
    np.testing.assert_array_equal(a.t_lpi, b.t_lpi)
    np.testing.assert_array_equal(a.patients, b.patients)


def test_stochastic_archetype_shapes_and_sanity(design, cat):
    rec = simulate_recruitment(design, cat, np.random.default_rng(42), 10_000)
    assert rec.patients.shape == (10_000, 15)
    np.testing.assert_allclose(rec.patients.sum(axis=1), 50)
    assert np.all(rec.t_lpi > rec.activation.min(axis=1))
    assert np.all(rec.patients[rec.activation >= rec.t_lpi[:, None]] == 0)


def test_fast_enough(design, cat):
    start = time.perf_counter()
    simulate_recruitment(design, cat, np.random.default_rng(0), 10_000)
    assert time.perf_counter() - start < 0.5
