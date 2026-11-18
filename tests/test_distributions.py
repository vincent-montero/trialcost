import math

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from trialcost.distributions import Beta, Dist, Fixed, LogNormal, Triangular, draw

N = 200_000
DIST = TypeAdapter(Dist)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_plain_number_coerces_to_fixed():
    assert DIST.validate_python(2.5) == Fixed(value=2.5)
    assert DIST.validate_python(3) == Fixed(value=3.0)


def test_dict_dispatches_on_kind():
    d = DIST.validate_python({"kind": "beta", "mean": 0.3, "kappa": 10})
    assert isinstance(d, Beta)


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "triangular", "low": 3, "mode": 1, "high": 5},
        {"kind": "lognormal", "median": 0.25, "gsd": 1.0},
        {"kind": "lognormal", "median": 0, "gsd": 1.5},
        {"kind": "beta", "mean": 1.0, "kappa": 10},
        {"kind": "beta", "mean": 0.3, "kappa": 0},
        {"kind": "uniform", "low": 0, "high": 1},
    ],
)
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValidationError):
        DIST.validate_python(bad)


@pytest.mark.parametrize(
    "dist",
    [
        Fixed(value=4.0),
        Triangular(low=3, mode=5, high=9),
        LogNormal(median=0.25, gsd=1.6),
        Beta(mean=0.35, kappa=20),
    ],
)
def test_sample_mean_matches_expectation(dist, rng):
    x = dist.sample(rng, N)
    assert x.shape == (N,)
    assert x.mean() == pytest.approx(dist.expectation(), rel=0.01)


def test_triangular_stays_in_bounds_and_handles_point_mass(rng):
    x = Triangular(low=3, mode=5, high=9).sample(rng, N)
    assert x.min() >= 3 and x.max() <= 9
    assert np.all(Triangular(low=2, mode=2, high=2).sample(rng, 10) == 2)


def test_lognormal_median_and_spread(rng):
    d = LogNormal(median=0.25, gsd=1.6)
    x = d.sample(rng, N)
    assert np.median(x) == pytest.approx(0.25, rel=0.01)
    assert np.log(x).std() == pytest.approx(math.log(1.6), rel=0.01)


def test_beta_variance(rng):
    d = Beta(mean=0.35, kappa=20)
    x = d.sample(rng, N)
    assert x.var() == pytest.approx(0.35 * 0.65 / 21, rel=0.02)


def test_sampling_supports_2d_shapes(rng):
    assert Triangular(low=3, mode=5, high=9).sample(rng, (100, 15)).shape == (100, 15)


def test_same_seed_same_draws():
    d = LogNormal(median=1, gsd=2)
    a = d.sample(np.random.default_rng(7), 5)
    b = d.sample(np.random.default_rng(7), 5)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    ("dist", "central"),
    [
        (Fixed(value=4.0), 4.0),
        (Triangular(low=3, mode=5, high=9), 5.0),
        (LogNormal(median=0.25, gsd=1.6), 0.25),
        (Beta(mean=0.35, kappa=20), 0.35),
    ],
)
def test_deterministic_draw_uses_central_value(dist, central, rng):
    x = draw(dist, rng, (3, 2), deterministic=True)
    assert x.shape == (3, 2)
    assert np.all(x == central)
