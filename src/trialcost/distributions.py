"""Probability distributions for uncertain inputs, with vectorised sampling.

Every numeric input of a design (and, later, every catalogue price) is one of
four distributions. Each one knows how to:

- ``sample(rng, size)``: draw an array of shape ``size`` from an explicit
  ``numpy.random.Generator`` (no global random state);
- ``central()``: the single value used in deterministic mode (golden tests);
- ``expectation()``: the analytical mean, used to check sampling in tests;
- ``quantile(p)``: inverse CDF, used by the tornado (P10 / P90 swings);
- ``is_point_mass``: True when the distribution has no spread.

``central()`` is the parameter the user wrote down for that distribution, so
deterministic totals can be checked by hand:

=============  ==========  ===================================================
Distribution   central()   Why
=============  ==========  ===================================================
Fixed          value       only value
Triangular     mode        "most likely" value given by the user
LogNormal      median      the parameter given; the mode is lower and the
                           mean higher, both depend on ``gsd``
Beta           mean        the parameter given; for proportions the mean is
                           what multiplies counts (e.g. N / (1 - SF))
=============  ==========  ===================================================

A plain number in YAML / Python is coerced to ``Fixed`` (see ``Dist``).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, BeforeValidator, Field, model_validator

Size = int | tuple[int, ...]


class Fixed(BaseModel):
    """A known value: every draw returns ``value``."""

    kind: Literal["fixed"] = "fixed"
    value: float

    def sample(self, rng: np.random.Generator, size: Size) -> np.ndarray:
        return np.full(size, self.value, dtype=float)

    def central(self) -> float:
        return self.value

    def expectation(self) -> float:
        return self.value

    def quantile(self, p: float) -> float:
        return self.value

    @property
    def is_point_mass(self) -> bool:
        return True


class Triangular(BaseModel):
    """Triangular(low, mode, high): bounded expert range with a most likely value.

    Mean = (low + mode + high) / 3. If ``low == high`` the distribution is a
    point mass (NumPy's sampler refuses that case, so it is handled here).
    """

    kind: Literal["triangular"] = "triangular"
    low: float
    mode: float
    high: float

    @model_validator(mode="after")
    def _check(self):
        if not self.low <= self.mode <= self.high:
            raise ValueError("need low <= mode <= high")
        return self

    def sample(self, rng: np.random.Generator, size: Size) -> np.ndarray:
        if self.low == self.high:
            return np.full(size, self.low, dtype=float)
        return rng.triangular(self.low, self.mode, self.high, size)

    def central(self) -> float:
        return self.mode

    def expectation(self) -> float:
        return (self.low + self.mode + self.high) / 3

    def quantile(self, p: float) -> float:
        """Closed-form inverse CDF: the mode splits the area in c = (m - a) / (b - a)."""
        a, m, b = self.low, self.mode, self.high
        if a == b:
            return a
        c = (m - a) / (b - a)
        if p < c:
            return a + math.sqrt(p * (b - a) * (m - a))
        return b - math.sqrt((1 - p) * (b - a) * (b - m))

    @property
    def is_point_mass(self) -> bool:
        return self.low == self.high


class LogNormal(BaseModel):
    """Log-normal given by its median and geometric standard deviation (gsd).

    If X ~ LogNormal then ln X ~ Normal(mu, sigma) with mu = ln(median) and
    sigma = ln(gsd). Roughly 68 % of draws fall in [median / gsd, median * gsd]
    and 95 % in [median / gsd^2, median * gsd^2].
    Mean = median * exp(sigma^2 / 2) > median (right-skewed).
    """

    kind: Literal["lognormal"] = "lognormal"
    median: float = Field(gt=0)
    gsd: float = Field(gt=1)

    @property
    def mu(self) -> float:
        return math.log(self.median)

    @property
    def sigma(self) -> float:
        return math.log(self.gsd)

    def sample(self, rng: np.random.Generator, size: Size) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma, size)

    def central(self) -> float:
        return self.median

    def expectation(self) -> float:
        return self.median * math.exp(self.sigma**2 / 2)

    def quantile(self, p: float) -> float:
        return math.exp(self.mu + self.sigma * NormalDist().inv_cdf(p))

    @property
    def is_point_mass(self) -> bool:
        return False


class Beta(BaseModel):
    """Beta for proportions, given by its mean and concentration kappa.

    Shape parameters: alpha = mean * kappa, beta = (1 - mean) * kappa.
    Variance = mean * (1 - mean) / (kappa + 1): larger kappa = tighter.
    Example: mean 0.35, kappa 20 gives SD ~0.10.
    """

    kind: Literal["beta"] = "beta"
    mean: float = Field(gt=0, lt=1)
    kappa: float = Field(gt=0)

    @property
    def alpha(self) -> float:
        return self.mean * self.kappa

    @property
    def beta(self) -> float:
        return (1 - self.mean) * self.kappa

    def sample(self, rng: np.random.Generator, size: Size) -> np.ndarray:
        return rng.beta(self.alpha, self.beta, size)

    def central(self) -> float:
        return self.mean

    def expectation(self) -> float:
        return self.mean

    def quantile(self, p: float) -> float:
        """Inverse CDF by bisection on the regularised incomplete beta function."""
        lo, hi = 0.0, 1.0
        for _ in range(60):  # 2^-60: far below any useful precision
            mid = (lo + hi) / 2
            if _beta_cdf(mid, self.alpha, self.beta) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    @property
    def is_point_mass(self) -> bool:
        return False


def _beta_cdf(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta I_x(a, b) (Numerical Recipes, 3rd ed., §6.4).

    I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * CF(x), where CF is a continued
    fraction that converges fast for x < (a + 1) / (a + b + 2); otherwise use
    the symmetry I_x(a, b) = 1 - I_{1-x}(b, a).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return math.exp(log_front) * _beta_cf(x, a, b) / a
    return 1 - math.exp(log_front) * _beta_cf(1 - x, b, a) / b


def _beta_cf(x: float, a: float, b: float) -> float:
    """Continued fraction for the incomplete beta (modified Lentz algorithm)."""
    tiny, eps = 1e-300, 1e-15
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1 + num * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < eps:
            break
    return h


def _coerce(v):
    return {"kind": "fixed", "value": v} if isinstance(v, (int, float)) else v


#: Any concrete distribution (for type hints on engine functions).
AnyDist = Fixed | Triangular | LogNormal | Beta

#: Pydantic field type: discriminated on ``kind``; a plain number becomes ``Fixed``.
Dist = Annotated[
    Fixed | Triangular | LogNormal | Beta,
    Field(discriminator="kind"),
    BeforeValidator(_coerce),
]


def draw(
    dist: AnyDist,
    rng: np.random.Generator,
    size: Size,
    deterministic: bool = False,
) -> np.ndarray:
    """Sample ``dist``, or fill with its central value in deterministic mode.

    This is the single entry point the engine should use, so that switching
    the whole model to deterministic mode is one flag.
    """
    if deterministic:
        return np.full(size, dist.central(), dtype=float)
    return dist.sample(rng, size)
