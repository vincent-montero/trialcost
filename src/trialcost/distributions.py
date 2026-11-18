"""Probability distributions for uncertain inputs, with vectorised sampling.

Every numeric input of a design (and, later, every catalogue price) is one of
four distributions. Each one knows how to:

- ``sample(rng, size)``: draw an array of shape ``size`` from an explicit
  ``numpy.random.Generator`` (no global random state);
- ``central()``: the single value used in deterministic mode (golden tests);
- ``expectation()``: the analytical mean, used to check sampling in tests.

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
