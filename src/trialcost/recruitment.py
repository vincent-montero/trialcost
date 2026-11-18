"""Site activation and Poisson-gamma recruitment, vectorised over simulations.

Model (Anisimov & Fedorov; README §2.4)
---------------------------------------
- Site ``i`` opens at ``a_i`` months after study start, drawn from its
  country's activation distribution (design value, else ``countries.csv``).
- Its recruitment rate is ``lambda_i ~ Gamma(shape, scale = rate_mean / shape)``,
  i.e. mean ``rate_mean`` and coefficient of variation ``1 / sqrt(shape)``.
  ``rate_mean`` itself is uncertain and drawn once per simulation, so all
  sites of one simulated trial share it (a "slow indication" moves them all).
- Once open, site ``i`` randomises patients as a Poisson process of rate
  ``lambda_i``.

Time of the N-th randomisation (LPI), without a time grid
---------------------------------------------------------
Given the rates and activation times, the expected number of patients
randomised by month ``t`` is the cumulative intensity::

    M(t) = sum_i lambda_i * max(0, t - a_i)

which is piecewise linear and increasing in ``t``, with a kink at each
activation. A Poisson process with cumulative intensity ``M`` is a unit-rate
Poisson process run on the clock ``M(t)`` (time-change theorem). The N-th
event of a unit-rate process happens at ``S ~ Gamma(N, 1)`` (sum of N unit
exponentials), so::

    T_LPI = M^{-1}(S)

``M^{-1}`` is found exactly: sort sites by activation, compute ``M`` at each
activation, pick the segment that contains ``S`` and solve the linear piece.
Everything is array arithmetic over ``(n_sims, n_sites)``.

Random streams
--------------
Each random input gets its own child generator (``rng.spawn``): activation of
each country, ``rate_mean``, site rates and the Gamma(N, 1) target. Pinning one
input (tornado, scenarios) then leaves every other draw unchanged (D29).

Deterministic mode
------------------
Activation at its central value, ``rate_mean`` at its median, every site at
``lambda_i = rate_mean`` (no between-site heterogeneity) and ``S = N`` (the
mean of Gamma(N, 1)), i.e. ``T_LPI`` solves ``M(T) = N``: hand-checkable.

Patients per site
-----------------
Costing needs patients per country, not the identity of each arrival. We use
the exact conditional expectation given ``T = T_LPI``: the first ``N - 1``
arrivals are spread over sites in proportion to ``lambda_i * (T - a_i)+``,
and the N-th one (which happens exactly at ``T``) comes from an open site in
proportion to ``lambda_i``. Counts are fractional and each row sums to N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trialcost.catalogue import Catalogue
from trialcost.design import Design, Population
from trialcost.distributions import draw


@dataclass(frozen=True)
class Recruitment:
    """Simulated recruitment for ``n_sims`` trials of one design."""

    site_country: np.ndarray  # (n_sites,) country code of each site
    activation: np.ndarray  # (n_sims, n_sites) months from study start
    site_rate: np.ndarray  # (n_sims, n_sites) randomised patients / month once open
    t_lpi: np.ndarray  # (n_sims,) months from study start to last patient in
    patients: np.ndarray  # (n_sims, n_sites) expected randomised patients, rows sum to N

    @property
    def n_sims(self) -> int:
        return self.activation.shape[0]

    @property
    def n_sites(self) -> int:
        return self.activation.shape[1]


def site_countries(design: Design) -> np.ndarray:
    """Country code of each site, in design order (e.g. FR, FR, FR, FR, DE, ...)."""
    return np.array([c.country for c in design.countries for _ in range(c.n_sites)])


def draw_activation(
    design: Design,
    catalogue: Catalogue,
    rng: np.random.Generator,
    n_sims: int,
    deterministic: bool = False,
) -> np.ndarray:
    """Activation month of every site, shape ``(n_sims, n_sites)``."""
    streams = rng.spawn(len(design.countries))
    blocks = [
        draw(catalogue.activation_months(design, c.country), s, (n_sims, c.n_sites), deterministic)
        for c, s in zip(design.countries, streams, strict=True)
    ]
    return np.hstack(blocks)


def draw_site_rates(
    population: Population,
    rng: np.random.Generator,
    n_sims: int,
    n_sites: int,
    deterministic: bool = False,
) -> np.ndarray:
    """Per-site rates ``lambda_i``, shape ``(n_sims, n_sites)`` (see module docstring)."""
    s_mean, s_sites = rng.spawn(2)
    rate_mean = draw(population.rate_mean, s_mean, n_sims, deterministic)[:, None]
    if deterministic:
        return np.broadcast_to(rate_mean, (n_sims, n_sites)).copy()
    shape = population.rate_shape
    return s_sites.gamma(shape, rate_mean / shape, (n_sims, n_sites))


def time_to_intensity(activation: np.ndarray, rates: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve ``M(T) = target`` for each simulation (``M`` from the module docstring).

    Parameters
    ----------
    activation, rates : (n_sims, n_sites)
    target : (n_sims,), the cumulative intensity to reach (> 0)

    Returns
    -------
    T : (n_sims,), months from study start
    """
    order = np.argsort(activation, axis=1)
    a = np.take_along_axis(activation, order, axis=1)  # sorted activation times
    r = np.take_along_axis(rates, order, axis=1)

    # total_rate[:, k]: summed rate of the k+1 earliest sites, valid on [a_k, a_{k+1})
    total_rate = np.cumsum(r, axis=1)
    # m_at[:, k] = M(a_k); M(a_0) = 0 and each segment adds rate * duration
    seg = total_rate[:, :-1] * np.diff(a, axis=1)
    m_at = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(seg, axis=1)], axis=1)

    # last activation whose M is still <= target: the segment containing the solution
    k = (m_at <= target[:, None]).sum(axis=1) - 1
    rows = np.arange(a.shape[0])
    return a[rows, k] + (target - m_at[rows, k]) / total_rate[rows, k]


def expected_patients_per_site(
    activation: np.ndarray, rates: np.ndarray, t_lpi: np.ndarray, n: int
) -> np.ndarray:
    """Expected randomised patients per site given ``T_LPI`` (see module docstring)."""
    t = t_lpi[:, None]
    exposure = rates * np.clip(t - activation, 0.0, None)  # lambda_i * (T - a_i)+
    open_rate = rates * (activation < t)
    first = exposure / exposure.sum(axis=1, keepdims=True)
    last = open_rate / open_rate.sum(axis=1, keepdims=True)
    return (n - 1) * first + last


def simulate_recruitment(
    design: Design,
    catalogue: Catalogue,
    rng: np.random.Generator,
    n_sims: int,
    deterministic: bool = False,
) -> Recruitment:
    """Draw activation, site rates and LPI for ``n_sims`` simulated trials."""
    n = design.n_randomised
    s_act, s_rates, s_target = rng.spawn(3)
    activation = draw_activation(design, catalogue, s_act, n_sims, deterministic)
    rates = draw_site_rates(design.population, s_rates, n_sims, design.n_sites, deterministic)
    target = np.full(n_sims, float(n)) if deterministic else s_target.gamma(n, 1.0, n_sims)
    t_lpi = time_to_intensity(activation, rates, target)
    return Recruitment(
        site_country=site_countries(design),
        activation=activation,
        site_rate=rates,
        t_lpi=t_lpi,
        patients=expected_patients_per_site(activation, rates, t_lpi, n),
    )
