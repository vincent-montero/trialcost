"""Design + recruitment -> physical quantities (patients, visits, procedures, site-months...).

No prices here: only counts and durations, per simulation and per country.
Array shapes: ``(n_sims,)`` for study-level quantities, ``(n_sims, n_countries)``
for country-level ones, countries in design order.

Rules (README §2, CLAUDE.md)
----------------------------
- Screened patients: ``N_screened = N_randomised / (1 - SF)``; screening visits
  apply to screened patients, every other visit to randomised patients.
- Dropout: constant hazard ``h = -ln(1 - p_annual) / 12`` per month, so the
  expected fraction of randomised patients still attending a visit at day
  ``d`` is ``exp(-h * d / 30.44)``. Expected (fractional) counts throughout.
- Every **on-site** visit adds ``coordinator_visit``, ``logistics_visit``,
  ``vital_signs`` and, if ``operations.patient_travel_support``,
  ``patient_travel`` (once, even if the visit already lists them). Remote
  visits only cost the procedures they list.
- Time: ``T_LPLV = T_LPI + patient duration``, ``T_total = T_LPLV + close-out``.
- Site ``i`` is active from its activation to LPLV: site-months
  ``= sum_i max(0, T_LPLV - a_i)``; monitoring visits = site-months / interval.
- Conduct window = T_LPLV - first site activation (D27); DMC meetings =
  ceil(conduct window / DMC interval); central monitoring is per conduct month.
- Interim analyses = ``operations.n_interim_analyses``.
- Scans = occurrences of procedures whose category is ``imaging``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trialcost.catalogue import AUTO_ONSITE_PROCEDURES, TRAVEL_PROCEDURE, Catalogue
from trialcost.design import DAYS_PER_MONTH, Design
from trialcost.distributions import draw
from trialcost.recruitment import Recruitment

#: Country label for study-level cost lines.
GLOBAL = "GLOBAL"


def dropout_hazard(annual_rate: np.ndarray) -> np.ndarray:
    """Monthly hazard from an annual dropout probability: ``-ln(1 - p) / 12``."""
    return -np.log1p(-annual_rate) / 12


def visit_attendance(design: Design, hazard: np.ndarray) -> np.ndarray:
    """Expected fraction attending each visit, shape ``(n_sims, n_visits)``.

    Screening visits get 1 (they are applied to screened patients, before
    randomisation); other visits ``exp(-h * day / 30.44)``.
    """
    days = np.array([0 if v.screening else v.day for v in design.visits], dtype=float)
    return np.exp(-hazard[:, None] * days[None, :] / DAYS_PER_MONTH)


def procedure_incidence(design: Design, procedure_ids: list[str]) -> np.ndarray:
    """Occurrences of each procedure at each visit, shape ``(n_visits, n_procedures)``.

    Listed procedures count as many times as they are listed; automatic
    on-site procedures are added once to every non-remote visit.
    """
    col = {p: j for j, p in enumerate(procedure_ids)}
    auto = list(AUTO_ONSITE_PROCEDURES)
    if design.operations.patient_travel_support:
        auto.append(TRAVEL_PROCEDURE)
    a = np.zeros((len(design.visits), len(procedure_ids)))
    for i, v in enumerate(design.visits):
        for p in v.procedures:
            a[i, col[p]] += 1
        if not v.remote:
            for p in auto:
                a[i, col[p]] = max(a[i, col[p]], 1)
    return a


@dataclass(frozen=True)
class Quantities:
    """Everything the costing step multiplies prices by."""

    countries: tuple[str, ...]
    n_sites: np.ndarray  # (n_c,)
    screen_failure: np.ndarray  # (n_sims,)
    annual_dropout: np.ndarray  # (n_sims,)
    t_lpi: np.ndarray  # (n_sims,) months from study start
    t_lplv: np.ndarray  # (n_sims,)
    t_total: np.ndarray  # (n_sims,) study start to CSR
    randomised: np.ndarray  # (n_sims, n_c)
    screened: np.ndarray  # (n_sims, n_c)
    visit_counts: np.ndarray  # (n_sims, n_c, n_visits) expected patient-visits
    procedure_ids: tuple[str, ...]
    procedure_counts: np.ndarray  # (n_sims, n_c, n_procedures) expected occurrences
    scans: np.ndarray  # (n_sims, n_c)
    site_months: np.ndarray  # (n_sims, n_c) active site-months up to LPLV
    monitoring_visits: np.ndarray  # (n_sims, n_c)
    conduct_months: np.ndarray  # (n_sims,) first activation -> LPLV
    dmc_meetings: np.ndarray  # (n_sims,)
    n_interim_analyses: int
    raters: np.ndarray  # (n_c,)

    @property
    def n_sims(self) -> int:
        return self.t_total.shape[0]

    def driver(self, name: str) -> np.ndarray:
        """Quantity for a study-item driver: ``(n_sims,)`` or ``(n_sims, n_c)``."""
        n_sims, n_c = self.n_sims, len(self.countries)
        per_country = np.ones((n_sims, n_c))
        return {
            "per_study": np.ones(n_sims),
            "per_month": self.t_total,
            "per_meeting": self.dmc_meetings,
            "per_conduct_month": self.conduct_months,
            "per_interim_analysis": np.full(n_sims, float(self.n_interim_analyses)),
            "per_country": per_country,
            "per_site": per_country * self.n_sites,
            "per_site_month": self.site_months,
            "per_site_year": self.site_months / 12,
            "per_patient": self.randomised,
            "per_patient_visit": self.visit_counts.sum(axis=2),
            "per_scan": self.scans,
            "per_monitoring_visit": self.monitoring_visits,
            "per_rater": per_country * self.raters,
        }[name]


def _site_to_country(design: Design) -> np.ndarray:
    """One-hot matrix ``(n_sites, n_c)``: ``site_values @ m`` sums sites by country."""
    codes = [c.country for c in design.countries]
    sites = [codes.index(c.country) for c in design.countries for _ in range(c.n_sites)]
    m = np.zeros((len(sites), len(codes)))
    m[np.arange(len(sites)), sites] = 1.0
    return m


def compute_quantities(
    design: Design,
    catalogue: Catalogue,
    recruitment: Recruitment,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> Quantities:
    """Derive all quantities for the simulated trials in ``recruitment``."""
    n_sims = recruitment.n_sims
    pop, ops = design.population, design.operations

    def d(dist):
        return draw(dist, rng, n_sims, deterministic)

    sf = d(pop.screen_failure_rate)
    p_drop = d(pop.annual_dropout_rate)
    monitoring_interval = d(ops.monitoring_interval_months)
    closeout = d(catalogue.constant("closeout_months"))
    dmc_interval = d(catalogue.constant("dmc_meeting_interval_months"))
    raters_per_site = catalogue.constant("raters_per_site").central()  # count, not sampled

    # time
    t_lpi = recruitment.t_lpi
    t_lplv = t_lpi + design.patient_duration_months
    t_total = t_lplv + closeout

    # patients and visits
    to_country = _site_to_country(design)
    randomised = recruitment.patients @ to_country  # (n_sims, n_c)
    screened = randomised / (1 - sf)[:, None]
    attendance = visit_attendance(design, dropout_hazard(p_drop))  # (n_sims, n_v)
    is_screening = np.array([v.screening for v in design.visits])
    base = np.where(is_screening[None, None, :], screened[:, :, None], randomised[:, :, None])
    visit_counts = base * attendance[:, None, :]  # (n_sims, n_c, n_v)

    # procedures
    procedure_ids = list(catalogue.procedures.index)
    incidence = procedure_incidence(design, procedure_ids)  # (n_v, n_p)
    procedure_counts = visit_counts @ incidence  # (n_sims, n_c, n_p)
    is_imaging = (catalogue.procedures["category"] == "imaging").to_numpy()
    scans = procedure_counts[:, :, is_imaging].sum(axis=2)

    # sites
    active = np.clip(t_lplv[:, None] - recruitment.activation, 0.0, None)  # (n_sims, n_sites)
    site_months = active @ to_country
    n_sites = np.array([c.n_sites for c in design.countries], dtype=float)

    conduct = t_lplv - recruitment.activation.min(axis=1)
    return Quantities(
        countries=tuple(c.country for c in design.countries),
        n_sites=n_sites,
        screen_failure=sf,
        annual_dropout=p_drop,
        t_lpi=t_lpi,
        t_lplv=t_lplv,
        t_total=t_total,
        randomised=randomised,
        screened=screened,
        visit_counts=visit_counts,
        procedure_ids=tuple(procedure_ids),
        procedure_counts=procedure_counts,
        scans=scans,
        site_months=site_months,
        monitoring_visits=site_months / monitoring_interval[:, None],
        conduct_months=conduct,
        dmc_meetings=np.ceil(conduct / dmc_interval),
        n_interim_analyses=ops.n_interim_analyses,
        raters=n_sites * raters_per_site,
    )
