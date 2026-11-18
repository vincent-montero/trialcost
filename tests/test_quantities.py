import math

import numpy as np
import pytest
from conftest import run_pipeline

from trialcost.design import DAYS_PER_MONTH
from trialcost.distributions import Fixed
from trialcost.quantities import dropout_hazard, procedure_incidence, visit_attendance

# Deterministic archetype (see test_recruitment): T_LPI = 8 + 41 / 3.75
T_LPI = 8 + 41 / 3.75
T_LPLV = T_LPI + 392 / DAYS_PER_MONTH
H = -math.log(0.85) / 12


def test_dropout_hazard_gives_annual_rate_back():
    h = dropout_hazard(np.array([0.15]))
    assert h[0] == pytest.approx(H)
    assert math.exp(-12 * h[0]) == pytest.approx(0.85)


def test_visit_attendance(hand):
    att = visit_attendance(hand, np.array([H, 0.0]))
    days = [v.day for v in hand.visits]
    assert att[0, 0] == 1  # screening
    assert att[0, days.index(0)] == 1  # baseline
    assert att[0, days.index(364)] == pytest.approx(math.exp(-H * 364 / DAYS_PER_MONTH))
    assert np.all(att[1] == 1)  # no dropout


def test_procedure_incidence_auto_procedures(hand, cat):
    ids = list(cat.procedures.index)
    col = ids.index
    a = procedure_incidence(hand, ids)
    remote = np.array([v.remote for v in hand.visits])
    assert np.all(a[~remote, col("coordinator_visit")] == 1)
    assert np.all(a[remote, col("coordinator_visit")] == 0)
    assert np.all(a[~remote, col("patient_travel")] == 1)  # travel support on
    assert a[:, col("mri_brain")].sum() == 3  # SCR, W24, W52
    assert a[remote, col("ae_review")].sum() == 2  # listed procedures still count
    hand.operations.patient_travel_support = False
    assert procedure_incidence(hand, ids)[:, col("patient_travel")].sum() == 0


def test_auto_procedure_listed_explicitly_is_not_doubled(design, cat):
    ids = list(cat.procedures.index)
    design.visits[1].procedures.append("vital_signs")
    assert procedure_incidence(design, ids)[1, ids.index("vital_signs")] == 1


def test_deterministic_time_and_patients(hand, cat):
    q, _ = run_pipeline(hand, cat)
    assert q.t_lpi[0] == pytest.approx(T_LPI)
    assert q.t_lplv[0] == pytest.approx(T_LPLV)
    assert q.t_total[0] == pytest.approx(T_LPLV + 6)
    assert q.randomised.sum() == pytest.approx(50)
    np.testing.assert_allclose(q.screened, q.randomised / 0.65)
    # US: 3 sites open at month 5; exposure share 13.93/200 each for the first 49
    # patients, and 1/15 each of the 50th (all 15 sites open at T_LPI)
    us = q.countries.index("US")
    assert q.randomised[0, us] == pytest.approx(3 * (49 * (T_LPI - 5) / 200 + 1 / 15))


def test_deterministic_site_quantities(hand, cat):
    q, _ = run_pipeline(hand, cat)
    fr, it = q.countries.index("FR"), q.countries.index("IT")
    assert q.site_months[0, fr] == pytest.approx(4 * (T_LPLV - 5))
    assert q.site_months[0, it] == pytest.approx(2 * (T_LPLV - 8))
    np.testing.assert_allclose(q.monitoring_visits, q.site_months / 2)
    assert q.dmc_meetings[0] == math.ceil((T_LPLV - 5) / 6)  # D27
    np.testing.assert_array_equal(q.raters, 2 * q.n_sites)


def test_scans_count_mri_only(hand, cat):
    q, _ = run_pipeline(hand, cat)
    us = q.countries.index("US")
    att = [math.exp(-H * d / DAYS_PER_MONTH) for d in (168, 364)]
    expected = q.screened[0, us] + q.randomised[0, us] * sum(att)
    assert q.scans[0, us] == pytest.approx(expected)


def test_zero_dropout_means_more_visits(hand, cat):
    q_drop, _ = run_pipeline(hand, cat)
    hand.population.annual_dropout_rate = Fixed(value=0.0)
    q_none, _ = run_pipeline(hand, cat)
    assert np.all(q_none.visit_counts >= q_drop.visit_counts - 1e-12)
    assert q_none.visit_counts.sum() > q_drop.visit_counts.sum()


def test_driver_shapes(design, cat):
    q, _ = run_pipeline(design, cat, n_sims=7, deterministic=False)
    for name in ["per_study", "per_month", "per_meeting"]:
        assert q.driver(name).shape == (7,)
    for name in ["per_country", "per_site", "per_site_month", "per_site_year", "per_patient",
                 "per_patient_visit", "per_scan", "per_monitoring_visit", "per_rater"]:  # fmt: skip
        assert q.driver(name).shape == (7, 5)
