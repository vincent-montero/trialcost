import math
import time

import numpy as np
import pytest
from conftest import run_pipeline

from trialcost.design import Visit
from trialcost.distributions import Fixed

#: Deterministic archetype total, EUR. Regression value: update deliberately when
#: data or methodology changes, and log the change in the README work log.
GOLDEN_TOTAL = 4_182_179.92


def line(costs, item, country):
    m = (costs.meta["item"] == item) & (costs.meta["country"] == country)
    assert m.sum() == 1, f"expected one line for {item} / {country}"
    return costs.values[m.to_numpy()][0]


# --- deterministic golden test --------------------------------------------- #


def test_golden_total(design, cat):
    _, costs = run_pipeline(design, cat)
    assert costs.total()[0] == pytest.approx(GOLDEN_TOTAL, abs=0.01)


def test_hand_checked_lines(design, cat):
    q, costs = run_pipeline(design, cat)
    k = q.countries.index
    # time-driven: 12,000 EUR/month x T_total
    assert line(costs, "project_management", "GLOBAL")[0] == pytest.approx(12_000 * q.t_total[0])
    # DMC: 5 meetings x 25,000
    assert line(costs, "dmc_meeting", "GLOBAL")[0] == 125_000
    # US MRI override: 1,000 EUR per scan, no index, no overhead (D25)
    assert line(costs, "mri_brain", "US")[0] == pytest.approx(1_000 * q.scans[0, k("US")])
    # FR MRI: FR price (index 1, overhead 0)
    assert line(costs, "mri_brain", "FR")[0] == pytest.approx(300 * q.scans[0, k("FR")])
    # DE consent: 87 x index 1.3 x (1 + 0.2), on screened patients
    assert line(costs, "consent", "DE")[0] == pytest.approx(87 * 1.3 * 1.2 * q.screened[0, k("DE")])
    # central lab panel: vendor price, not scaled
    de_panels = q.procedure_counts[0, k("DE"), q.procedure_ids.index("labs_central_panel")]
    assert line(costs, "labs_central_panel", "DE")[0] == pytest.approx(200 * de_panels)
    # investigator fee: country-scaled study item (D23), US index 3.0, overhead 0.3
    assert line(costs, "investigator_fee", "US")[0] == pytest.approx(
        1_500 * 3.0 * 1.3 * q.randomised[0, k("US")]
    )
    # monitoring visits: site-months / 2 x 2,200 (CRO price, not scaled)
    assert line(costs, "monitoring_visit", "IT")[0] == pytest.approx(
        2_200 * q.site_months[0, k("IT")] / 2
    )
    # CTA fees from countries.csv
    assert line(costs, "cta_fee", "FR")[0] == 0
    assert line(costs, "cta_fee", "DE")[0] == 10_000


def test_country_conditions(design, cat):
    _, costs = run_pipeline(design, cat)
    by_item = costs.meta.groupby("item")["country"].apply(set)
    assert by_item["site_fixed_fees_fr"] == {"FR"}
    assert by_item["irb_local"] == {"US"}
    assert by_item["protocol_writing"] == {"GLOBAL"}


# --- structure ------------------------------------------------------------- #


def test_breakdowns_add_up(design, cat):
    _, costs = run_pipeline(design, cat, n_sims=50, deterministic=False)
    total = costs.total()
    for cols in [("category",), ("resource",), ("country",), ("country", "category")]:
        np.testing.assert_allclose(costs.by(*cols).sum(axis=0).to_numpy(), total)
    assert np.all(costs.values >= 0)


def test_to_frame_is_tidy(design, cat):
    _, costs = run_pipeline(design, cat, n_sims=3, deterministic=False)
    frame = costs.to_frame()
    assert len(frame) == 3 * len(costs.meta)
    assert set(frame.columns) >= {"sim", "item", "country", "category", "resource", "cost"}
    np.testing.assert_allclose(frame.groupby("sim")["cost"].sum().to_numpy(), costs.total())


# --- invariants (deterministic: exact comparisons) -------------------------- #


def _total(design, cat):
    return run_pipeline(design, cat)[1].total()[0]


@pytest.mark.parametrize(
    "change",
    ["more_patients", "more_sites", "extra_visit", "more_frequent_monitoring", "no_dropout"],
)
def test_cost_increases(design, cat, change):
    base = _total(design, cat)
    if change == "more_patients":
        design.n_randomised = 80
    elif change == "more_sites":
        design.countries[0].n_sites += 2
    elif change == "extra_visit":
        design.visits.append(Visit(id="W44", day=308, procedures=["physical_exam"]))
    elif change == "more_frequent_monitoring":
        design.operations.monitoring_interval_months = Fixed(value=1.0)
    else:
        design.population.annual_dropout_rate = Fixed(value=0.0)
    assert _total(design, cat) > base


def test_removing_central_imaging_removes_imaging_lines(design, cat):
    base = _total(design, cat)
    design.operations.central_imaging_reading = False
    _, costs = run_pipeline(design, cat)
    assert not costs.meta["item"].str.startswith("imaging_").any()
    assert costs.total()[0] < base


def test_toggle_does_not_shift_other_draws(design, cat):
    """Common random numbers: turning the DMC off changes only DMC lines."""
    _, on = run_pipeline(design, cat, n_sims=20, seed=3, deterministic=False)
    design.operations.dmc = False
    _, off = run_pipeline(design, cat, n_sims=20, seed=3, deterministic=False)
    dmc = on.meta["item"].str.startswith("dmc_").to_numpy()
    np.testing.assert_allclose(on.values[~dmc], off.values)


def test_fast_enough(design, cat):
    start = time.perf_counter()
    run_pipeline(design, cat, n_sims=10_000, deterministic=False)
    assert time.perf_counter() - start < 1.0  # target is < 5 s


def test_stochastic_archetype_regression_band(design, cat):
    """P50 regression band (CLAUDE.md). Update deliberately if data changes."""
    _, costs = run_pipeline(design, cat, n_sims=10_000, seed=42, deterministic=False)
    p50 = np.median(costs.total())
    assert 4.6e6 < p50 < 5.0e6
    assert math.isfinite(costs.total().max())
