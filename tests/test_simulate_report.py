import math

import numpy as np
import pandas as pd
import pytest
from conftest import ARCHETYPE
from test_costing import GOLDEN_TOTAL

from trialcost import report
from trialcost.catalogue import CatalogueError
from trialcost.distributions import Fixed
from trialcost.quantities import dropout_hazard
from trialcost.simulate import main, run


def test_run_deterministic_matches_golden(design, cat):
    assert run(design, cat, 1, seed=0, deterministic=True).total[0] == pytest.approx(
        GOLDEN_TOTAL, abs=0.01
    )


def test_run_checks_design_against_catalogue(design, cat):
    design.visits[0].procedures.append("teleportation")
    with pytest.raises(CatalogueError):
        run(design, cat, 10, seed=0)


def test_pinning_one_price_changes_only_its_lines(design, cat):
    """Common random numbers through run(): independent child stream per input."""
    import dataclasses

    base = run(design, cat, 50, seed=1)
    procs = cat.procedures.astype({"low": float, "mode": float, "high": float})
    procs.loc["ecg", ["low", "mode", "high"]] = 30.0
    other = run(design, dataclasses.replace(cat, procedures=procs), 50, seed=1)
    ecg = (base.costs.meta["item"] == "ecg").to_numpy()
    np.testing.assert_allclose(base.costs.values[~ecg], other.costs.values[~ecg])
    assert not np.allclose(base.costs.values[ecg], other.costs.values[ecg])


# --- report ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def result(cat):
    from trialcost.design import load_design

    return run(load_design(ARCHETYPE), cat, 2_000, seed=42)


def test_summary_and_probabilities(result):
    s = report.summary(result)
    assert list(s.columns) == ["P10", "P50", "P90", "mean"]
    assert (
        s.loc["total_cost_eur", "P10"]
        < s.loc["total_cost_eur", "P50"]
        < s.loc["total_cost_eur", "P90"]
    )
    p = report.probabilities(result, budget=5e6, recruit_by=24)
    assert all(0 <= v <= 1 for v in p.values())
    assert report.probabilities(result, budget=1e9)["P(total cost <= 1,000,000,000 EUR)"] == 1


@pytest.mark.parametrize("by", ["category", "resource", "country", "item"])
def test_breakdown_shares_sum_to_one(result, by):
    b = report.breakdown(result, by)
    assert b["share"].sum() == pytest.approx(1)
    assert b["mean"].is_monotonic_decreasing


def test_dropout_report_deterministic(hand, cat):
    dp = report.dropout_report(hand, cat, 1, seed=0, deterministic=True)
    h = dropout_hazard(np.array([0.15]))[0]
    assert dp.loc["completers", "P50"] == pytest.approx(
        50 * math.exp(-h * hand.patient_duration_months)
    )
    assert dp.loc["dropout_delta_cost_eur", "P50"] < 0  # dropouts stop generating visit costs
    assert dp.loc["completer_cost_premium_pct", "P50"] > 0  # but each completer costs more
    total0 = dp.loc["total_cost_no_dropout_eur", "P50"]
    assert dp.loc["cost_per_completer_no_dropout_eur", "P50"] == pytest.approx(total0 / 50)


def test_dropout_delta_is_exactly_zero_without_dropout(design, cat):
    design.population.annual_dropout_rate = Fixed(value=0.0)
    dp = report.dropout_report(design, cat, 200, seed=5)
    assert dp.loc["dropout_delta_cost_eur", "P10"] == 0
    assert dp.loc["dropout_delta_cost_eur", "P90"] == 0


def test_tornado(design, cat):
    t = report.tornado(design, cat, n_sims=300, seed=3)
    assert len(t) == len(report.TORNADO_SWINGS)
    assert t["swing"].is_monotonic_decreasing
    row = t.set_index("parameter")
    slow, fast = row.loc["Recruitment rate (rate_mean)", ["cost_at_low", "cost_at_high"]]
    assert slow > fast  # slower recruitment costs more
    lo, hi = row.loc["CRO labour prices", ["cost_at_low", "cost_at_high"]]
    assert lo < row["base"].iloc[0] < hi


def test_tornado_does_not_mutate_inputs(design, cat):
    before_design = design.model_dump()
    before_items = cat.study_items.copy()
    report.tornado(design, cat, n_sims=50, seed=0)
    assert design.model_dump() == before_design
    pd.testing.assert_frame_equal(cat.study_items, before_items)


# --- CLI --------------------------------------------------------------------- #


def test_cli_end_to_end(tmp_path, capsys):
    code = main(
        [str(ARCHETYPE), "--n-sims", "300", "--seed", "1", "--tornado-sims", "100",
         "--budget", "5e6", "--recruit-by", "24", "--out", str(tmp_path), "--lines"]
    )  # fmt: skip
    assert code == 0
    text = capsys.readouterr().out
    for heading in ("Headline", "By category", "Dropout cost", "Tornado", "P(total cost"):
        assert heading in text
    names = {p.name for p in tmp_path.iterdir()}
    assert {"summary.csv", "breakdown_category.csv", "dropout.csv", "tornado.csv",
            "probabilities.csv", "cost_lines.csv.gz"} <= names  # fmt: skip
    lines = pd.read_csv(tmp_path / "cost_lines.csv.gz")
    assert lines["sim"].nunique() == 300


def test_cli_deterministic_skips_tornado(capsys):
    assert main([str(ARCHETYPE), "--deterministic", "--n-sims", "1"]) == 0
    text = capsys.readouterr().out
    assert "Tornado" not in text and f"{GOLDEN_TOTAL / 1e6:,.2f} M€" in text


def test_compare_and_category_deltas(design, cat):
    base = run(design, cat, 500, seed=7)
    other_design = design.model_copy(deep=True)
    other_design.operations.dmc = False
    other = run(other_design, cat, 500, seed=7)
    table = report.compare({"base": base, "no DMC": other})
    assert table.loc["base", "delta_cost_P50"] == 0
    assert table.loc["no DMC", "delta_cost_P50"] < 0
    deltas = report.category_deltas(base, other)
    assert deltas.index[0] == "safety"  # DMC lines are in the safety category
    nonzero = deltas[deltas.abs() > 1e-6]
    assert set(nonzero.index) == {"safety"}  # common random numbers: nothing else moves
