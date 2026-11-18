"""The benchmark report must run and convert units correctly.

Agreement with benchmarks is deliberately NOT asserted (CLAUDE.md): it is read
in reports/benchmark-validation.md.
"""

import pandas as pd
import pytest

from trialcost import validation


def test_inputs_load_and_reference_designs_exist(cat):
    inputs = validation.load_inputs(cat)
    assert len(inputs.checks) == len(pd.read_csv(validation.DATA_DIR / "benchmark_checks.csv"))
    assert inputs.checks["value"].notna().all()


def test_to_eur_2026_by_hand():
    cpi = pd.Series({2018: 484.707, 2026: 592.116})
    eur = validation.to_eur_2026(48_767, "USD/patient", 2018, cpi, fx=1.15)
    assert eur == pytest.approx(48_767 * 592.116 / 484.707 / 1.15)
    assert validation.to_eur_2026(40.1, "months", float("nan"), cpi, fx=1.15) == 40.1


def test_metrics_cover_every_check(cat):
    checks = validation.load_inputs(cat).checks
    assert set(checks["metric"]) <= set(validation.METRICS)


def test_report_runs(cat):
    df = validation.run_validation(cat, n_sims=200, seed=1)
    assert len(df) == len(validation.load_inputs(cat).checks)
    assert df[["benchmark_2026", "model_P50", "ratio_model_to_benchmark"]].notna().all().all()
    assert (df["model_P10"] <= df["model_P90"]).all()
    text = validation.format_markdown(df, n_sims=200, seed=1, fx=1.15)
    assert text.count("\n| ") == len(df) + 1  # header + one row per check


def test_bad_check_rejected(cat, tmp_path):
    import shutil

    for f in validation.DATA_DIR.glob("*.csv"):
        shutil.copy(f, tmp_path / f.name)
    checks = pd.read_csv(tmp_path / "benchmark_checks.csv")
    checks.loc[0, "metric"] = "happiness"
    checks.to_csv(tmp_path / "benchmark_checks.csv", index=False)
    with pytest.raises(ValueError, match="unknown metric"):
        validation.load_inputs(cat, tmp_path)
