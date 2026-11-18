import shutil

import pandas as pd
import pytest

from trialcost import DATA_DIR, suggestions
from trialcost.catalogue import load_catalogue


def _df(*rows):
    return suggestions.merge(suggestions.empty(), [suggestions.make(*r) for r in rows])


def test_make_merge_dedupes_latest():
    df = _df(
        ("study_items", "monitoring_visit", "mode", 2200, 3000, "CRA said 3k"),
        ("study_items", "monitoring_visit", "mode", 2200, 2800, "on reflection 2.8k"),
    )
    assert len(df) == 1 and df.loc[0, "suggested_value"] == 2800
    assert df.loc[0, "suggestion_id"] == 1


def test_apply_to_catalogue_is_a_copy(cat):
    df = _df(("study_items", "monitoring_visit", "mode", 2200, 3000, "r"))
    new = suggestions.apply_to_catalogue(cat, df)
    assert new.item_price("monitoring_visit").mode == 3000
    assert cat.item_price("monitoring_visit").mode == 2200  # base untouched


def test_composite_and_country_rows(cat):
    df = _df(
        ("procedure_overrides", "mri_brain|US", "mode", 1000, 1100, "r"),
        ("countries", "DE", "site_price_mode", 1.3, 1.4, "r"),
    )
    new = suggestions.apply_to_catalogue(cat, df)
    assert new.procedure_override("mri_brain", "US").price.mode == 1100
    assert new.country_range("DE", "site_price").mode == 1.4


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("study_items", "nope", "mode", 1, 2, "r"), "unknown row"),
        (("study_items", "monitoring_visit", "label", 1, 2, "r"), "not editable"),
        (("study_items", "monitoring_visit", "mode", 2200, 2300, "  "), "rationale"),
        (("study_items", "monitoring_visit", "mode", 2200, 9000, "r"), "low <= mode <= high"),
        (("planets", "x", "mode", 1, 2, "r"), "unknown table"),
    ],
)
def test_validate(cat, row, message):
    problems = suggestions.validate(_df(row), cat)
    assert any(message in p for p in problems)


def test_moving_whole_range_is_valid(cat):
    df = _df(
        ("study_items", "monitoring_visit", "high", 3500, 9000, "r"),
        ("study_items", "monitoring_visit", "mode", 2200, 5000, "r"),
    )
    assert suggestions.validate(df, cat) == []


def test_csv_roundtrip(tmp_path):
    df = _df(("procedure_overrides", "mri_brain|US", "mode", 1000, 1100, "r"))
    df.to_csv(tmp_path / "s.csv", index=False)
    back = suggestions.read(tmp_path / "s.csv")
    assert back.loc[0, "row_id"] == "mri_brain|US"


@pytest.fixture
def data_copy(tmp_path):
    for f in DATA_DIR.glob("*.csv"):
        shutil.copy(f, tmp_path / f.name)
    return tmp_path


def test_apply_to_data_edits_only_accepted_cells(data_copy):
    df = _df(
        ("study_items", "monitoring_visit", "mode", 2200, 2500, "CRA review"),
        ("procedures", "ecg", "mode", 30, 40, "ignored"),
    )
    before_proc = (data_copy / "procedures.csv").read_text()
    before = (data_copy / "study_items.csv").read_text().splitlines()
    log = suggestions.apply_to_data(
        df, accept=[1], grade="C", data_dir=data_copy, today="2026-12-01"
    )
    assert log == ["study_items.csv: monitoring_visit.mode 2200 -> 2500"]
    after = (data_copy / "study_items.csv").read_text().splitlines()
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == 1  # one row, everything else byte-identical
    row = pd.read_csv(data_copy / "study_items.csv").set_index("item_id").loc["monitoring_visit"]
    assert row["mode"] == 2500 and row["evidence"] == "C"
    assert "Expert review 2026-12-01" in row["source"] and "CRA review" in row["notes"]
    assert (data_copy / "procedures.csv").read_text() == before_proc  # not accepted


def test_apply_to_data_restores_on_failure(data_copy):
    df = _df(("study_items", "monitoring_visit", "mode", 2200, 9000, "breaks the range"))
    before = (data_copy / "study_items.csv").read_text()
    with pytest.raises(ValueError):
        suggestions.apply_to_data(df, accept=[1], data_dir=data_copy)
    assert (data_copy / "study_items.csv").read_text() == before
    load_catalogue(data_copy)


def test_impact(cat, tmp_path):
    from conftest import ARCHETYPE

    df = _df(("study_items", "project_management", "mode", 12000, 18000, "r"))
    imp = suggestions.impact(df, cat, ARCHETYPE, n_sims=200, seed=1)
    assert imp.loc[0, "delta_mean_alone_eur"] > 0
    assert imp.attrs["together"] > imp.attrs["base"]


def test_from_edits():
    before = pd.DataFrame(
        {
            "row_id": ["a", "b"],
            "low": [1.0, 1.0],
            "mode": [2.0, 2.0],
            "high": [3.0, 3.0],
            "rationale": ["", ""],
        }
    )
    after = before.copy()
    after.loc[0, ["mode", "rationale"]] = [2.5, "seen higher"]
    after.loc[1, "high"] = 4  # no rationale
    new, missing = suggestions.from_edits(before, after, "study_items", "me")
    assert [(s["row_id"], s["field"], s["suggested_value"]) for s in new] == [("a", "mode", 2.5)]
    assert missing == ["b"]
