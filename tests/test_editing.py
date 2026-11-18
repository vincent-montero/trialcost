import pytest
import yaml
from pydantic import ValidationError

from trialcost import editing
from trialcost.design import Design
from trialcost.distributions import Fixed, LogNormal


def test_countries_roundtrip_keeps_activation(design):
    design.countries[0].activation_months = Fixed(value=2.0)
    table = editing.countries_table(design)
    table.loc[0, "n_sites"] = 6
    table.loc[len(table)] = ["NL", 2]
    new = editing.apply_countries(design, table)
    assert new.n_sites == design.n_sites + 2 + 2
    assert new.countries[0].activation_months == Fixed(value=2.0)
    assert new.countries[-1].country == "NL" and new.countries[-1].activation_months is None
    assert design.n_sites == 15  # input not mutated


def test_countries_zero_sites_dropped(design):
    table = editing.countries_table(design)
    table.loc[table["country"] == "IT", "n_sites"] = 0
    assert "IT" not in [c.country for c in editing.apply_countries(design, table).countries]


def test_visits_grid_roundtrip(design, cat):
    ids = list(cat.procedures.index)
    grid = editing.visits_grid(design, ids)
    assert list(grid.columns[:4]) == editing.VISIT_COLUMNS
    assert grid.loc[grid["id"] == "W24", "mri_brain"].item()
    assert editing.apply_visits(design, grid).model_dump() == design.model_dump()


def test_visits_grid_edit_and_invalid(design, cat):
    grid = editing.visits_grid(design, list(cat.procedures.index))
    grid.loc[grid["id"] == "W4", "ecg"] = True
    new = editing.apply_visits(design, grid)
    assert "ecg" in next(v for v in new.visits if v.id == "W4").procedures
    grid.loc[grid["id"] == "W4", "day"] = -5  # negative day without screening flag
    with pytest.raises(ValidationError):
        editing.apply_visits(design, grid)


def test_regular_visits(design):
    new = editing.regular_visits(design, every_weeks=8, until_week=48, template_visit_id="W24")
    post = [v for v in new.visits if v.day > 0]
    assert [v.id for v in post] == ["W8", "W16", "W24", "W32", "W40", "W48"]
    assert all("mri_brain" in v.procedures for v in post)
    assert [v.id for v in new.visits if v.day <= 0] == ["SCR", "BL"]


def test_prior_presets():
    presets = editing.prior_presets()
    ph2 = presets["rare_neuro, phase=PHASE2"]
    assert isinstance(ph2["rate_mean"], LogNormal) and ph2["n_trials"] > 0


def test_apply_population_and_fields(design):
    new = editing.apply_population(design, rate_mean=Fixed(value=0.5), rate_shape=3)
    assert new.population.rate_mean == Fixed(value=0.5) and new.population.rate_shape == 3
    new = editing.apply_fields(design, n_randomised=80, operations={"dmc": False})
    assert new.n_randomised == 80 and new.operations.dmc is False
    assert new.operations.central_imaging_reading is True  # untouched
    with pytest.raises(ValidationError):
        editing.apply_fields(design, n_randomised=0)


def test_to_yaml_roundtrip(design):
    text = editing.to_yaml(design)
    assert Design.model_validate(yaml.safe_load(text)) == design
    assert "monitoring_interval_months: 2.0" in text  # Fixed written as a plain number


def test_diff_lists_changes(design):
    other = editing.apply_fields(design, n_randomised=80, operations={"dmc": False})
    other = editing.apply_population(other, rate_mean=Fixed(value=0.5))
    changes = editing.diff(design, other)
    assert "n_randomised: 50 → 80" in changes
    assert "operations.dmc: True → False" in changes
    assert any(c.startswith("population.rate_mean") for c in changes)
    assert editing.diff(design, design) == []
