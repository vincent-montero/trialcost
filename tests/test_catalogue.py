import shutil
from pathlib import Path

import pandas as pd
import pytest

from trialcost import DATA_DIR
from trialcost.catalogue import (
    AUTO_ONSITE_PROCEDURES,
    EVIDENCE_GRADES,
    TRAVEL_PROCEDURE,
    Always,
    CatalogueError,
    CountryIs,
    OperationsFlag,
    UsesProcedure,
    load_catalogue,
    parse_condition,
)
from trialcost.design import load_design
from trialcost.distributions import Fixed, Triangular

DESIGNS = sorted((Path(__file__).resolve().parents[1] / "designs").glob("*.yaml"))
ARCHETYPE = next(p for p in DESIGNS if p.stem == "rare-neuro-ph2-mri-scales")


@pytest.fixture(scope="module")
def cat():
    return load_catalogue()


@pytest.fixture
def design():
    return load_design(ARCHETYPE)


# --- the real catalogue ----------------------------------------------------- #


def test_real_catalogue_loads(cat):
    assert len(cat.procedures) == 20
    assert len(cat.study_items) == 39
    assert set(cat.countries.index) == {"FR", "DE", "ES", "IT", "NL", "US"}


@pytest.mark.parametrize("table", ["procedures", "study_items", "countries", "constants"])
def test_grades_and_sources(cat, table):
    df = getattr(cat, table)
    for grade in df["evidence"]:
        assert set(grade.split("/")) <= EVIDENCE_GRADES
    assert df["source"].str.strip().ne("").all()


@pytest.mark.parametrize("table", ["procedures", "study_items", "constants"])
def test_low_mode_high(cat, table):
    df = getattr(cat, table)
    assert ((df["low"] <= df["mode"]) & (df["mode"] <= df["high"])).all()


def test_site_paid_study_items_are_country_scaled(cat):
    scaled = set(cat.study_items.index[cat.study_items["country_scaling"] == "site_price"])
    assert {"pharmacy_annual", "site_monitoring_fee", "investigator_fee"} <= scaled
    assert "monitoring_visit" not in scaled  # CRO price, not a site price


def test_automatic_procedures_exist(cat):
    assert set(AUTO_ONSITE_PROCEDURES) | {TRAVEL_PROCEDURE} <= set(cat.procedures.index)


@pytest.mark.parametrize("path", DESIGNS, ids=lambda p: p.stem)
def test_every_design_matches_catalogue(cat, path):
    cat.check_design(load_design(path))  # raises on unknown procedure / country


def test_prices_are_triangular(cat):
    assert cat.procedure_price("mri_brain") == Triangular(low=170, mode=300, high=420)
    assert cat.country_range("US", "site_price") == Triangular(low=2.5, mode=3.0, high=4.0)
    assert cat.constant("closeout_months").central() == 6
    with pytest.raises(KeyError):
        cat.country_range("US", "labour_cost_eur_h")


def test_procedure_override(cat):
    us_mri = cat.procedure_override("mri_brain", "US")
    assert us_mri.price == Triangular(low=687, mode=1000, high=1261)
    assert us_mri.apply_overhead is False
    assert cat.procedure_override("mri_brain", "FR") is None
    assert cat.procedure_override("ecg", "US") is None


def test_activation_defaults_to_catalogue(cat, design):
    assert cat.activation_months(design, "IT") == Triangular(low=5, mode=8, high=12)
    design.countries[0].activation_months = Fixed(value=2.0)
    assert cat.activation_months(design, design.countries[0].country) == Fixed(value=2.0)


# --- applies_if -------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "atoms"),
    [
        ("always", (Always(),)),
        ("country==FR", (CountryIs("FR"),)),
        ("operations.irt", (OperationsFlag("irt"),)),
        ("uses:scale_pro", (UsesProcedure("scale_pro"),)),
        (
            "operations.central_imaging_reading and country==FR",
            (OperationsFlag("central_imaging_reading"), CountryIs("FR")),
        ),
    ],
)
def test_parse_condition(text, atoms):
    assert parse_condition(text).atoms == atoms


@pytest.mark.parametrize(
    "text",
    [
        "",
        "never",
        "country==fr",
        "country=FR",
        "operations.monitoring",  # not a boolean flag
        "operations.unknown_flag",
        "always or country==FR",
        "__import__('os')",
    ],
)
def test_parse_condition_rejects(text):
    with pytest.raises(ValueError):
        parse_condition(text)


def test_condition_semantics(design):
    assert parse_condition("operations.dmc").holds(design)
    assert parse_condition("uses:scale_clinician").holds(design)
    assert not parse_condition("uses:lumbar_puncture").holds(design)
    cond = parse_condition("operations.central_imaging_reading and country==FR")
    assert cond.holds(design, "FR") and not cond.holds(design, "DE")
    with pytest.raises(ValueError):
        cond.holds(design)  # country condition without a country


def test_applicable_items(cat, design):
    study = set(cat.applicable_items(design))
    assert "dmc_meeting" in study and "site_fixed_fees_fr" not in study
    assert "site_fixed_fees_fr" in cat.applicable_items(design, "FR")
    assert "irb_local" in cat.applicable_items(design, "US")
    assert "irb_local" not in cat.applicable_items(design, "FR")


def test_no_central_imaging_removes_imaging_items(cat, design):
    design.operations.central_imaging_reading = False
    for country in [None, "FR", "US"]:
        assert not [i for i in cat.applicable_items(design, country) if i.startswith("imaging_")]


# --- broken catalogues are rejected with every problem listed ---------------- #


@pytest.fixture
def data_copy(tmp_path):
    for f in DATA_DIR.glob("*.csv"):
        shutil.copy(f, tmp_path / f.name)
    return tmp_path


def _mutate(data_dir: Path, table: str, fn) -> None:
    """Rewrite one CSV of the copied catalogue with ``fn(df) -> df``."""
    path = data_dir / f"{table}.csv"
    fn(pd.read_csv(path)).to_csv(path, index=False)


def set_cell(row: int, col: str, value):
    """Mutation: set one cell (row = 0-based position in the CSV)."""

    def fn(df: pd.DataFrame) -> pd.DataFrame:
        df = df.astype({col: object})
        df.loc[row, col] = value
        return df

    return fn


def drop_row(row: int):
    return lambda df: df.drop(index=row)


def drop_column(col: str):
    return lambda df: df.drop(columns=col)


@pytest.mark.parametrize(
    ("table", "mutation", "message"),
    [
        ("procedures", set_cell(0, "low", 999), "low <= mode <= high"),
        ("procedures", set_cell(0, "evidence", "E"), "evidence grade"),
        ("procedures", set_cell(0, "source", " "), "empty source"),
        ("procedures", set_cell(0, "country_scaling", "x"), "country_scaling"),
        ("procedures", set_cell(1, "procedure_id", "consent"), "duplicate"),
        ("study_items", set_cell(0, "driver", "per_lunch"), "unknown driver"),
        ("study_items", set_cell(0, "country_scaling", "x"), "country_scaling"),
        ("study_items", set_cell(0, "country_scaling", "site_price"), "site_price scaling on"),
        ("study_items", set_cell(0, "applies_if", "maybe"), "cannot parse"),
        (
            "study_items",
            set_cell(0, "applies_if", "country==FR"),
            "country condition",
        ),
        (
            "study_items",
            set_cell(0, "applies_if", "uses:nope"),
            "unknown procedure",
        ),
        (
            "countries",
            set_cell(0, "site_price_mode", 1.2),
            "index must be fixed",
        ),
        ("countries", set_cell(1, "country", "Germany"), "bad country code"),
        ("constants", drop_row(0), "required constant"),
        ("procedure_overrides", set_cell(0, "procedure_id", "nope"), "unknown procedure"),
        ("procedure_overrides", set_cell(0, "country", "ZZ"), "unknown country"),
        ("procedure_overrides", set_cell(0, "apply_overhead", "maybe"), "apply_overhead"),
        ("procedure_overrides", set_cell(0, "low", 5000), "low <= mode <= high"),
        ("benchmarks", drop_column("source"), "columns"),
    ],
)
def test_broken_catalogue_rejected(data_copy, table, mutation, message):
    _mutate(data_copy, table, mutation)
    with pytest.raises(CatalogueError, match=message):
        load_catalogue(data_copy)


def test_all_problems_reported_together(data_copy):
    _mutate(data_copy, "procedures", set_cell(0, "low", 999))
    _mutate(data_copy, "study_items", set_cell(0, "driver", "per_lunch"))
    with pytest.raises(CatalogueError) as err:
        load_catalogue(data_copy)
    assert len(err.value.problems) == 2


def test_design_with_unknown_ids_rejected(cat, design):
    design.visits[0].procedures.append("teleportation")
    design.countries[0].country = "ZZ"
    with pytest.raises(CatalogueError) as err:
        cat.check_design(design)
    assert len(err.value.problems) == 2
