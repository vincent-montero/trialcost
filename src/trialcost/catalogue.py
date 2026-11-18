"""Load and validate the cost catalogue in ``data/``.

The CSVs are the only source of numbers. This module:

1. reads them into pandas DataFrames indexed by their ID column;
2. validates them (columns, unique IDs, ``low <= mode <= high``, evidence
   grades, non-empty sources, known drivers / scalings, ``applies_if``
   conditions) and reports **all** problems at once in a ``CatalogueError``;
3. exposes prices and ranges as ``Triangular`` distributions, so the engine
   samples catalogue values exactly like design inputs;
4. checks a design against the catalogue (procedure and country IDs).

``applies_if`` mini-language (no ``eval``)::

    condition := atom ( " and " atom )*
    atom      := "always"
               | "country==" CC           # CC = ISO 3166 alpha-2, e.g. FR
               | "operations." FLAG       # a boolean field of design.Operations
               | "operations." FIELD "==" VALUE   # a choice field, e.g. monitoring==full_sdv
               | "uses:" PROCEDURE_ID     # some visit of the design uses it

A ``country==`` atom only makes sense for items whose driver lives in a
country (per site, per patient...), not for per-study / per-month items.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args, get_origin

import pandas as pd

from trialcost import DATA_DIR
from trialcost.design import Design, Operations
from trialcost.distributions import AnyDist, Triangular

# --------------------------------------------------------------------------- #
# Vocabulary (identifiers, not numbers)
# --------------------------------------------------------------------------- #

EVIDENCE_GRADES = frozenset("ABCD")
COUNTRY_SCALINGS = frozenset({"site_price", "none"})
DRIVERS = frozenset(
    {
        "per_study",
        "per_country",
        "per_site",
        "per_site_month",
        "per_site_year",
        "per_month",
        "per_patient",
        "per_patient_visit",
        "per_scan",
        "per_monitoring_visit",
        "per_meeting",
        "per_rater",
        "per_conduct_month",
        "per_interim_analysis",
    }
)
#: Drivers with no country attached: a ``country==`` condition is meaningless.
STUDY_LEVEL_DRIVERS = frozenset(
    {"per_study", "per_month", "per_meeting", "per_conduct_month", "per_interim_analysis"}
)

#: Procedures added automatically to every on-site visit (CLAUDE.md).
AUTO_ONSITE_PROCEDURES = ("coordinator_visit", "logistics_visit", "vital_signs")
#: Added to every on-site visit only if operations.patient_travel_support.
TRAVEL_PROCEDURE = "patient_travel"

REFERENCE_COUNTRY = "FR"  # D16: procedure prices are French; index 1, overhead 0
REQUIRED_CONSTANTS = frozenset(
    {"fx_usd_per_eur", "closeout_months", "raters_per_site", "dmc_meeting_interval_months"}
)

#: Ranges stored in countries.csv as <name>_low / _mode / _high.
COUNTRY_RANGES = ("site_price", "site_overhead", "cta_fee_eur", "activation_months")

PROCEDURE_COLUMNS = [
    "procedure_id", "label", "category", "resource", "low", "mode", "high",
    "country_scaling", "evidence", "source", "notes",
]  # fmt: skip
STUDY_ITEM_COLUMNS = [
    "item_id", "label", "category", "resource", "driver", "low", "mode", "high",
    "country_scaling", "applies_if", "evidence", "source", "notes",
]  # fmt: skip
COUNTRY_COLUMNS = [
    "country",
    "site_price_low", "site_price_mode", "site_price_high",
    "site_overhead_low", "site_overhead_mode", "site_overhead_high",
    "labour_cost_eur_h",
    "cta_fee_eur_low", "cta_fee_eur_mode", "cta_fee_eur_high",
    "activation_months_low", "activation_months_mode", "activation_months_high",
    "evidence", "source", "notes",
]  # fmt: skip
CONSTANT_COLUMNS = [
    "constant_id", "label", "unit", "low", "mode", "high", "evidence", "source", "notes",
]  # fmt: skip
OVERRIDE_COLUMNS = [
    "procedure_id", "country", "low", "mode", "high", "apply_overhead",
    "evidence", "source", "notes",
]  # fmt: skip
SPONSOR_FACTOR_COLUMNS = [
    "sponsor_model", "resources", "low", "mode", "high", "evidence", "source", "notes",
]  # fmt: skip
BENCHMARK_COLUMNS = ["benchmark_id", "description", "value", "unit", "price_year", "source"]

_OPERATION_FLAGS = frozenset(
    name for name, f in Operations.model_fields.items() if f.annotation is bool
)
#: Choice fields of Operations and their allowed values, e.g. monitoring -> {full_sdv, ...}.
_OPERATION_CHOICES = {
    name: frozenset(get_args(f.annotation))
    for name, f in Operations.model_fields.items()
    if get_origin(f.annotation) is Literal
}


class CatalogueError(ValueError):
    """One or more problems in the catalogue (or a design checked against it)."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(
            f"{len(problems)} catalogue problem(s):\n" + "\n".join(f"- {p}" for p in problems)
        )


# --------------------------------------------------------------------------- #
# applies_if conditions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Always:
    def holds(self, design: Design, country: str | None) -> bool:
        return True


@dataclass(frozen=True)
class CountryIs:
    code: str

    def holds(self, design: Design, country: str | None) -> bool:
        if country is None:
            raise ValueError(f"condition 'country=={self.code}' needs a country")
        return country == self.code


@dataclass(frozen=True)
class OperationsFlag:
    flag: str

    def holds(self, design: Design, country: str | None) -> bool:
        return bool(getattr(design.operations, self.flag))


@dataclass(frozen=True)
class OperationsEquals:
    field: str
    value: str

    def holds(self, design: Design, country: str | None) -> bool:
        return getattr(design.operations, self.field) == self.value


@dataclass(frozen=True)
class UsesProcedure:
    procedure_id: str

    def holds(self, design: Design, country: str | None) -> bool:
        return any(self.procedure_id in v.procedures for v in design.visits)


Atom = Always | CountryIs | OperationsFlag | OperationsEquals | UsesProcedure

_ATOM_PATTERNS: list[tuple[re.Pattern[str], type]] = [
    (re.compile(r"always"), Always),
    (re.compile(r"country==([A-Z]{2})"), CountryIs),
    (re.compile(r"operations\.([a-z_]+)==([a-z_]+)"), OperationsEquals),
    (re.compile(r"operations\.([a-z_]+)"), OperationsFlag),
    (re.compile(r"uses:([a-z0-9_]+)"), UsesProcedure),
]


@dataclass(frozen=True)
class Condition:
    """A conjunction of atoms parsed from an ``applies_if`` string."""

    text: str
    atoms: tuple[Atom, ...] = field(default_factory=tuple)

    @property
    def needs_country(self) -> bool:
        return any(isinstance(a, CountryIs) for a in self.atoms)

    @property
    def procedures_used(self) -> list[str]:
        return [a.procedure_id for a in self.atoms if isinstance(a, UsesProcedure)]

    def holds(self, design: Design, country: str | None = None) -> bool:
        return all(a.holds(design, country) for a in self.atoms)


def parse_condition(text: str) -> Condition:
    """Parse an ``applies_if`` string; raise ``ValueError`` if it is not valid.

    >>> parse_condition("operations.dmc and country==FR").atoms
    (OperationsFlag(flag='dmc'), CountryIs(code='FR'))
    """
    atoms: list[Atom] = []
    for part in re.split(r"\s+and\s+", text.strip()):
        for pattern, cls in _ATOM_PATTERNS:
            if m := pattern.fullmatch(part):
                atoms.append(cls(*m.groups()))
                break
        else:
            raise ValueError(f"cannot parse condition atom {part!r} in {text!r}")
        atom = atoms[-1]
        if isinstance(atom, OperationsFlag) and atom.flag not in _OPERATION_FLAGS:
            raise ValueError(
                f"unknown operations flag {atom.flag!r}; known: {sorted(_OPERATION_FLAGS)}"
            )
        if isinstance(atom, OperationsEquals):
            allowed = _OPERATION_CHOICES.get(atom.field)
            if allowed is None:
                raise ValueError(f"operations.{atom.field} is not a choice field")
            if atom.value not in allowed:
                raise ValueError(
                    f"operations.{atom.field} cannot be {atom.value!r}; allowed: {sorted(allowed)}"
                )
    return Condition(text=text, atoms=tuple(atoms))


# --------------------------------------------------------------------------- #
# Validation helpers: each returns a list of problem strings
# --------------------------------------------------------------------------- #


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def _check_columns(df: pd.DataFrame, expected: list[str], table: str) -> list[str]:
    if list(df.columns) != expected:
        return [f"{table}: columns {list(df.columns)} != expected {expected}"]
    return []


def _check_ids(df: pd.DataFrame, id_col: str, table: str) -> list[str]:
    problems = []
    if n_blank := int(_blank(df[id_col]).sum()):
        problems.append(f"{table}: {n_blank} row(s) with empty {id_col}")
    dup = df[id_col][df[id_col].duplicated()].tolist()
    if dup:
        problems.append(f"{table}: duplicate {id_col} {dup}")
    return problems


def _check_range(df: pd.DataFrame, id_col: str, table: str, prefix: str = "") -> list[str]:
    lo, mo, hi = (df[f"{prefix}{b}"] for b in ("low", "mode", "high"))
    problems = []
    for i in df.index[lo.isna() | mo.isna() | hi.isna()]:
        problems.append(f"{table}: {df.loc[i, id_col]}: missing {prefix}low/mode/high")
    for i in df.index[~((lo <= mo) & (mo <= hi))]:
        if not (pd.isna(lo[i]) or pd.isna(mo[i]) or pd.isna(hi[i])):
            problems.append(
                f"{table}: {df.loc[i, id_col]}: need {prefix}low <= mode <= high, "
                f"got {lo[i]}, {mo[i]}, {hi[i]}"
            )
    for i in df.index[lo < 0]:
        problems.append(f"{table}: {df.loc[i, id_col]}: negative {prefix}low {lo[i]}")
    return problems


def _check_evidence(df: pd.DataFrame, id_col: str, table: str) -> list[str]:
    """Grades are A-D; a row mixing sources may use 'A/D' (one grade per part)."""
    problems = []
    for _, row in df.iterrows():
        grades = str(row["evidence"]).split("/")
        if pd.isna(row["evidence"]) or not set(grades) <= EVIDENCE_GRADES:
            problems.append(f"{table}: {row[id_col]}: bad evidence grade {row['evidence']!r}")
    return problems


def _check_sources(df: pd.DataFrame, id_col: str, table: str) -> list[str]:
    return [f"{table}: {df.loc[i, id_col]}: empty source" for i in df.index[_blank(df["source"])]]


def _check_allowed(
    df: pd.DataFrame, id_col: str, col: str, allowed: frozenset[str], table: str
) -> list[str]:
    bad = df[~df[col].isin(allowed)]
    return [f"{table}: {r[id_col]}: unknown {col} {r[col]!r}" for _, r in bad.iterrows()]


def _check_common(
    df: pd.DataFrame, columns: list[str], id_col: str, table: str, graded: bool = True
) -> list[str]:
    problems = _check_columns(df, columns, table)
    if problems:  # the other checks assume the right columns
        return problems
    problems += _check_ids(df, id_col, table) + _check_sources(df, id_col, table)
    if graded:
        problems += _check_evidence(df, id_col, table)
    return problems


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProcedureOverride:
    """Country-specific price replacing ``FR price x site index`` (D25).

    ``price`` is already a local price in EUR, so it is never multiplied by the
    site price index; ``apply_overhead`` says whether the country's site
    overhead is still added on top.
    """

    price: Triangular
    apply_overhead: bool


@dataclass(frozen=True)
class SponsorModelFactor:
    """Multiplier on CRO-delivered labour for a sponsor model (D31).

    Applies to study items with ``country_scaling == none`` whose resource is
    in ``resources``. ``full_cro`` has no row: factor 1.
    """

    factor: Triangular
    resources: frozenset[str]


@dataclass(frozen=True)
class Catalogue:
    """Validated catalogue. DataFrames are indexed by their ID column."""

    procedures: pd.DataFrame
    study_items: pd.DataFrame
    countries: pd.DataFrame
    constants: pd.DataFrame
    benchmarks: pd.DataFrame
    overrides: pd.DataFrame  # indexed by (procedure_id, country)
    sponsor_factors: pd.DataFrame  # indexed by sponsor_model
    conditions: dict[str, Condition]  # item_id -> parsed applies_if

    @staticmethod
    def _triangular(row: pd.Series, prefix: str = "") -> Triangular:
        return Triangular(
            low=float(row[f"{prefix}low"]),
            mode=float(row[f"{prefix}mode"]),
            high=float(row[f"{prefix}high"]),
        )

    def procedure_price(self, procedure_id: str) -> Triangular:
        """Per-occurrence price at a French (reference) site, EUR 2026."""
        return self._triangular(self.procedures.loc[procedure_id])

    def procedure_override(self, procedure_id: str, country: str) -> ProcedureOverride | None:
        """Country-specific price for a procedure, or ``None`` if there is none."""
        key = (procedure_id, country)
        if key not in self.overrides.index:
            return None
        row = self.overrides.loc[key]
        return ProcedureOverride(
            price=self._triangular(row), apply_overhead=bool(row["apply_overhead"])
        )

    def sponsor_model_factor(self, sponsor_model: str) -> SponsorModelFactor | None:
        """Labour multiplier for a sponsor model, or ``None`` (factor 1)."""
        if sponsor_model not in self.sponsor_factors.index:
            return None
        row = self.sponsor_factors.loc[sponsor_model]
        return SponsorModelFactor(
            factor=self._triangular(row), resources=frozenset(row["resources"].split(";"))
        )

    def item_price(self, item_id: str) -> Triangular:
        """Unit price of a study item per unit of its driver, EUR 2026."""
        return self._triangular(self.study_items.loc[item_id])

    def constant(self, constant_id: str) -> Triangular:
        return self._triangular(self.constants.loc[constant_id])

    def country_range(self, country: str, name: str) -> Triangular:
        """One of ``COUNTRY_RANGES`` for a country, e.g. ``("DE", "site_price")``."""
        if name not in COUNTRY_RANGES:
            raise KeyError(f"unknown country range {name!r}; known: {COUNTRY_RANGES}")
        return self._triangular(self.countries.loc[country], prefix=f"{name}_")

    def activation_months(self, design: Design, country: str) -> AnyDist:
        """Activation delay for a country: design value if given, else catalogue (D21)."""
        alloc = next(c for c in design.countries if c.country == country)
        if alloc.activation_months is not None:
            return alloc.activation_months
        return self.country_range(country, "activation_months")

    def applicable_items(self, design: Design, country: str | None = None) -> list[str]:
        """Item IDs whose condition holds. Country conditions need ``country``.

        With ``country=None`` only items whose condition has no country atom
        are considered; with a country, all items whose condition holds there.
        """
        return [
            item_id
            for item_id, cond in self.conditions.items()
            if (country is not None or not cond.needs_country) and cond.holds(design, country)
        ]

    def check_design(self, design: Design) -> None:
        """Raise ``CatalogueError`` if the design refers to unknown IDs."""
        problems = []
        known_procedures = set(self.procedures.index)
        for v in design.visits:
            for p in v.procedures:
                if p not in known_procedures:
                    problems.append(f"design: visit {v.id}: unknown procedure {p!r}")
        for c in design.countries:
            if c.country not in self.countries.index:
                problems.append(f"design: unknown country {c.country!r} (not in countries.csv)")
        if problems:
            raise CatalogueError(problems)


def _validate(
    procedures: pd.DataFrame,
    study_items: pd.DataFrame,
    countries: pd.DataFrame,
    constants: pd.DataFrame,
    benchmarks: pd.DataFrame,
    procedure_overrides: pd.DataFrame,
    sponsor_model_factors: pd.DataFrame,
) -> tuple[list[str], dict[str, Condition]]:
    problems: list[str] = []
    conditions: dict[str, Condition] = {}

    # procedures.csv
    p = _check_common(procedures, PROCEDURE_COLUMNS, "procedure_id", "procedures")
    if not p:
        p += _check_range(procedures, "procedure_id", "procedures")
        p += _check_allowed(
            procedures, "procedure_id", "country_scaling", COUNTRY_SCALINGS, "procedures"
        )
        known = set(procedures["procedure_id"])
        for pid in (*AUTO_ONSITE_PROCEDURES, TRAVEL_PROCEDURE):
            if pid not in known:
                p.append(f"procedures: automatic procedure {pid!r} missing")
    problems += p
    known_procedures = set(procedures.get("procedure_id", []))

    # study_items.csv
    s = _check_common(study_items, STUDY_ITEM_COLUMNS, "item_id", "study_items")
    if not s:
        s += _check_range(study_items, "item_id", "study_items")
        s += _check_allowed(study_items, "item_id", "driver", DRIVERS, "study_items")
        s += _check_allowed(
            study_items, "item_id", "country_scaling", COUNTRY_SCALINGS, "study_items"
        )
        for _, row in study_items.iterrows():
            item = row["item_id"]
            try:
                cond = parse_condition(str(row["applies_if"]))
            except ValueError as e:
                s.append(f"study_items: {item}: {e}")
                continue
            if cond.needs_country and row["driver"] in STUDY_LEVEL_DRIVERS:
                s.append(f"study_items: {item}: country condition on {row['driver']} item")
            if row["country_scaling"] == "site_price" and row["driver"] in STUDY_LEVEL_DRIVERS:
                s.append(f"study_items: {item}: site_price scaling on {row['driver']} item")
            for pid in cond.procedures_used:
                if pid not in known_procedures:
                    s.append(f"study_items: {item}: uses unknown procedure {pid!r}")
            conditions[item] = cond
    problems += s

    # countries.csv
    c = _check_common(countries, COUNTRY_COLUMNS, "country", "countries")
    if not c:
        for code in countries["country"][
            ~countries["country"].astype(str).str.fullmatch("[A-Z]{2}")
        ]:
            c.append(f"countries: bad country code {code!r}")
        for name in COUNTRY_RANGES:
            c += _check_range(countries, "country", "countries", prefix=f"{name}_")
        ref = countries[countries["country"] == REFERENCE_COUNTRY]
        if ref.empty:
            c.append(f"countries: reference country {REFERENCE_COUNTRY} missing")
        else:
            r = ref.iloc[0]
            if not (r.site_price_low == r.site_price_mode == r.site_price_high == 1.0):
                c.append(f"countries: {REFERENCE_COUNTRY} site price index must be fixed at 1")
            if not (r.site_overhead_low == r.site_overhead_mode == r.site_overhead_high == 0.0):
                c.append(f"countries: {REFERENCE_COUNTRY} site overhead must be fixed at 0")
    problems += c

    # constants.csv
    k = _check_common(constants, CONSTANT_COLUMNS, "constant_id", "constants")
    if not k:
        k += _check_range(constants, "constant_id", "constants")
        for cid in sorted(REQUIRED_CONSTANTS - set(constants["constant_id"])):
            k.append(f"constants: required constant {cid!r} missing")
    problems += k

    # procedure_overrides.csv
    o = _check_common(procedure_overrides, OVERRIDE_COLUMNS, "procedure_id", "procedure_overrides")
    if not o:
        o += _check_range(procedure_overrides, "procedure_id", "procedure_overrides")
        keys = procedure_overrides[["procedure_id", "country"]]
        for _, r in keys[keys.duplicated()].iterrows():
            o.append(f"procedure_overrides: duplicate ({r.procedure_id}, {r.country})")
        for _, r in procedure_overrides.iterrows():
            tag = f"procedure_overrides: ({r.procedure_id}, {r.country})"
            if r.procedure_id not in known_procedures:
                o.append(f"{tag}: unknown procedure")
            if r.country not in set(countries.get("country", [])):
                o.append(f"{tag}: unknown country")
            if r.apply_overhead not in (True, False):
                o.append(f"{tag}: apply_overhead must be true/false, got {r.apply_overhead!r}")
    problems += o

    # sponsor_model_factors.csv
    f = _check_common(
        sponsor_model_factors, SPONSOR_FACTOR_COLUMNS, "sponsor_model", "sponsor_model_factors"
    )
    if not f:
        f += _check_range(sponsor_model_factors, "sponsor_model", "sponsor_model_factors")
        f += _check_allowed(
            sponsor_model_factors,
            "sponsor_model",
            "sponsor_model",
            _OPERATION_CHOICES["sponsor_model"],
            "sponsor_model_factors",
        )
        known_resources = set(study_items.get("resource", []))
        for _, r in sponsor_model_factors.iterrows():
            for res in str(r.resources).split(";"):
                if res not in known_resources:
                    f.append(f"sponsor_model_factors: {r.sponsor_model}: unknown resource {res!r}")
    problems += f

    # benchmarks.csv (validation only; no evidence grade)
    b = _check_common(benchmarks, BENCHMARK_COLUMNS, "benchmark_id", "benchmarks", graded=False)
    if not b:
        for i in benchmarks.index[~(benchmarks["value"] > 0)]:
            b.append(f"benchmarks: {benchmarks.loc[i, 'benchmark_id']}: value must be > 0")
    problems += b

    return problems, conditions


def load_catalogue(data_dir: str | Path = DATA_DIR) -> Catalogue:
    """Read and validate every CSV in ``data_dir``; raise ``CatalogueError`` on problems."""
    data_dir = Path(data_dir)
    tables = {
        name: pd.read_csv(data_dir / f"{name}.csv")
        for name in (
            "procedures",
            "study_items",
            "countries",
            "constants",
            "benchmarks",
            "procedure_overrides",
            "sponsor_model_factors",
        )
    }
    problems, conditions = _validate(**tables)
    if problems:
        raise CatalogueError(problems)
    return Catalogue(
        procedures=tables["procedures"].set_index("procedure_id"),
        study_items=tables["study_items"].set_index("item_id"),
        countries=tables["countries"].set_index("country"),
        constants=tables["constants"].set_index("constant_id"),
        benchmarks=tables["benchmarks"].set_index("benchmark_id"),
        overrides=tables["procedure_overrides"].set_index(["procedure_id", "country"]),
        sponsor_factors=tables["sponsor_model_factors"].set_index("sponsor_model"),
        conditions=conditions,
    )
