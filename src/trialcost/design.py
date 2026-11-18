"""Trial design schema (README §3), YAML loader and quick-mode helpers.

The fields are exactly the agreed schema. On top of it this module adds:

- consistency checks that do not change the schema (unique IDs, screening
  visits before randomisation, at least one post-randomisation visit);
- derived quantities that have a single source of truth in the design
  (``n_sites``, ``patient_duration_months``);
- ``visits_every()`` to generate a regular visit schedule in quick mode;
- ``load_design()`` to read a YAML file.

Procedure IDs are *not* checked here: that needs the catalogue and is done in
``catalogue.py``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from trialcost.distributions import Dist

#: Average days per month (365.25 / 12). Visit days are converted with it.
DAYS_PER_MONTH = 30.44


def _duplicates(values: list[str]) -> list[str]:
    return sorted(v for v, n in Counter(values).items() if n > 1)


class Arm(BaseModel):
    name: str
    ratio: int = Field(ge=1)
    placebo: bool = False


class Visit(BaseModel):
    id: str
    day: int  # relative to randomisation; screening < 0
    procedures: list[str]  # IDs from data/procedures.csv
    screening: bool = False  # applied to N_screened
    remote: bool = False  # no on-site costs

    @model_validator(mode="after")
    def _check_day(self):
        if self.screening and self.day >= 0:
            raise ValueError(f"screening visit {self.id!r} must have day < 0")
        if not self.screening and self.day < 0:
            raise ValueError(f"visit {self.id!r} has day < 0 but is not marked screening")
        return self


class CountryAllocation(BaseModel):
    country: str = Field(pattern=r"^[A-Z]{2}$")
    n_sites: int = Field(ge=1)
    activation_months: Dist


class Population(BaseModel):
    screen_failure_rate: Dist
    annual_dropout_rate: Dist
    rate_mean: Dist  # randomised patients / site / month
    rate_shape: float = Field(gt=0)


class Operations(BaseModel):
    sponsor_model: Literal["full_cro", "functional", "in_house"] = "full_cro"
    monitoring: Literal["full_sdv", "risk_based"] = "risk_based"
    # validate_default so the plain 2.0 is coerced to Fixed like a YAML value.
    monitoring_interval_months: Dist = Field(default=2.0, validate_default=True)
    central_lab: bool = True
    central_imaging_reading: bool = False
    irt: bool = True
    dmc: bool = False
    n_interim_analyses: int = 0
    patient_travel_support: bool = True


class Design(BaseModel):
    name: str
    n_randomised: int = Field(ge=1)
    arms: list[Arm]
    visits: list[Visit]
    countries: list[CountryAllocation]
    population: Population
    operations: Operations = Operations()

    @model_validator(mode="after")
    def _check_consistency(self):
        for label, values in [
            ("arm names", [a.name for a in self.arms]),
            ("visit ids", [v.id for v in self.visits]),
            ("countries", [c.country for c in self.countries]),
        ]:
            if dup := _duplicates(values):
                raise ValueError(f"duplicate {label}: {dup}")
        if not self.arms:
            raise ValueError("design needs at least one arm")
        if not self.countries:
            raise ValueError("design needs at least one country")
        if not any(not v.screening for v in self.visits):
            raise ValueError("design needs at least one post-randomisation visit")
        return self

    @property
    def n_sites(self) -> int:
        return sum(c.n_sites for c in self.countries)

    @property
    def patient_duration_days(self) -> int:
        """Randomisation to last visit, in days (derived from the schedule)."""
        return max(v.day for v in self.visits if not v.screening)

    @property
    def patient_duration_months(self) -> float:
        return self.patient_duration_days / DAYS_PER_MONTH


def visits_every(
    weeks: int,
    until: int,
    procedures: list[str],
    *,
    remote: bool = False,
    id_prefix: str = "W",
) -> list[Visit]:
    """Regular post-randomisation visits every ``weeks`` weeks up to week ``until``.

    Visit days are ``7 * week``; IDs are ``W4``, ``W8``, ... Baseline (day 0)
    and screening visits are not generated: add them explicitly.

    >>> [v.day for v in visits_every(weeks=4, until=12, procedures=["physical_exam"])]
    [28, 56, 84]
    """
    if weeks < 1 or until < weeks:
        raise ValueError("need weeks >= 1 and until >= weeks")
    return [
        Visit(id=f"{id_prefix}{w}", day=7 * w, procedures=list(procedures), remote=remote)
        for w in range(weeks, until + 1, weeks)
    ]


def load_design(path: str | Path) -> Design:
    """Read and validate a design YAML file."""
    with open(path, encoding="utf-8") as f:
        return Design.model_validate(yaml.safe_load(f))
