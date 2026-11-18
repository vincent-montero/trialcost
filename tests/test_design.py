from pathlib import Path

import pytest
from pydantic import ValidationError

from trialcost.design import DAYS_PER_MONTH, Design, Visit, load_design, visits_every
from trialcost.distributions import Beta, Fixed, LogNormal, Triangular

ARCHETYPE = Path(__file__).resolve().parents[1] / "designs" / "rare-neuro-ph2-mri-scales.yaml"


@pytest.fixture
def archetype() -> Design:
    return load_design(ARCHETYPE)


def _minimal(**overrides) -> dict:
    d = {
        "name": "t",
        "n_randomised": 10,
        "arms": [{"name": "a", "ratio": 1}],
        "visits": [
            {"id": "SCR", "day": -14, "screening": True, "procedures": ["consent"]},
            {"id": "BL", "day": 0, "procedures": ["physical_exam"]},
        ],
        "countries": [{"country": "FR", "n_sites": 2, "activation_months": 4}],
        "population": {
            "screen_failure_rate": 0.2,
            "annual_dropout_rate": 0.1,
            "rate_mean": 0.5,
            "rate_shape": 2,
        },
    }
    d.update(overrides)
    return d


def test_archetype_loads(archetype):
    assert archetype.n_randomised == 50
    assert archetype.n_sites == 15
    assert [c.country for c in archetype.countries] == ["FR", "DE", "ES", "IT", "US"]
    assert len(archetype.visits) == 8
    assert sum(v.remote for v in archetype.visits) == 2
    assert sum(v.screening for v in archetype.visits) == 1
    assert archetype.operations.central_imaging_reading and archetype.operations.dmc


def test_archetype_distributions_parsed(archetype):
    pop = archetype.population
    assert isinstance(pop.screen_failure_rate, Beta)
    assert isinstance(pop.rate_mean, LogNormal)
    assert isinstance(archetype.countries[0].activation_months, Triangular)
    assert archetype.operations.monitoring_interval_months == Fixed(value=2.0)


def test_patient_duration_derived_from_last_visit(archetype):
    assert archetype.patient_duration_days == 392
    assert archetype.patient_duration_months == pytest.approx(392 / DAYS_PER_MONTH)


def test_plain_numbers_coerced_and_defaults_validated():
    d = Design.model_validate(_minimal())
    assert d.countries[0].activation_months == Fixed(value=4.0)
    assert d.operations.monitoring_interval_months == Fixed(value=2.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_randomised": 0},
        {"arms": []},
        {"countries": []},
        {"arms": [{"name": "a", "ratio": 1}, {"name": "a", "ratio": 1}]},
        {"countries": [{"country": "fr", "n_sites": 1, "activation_months": 4}]},
        {
            "countries": [
                {"country": "FR", "n_sites": 1, "activation_months": 4},
                {"country": "FR", "n_sites": 2, "activation_months": 4},
            ]
        },
        {"visits": [{"id": "SCR", "day": -14, "screening": True, "procedures": []}]},
        {
            "visits": [
                {"id": "BL", "day": 0, "procedures": []},
                {"id": "BL", "day": 28, "procedures": []},
            ]
        },
    ],
)
def test_inconsistent_designs_rejected(overrides):
    with pytest.raises(ValidationError):
        Design.model_validate(_minimal(**overrides))


@pytest.mark.parametrize(
    ("day", "screening"),
    [(0, True), (5, True), (-3, False)],
)
def test_screening_flag_must_match_day_sign(day, screening):
    with pytest.raises(ValidationError):
        Visit(id="V", day=day, screening=screening, procedures=[])


def test_visits_every():
    vs = visits_every(weeks=4, until=52, procedures=["physical_exam"])
    assert len(vs) == 13
    assert vs[0].id == "W4" and vs[0].day == 28
    assert vs[-1].id == "W52" and vs[-1].day == 364
    vs[0].procedures.append("ecg")  # each visit owns its own list
    assert vs[1].procedures == ["physical_exam"]


def test_visits_every_rejects_bad_arguments():
    with pytest.raises(ValueError):
        visits_every(weeks=0, until=52, procedures=[])
    with pytest.raises(ValueError):
        visits_every(weeks=8, until=4, procedures=[])
