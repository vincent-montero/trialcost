"""ClinicalTrials.gov priors: parsing and estimation, offline (no network)."""

import math

import numpy as np
import pandas as pd
import pytest

from trialcost import ctgov
from trialcost.catalogue import load_catalogue


@pytest.mark.parametrize(
    ("text", "days"),
    [
        ("24 Weeks", 168),
        ("Baseline to Week 52", 364),
        ("Day 1 to Day 29", 29),
        ("up to 12 months", 12 * 30.44),
        ("2 years", 730.5),
        ("Baseline, Month 6, Month 12", 12 * 30.44),
        ("12-week double-blind period", 84),
    ],
)
def test_time_frame_days(text, days):
    assert ctgov.time_frame_days(text) == pytest.approx(days)


def test_time_frame_unparseable():
    assert ctgov.time_frame_days("End of study") is None


def _period(started: tuple, completed: tuple) -> dict:
    def milestone(kind, counts):
        return {"type": kind, "achievements": [{"numSubjects": str(n)} for n in counts]}

    return {"milestones": [milestone("STARTED", started), milestone("COMPLETED", completed)]}


def _study(**over):
    s = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001", "briefTitle": "t"},
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2020-01-01"},
                "primaryCompletionDateStruct": {"date": "2022-01-01"},
            },
            "conditionsModule": {"conditions": ["ALS"]},
            "designModule": {"phases": ["PHASE2"], "enrollmentInfo": {"count": 60}},
            "outcomesModule": {"primaryOutcomes": [{"timeFrame": "24 Weeks"}]},
            "contactsLocationsModule": {
                "locations": [{"country": "United States"}] * 3 + [{"country": "France"}] * 2
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "X"}},
        },
        "resultsSection": {
            "participantFlowModule": {
                "periods": [
                    _period(started=(30, 30), completed=(27, 24)),  # randomised phase
                    _period(started=(50,), completed=(10,)),  # extension: ignored
                ]
            }
        },
    }
    s["protocolSection"]["designModule"].update(over.get("design", {}))
    return s


def test_parse_study_by_hand():
    row = ctgov.parse_study(_study())
    assert row["n_sites"] == 5 and row["n_countries"] == 2 and row["us_share"] == 0.6
    assert row["window_months"] == pytest.approx(731 / 30.44)
    assert row["primary_months"] == pytest.approx(168 / 30.44)
    assert row["recruit_months"] == pytest.approx((731 - 168) / 30.44)
    assert (row["flow_started"], row["flow_completed"]) == (60, 51)  # first period only


def test_add_estimates_by_hand():
    t = ctgov.add_estimates(pd.DataFrame([ctgov.parse_study(_study())]), site_open_fraction=0.75)
    r = t.iloc[0]
    assert r["rate_crude"] == pytest.approx(60 / (5 * (731 - 168) / 30.44))
    assert r["rate"] == pytest.approx(r["rate_crude"] / 0.75)
    # 24-week period < 6 months minimum: no dropout estimate (D38)
    assert np.isnan(r["dropout_annual"])


def test_dropout_annualised_for_long_periods():
    study = _study()
    study["protocolSection"]["outcomesModule"]["primaryOutcomes"] = [{"timeFrame": "52 Weeks"}]
    r = ctgov.add_estimates(pd.DataFrame([ctgov.parse_study(study)]), 0.75).iloc[0]
    months = 364 / 30.44
    assert r["dropout_annual"] == pytest.approx(1 - (51 / 60) ** (12 / months))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"phases": ["PHASE1", "PHASE2"]}, "phase out of scope"),
        ({"enrollmentInfo": {"count": 0}}, "no enrollment"),
    ],
)
def test_exclusion_reasons(change, reason):
    t = ctgov.add_estimates(pd.DataFrame([ctgov.parse_study(_study(design=change))]), 0.75)
    assert t.iloc[0]["rate_excluded"] == reason
    assert np.isnan(t.iloc[0]["rate"])


def test_single_site_excluded():
    study = _study()
    study["protocolSection"]["contactsLocationsModule"]["locations"] = [{"country": "US"}]
    t = ctgov.add_estimates(pd.DataFrame([ctgov.parse_study(study)]), 0.75)
    assert t.iloc[0]["rate_excluded"].startswith("fewer than 2 sites")


def test_fits_recover_parameters():
    rng = np.random.default_rng(0)
    med, gsd = ctgov.fit_lognormal(pd.Series(rng.lognormal(math.log(0.4), math.log(2.5), 50_000)))
    assert med == pytest.approx(0.4, rel=0.02) and gsd == pytest.approx(2.5, rel=0.02)
    mean, kappa = ctgov.fit_beta(pd.Series(rng.beta(0.25 * 4, 0.75 * 4, 50_000)))
    assert mean == pytest.approx(0.25, abs=0.005) and kappa == pytest.approx(4, rel=0.05)


def test_committed_priors_reproduce_from_snapshot():
    """data/recruitment_priors.csv must be what the committed snapshot gives."""
    meta, studies = ctgov.load_snapshot()
    f = load_catalogue().constant("site_open_fraction").central()
    trials = ctgov.add_estimates(pd.DataFrame([ctgov.parse_study(s) for s in studies]), f)
    fresh = ctgov.priors_table(trials, meta, f)
    committed = pd.read_csv(ctgov.DATA_DIR / "recruitment_priors.csv")
    pd.testing.assert_frame_equal(
        fresh[["subset", "parameter", "p1", "p2", "n_trials"]].reset_index(drop=True),
        committed[["subset", "parameter", "p1", "p2", "n_trials"]],
        check_dtype=False,
    )
