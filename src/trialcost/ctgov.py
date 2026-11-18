"""Recruitment and dropout priors from ClinicalTrials.gov (README §2.4, roadmap #2).

    uv run python -m trialcost.ctgov fetch      # network: query API v2, save raw snapshot
    uv run python -m trialcost.ctgov estimate   # offline: trial table + priors CSV

Two stages, so the estimation is reproducible from the saved snapshot and
testable without network access.

Query (D37, agreed with Vincent)
--------------------------------
Conditions from ``data/ctgov/rare_neuro_conditions.csv`` (OR-ed); interventional;
phase 2, 2/3 or 3; randomised; industry lead sponsor; start date >= 2010;
status COMPLETED or TERMINATED; actual (not anticipated) enrollment.

Per-trial recruitment rate
--------------------------
ClinicalTrials.gov gives the actual enrollment N, the number of listed sites S,
the start date (first participant enrolled) and the primary completion date
(last participant's primary-outcome assessment). The recruitment window is::

    T_recruit = (primary completion - start) - D_primary

with ``D_primary`` the longest primary-outcome time frame that can be parsed
("24 Weeks", "Baseline to Month 12", "Day 1 to Day 29"...). Sites open
progressively, so on average a site is open for only a fraction ``f`` of the
window (``constants.csv: site_open_fraction``). The rate per open site-month,
comparable with the engine's ``rate_mean``, is::

    rate = N / (S x T_recruit x f)

Data-quality rules (D38). A trial stays in the table but is excluded from the
rate fit, with the reason recorded, if: its phase is out of scope (phase 1/2,
returned by the API's phase filter); fewer than 2 sites are listed (single
generic locations are common placeholders: 144-421 patients at "1 site");
the primary time frame cannot be parsed; or ``T_recruit`` < 3 months (the
primary time frame fills the window, so the window estimate is unreliable).

Per-trial dropout
-----------------
From posted results (participant flow, **first period only**, i.e. the
randomised phase): ``p = 1 - completed / started``, annualised with the
engine's constant-hazard model over ``D_primary``::

    p_annual = 1 - (1 - p) ** (12 / D_primary_months)

Only trials in scope with ``D_primary`` >= 6 months are used: dropout is often
front-loaded, so annualising 2-3 month periods with a constant hazard
over-extrapolates (D38).

Priors
------
- ``rate_mean`` ~ LogNormal(median, gsd) fitted on log rates across trials:
  median = exp(mean log rate), gsd = exp(SD of log rate). The spread is
  between-trial variability, which is what ``rate_mean`` represents for a new
  trial of the segment.
- ``annual_dropout_rate`` ~ Beta(mean, kappa) by method of moments:
  kappa = mean (1 - mean) / variance - 1.
- ``rate_shape`` (between-site heterogeneity) cannot be estimated from
  trial-level totals and is not produced here.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trialcost import DATA_DIR
from trialcost.catalogue import load_catalogue
from trialcost.design import DAYS_PER_MONTH

API = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_DIR = DATA_DIR / "ctgov"
SEGMENT = "rare_neuro"
IN_SCOPE_PHASES = {"PHASE2", "PHASE2/PHASE3", "PHASE3"}
MIN_SITES = 2
MIN_RECRUIT_MONTHS = 3.0
MIN_DROPOUT_PERIOD_MONTHS = 6.0
FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.designModule",
    "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.contactsLocationsModule.locations",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
    "resultsSection.participantFlowModule",
]
FILTER = (
    "AREA[Phase](PHASE2 OR PHASE3) AND AREA[StudyType]INTERVENTIONAL "
    "AND AREA[DesignAllocation]RANDOMIZED AND AREA[LeadSponsorClass]INDUSTRY "
    "AND AREA[OverallStatus](COMPLETED OR TERMINATED) "
    "AND AREA[EnrollmentType]ACTUAL AND AREA[StartDate]RANGE[2010-01-01,MAX]"
)

# --------------------------------------------------------------------------- #
# Fetch (network)
# --------------------------------------------------------------------------- #


def conditions(path: Path = CTGOV_DIR / "rare_neuro_conditions.csv") -> list[str]:
    return pd.read_csv(path)["condition_term"].tolist()


def fetch(terms: list[str], page_size: int = 500) -> list[dict]:
    """All studies matching the query, following ``nextPageToken``."""
    params = {
        "query.cond": " OR ".join(f'"{t}"' for t in terms),
        "filter.advanced": FILTER,
        "fields": ",".join(FIELDS),
        "pageSize": str(page_size),
    }
    studies: list[dict] = []
    while True:
        url = f"{API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "trialcost-research/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            page = json.load(resp)
        studies += page.get("studies", [])
        token = page.get("nextPageToken")
        if not token:
            return studies
        params["pageToken"] = token


def save_snapshot(studies: list[dict], terms: list[str], out_dir: Path = CTGOV_DIR) -> Path:
    path = out_dir / f"{SEGMENT}_snapshot.json.gz"
    meta = {"fetched": date.today().isoformat(), "api": API, "filter": FILTER, "conditions": terms}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"meta": meta, "studies": studies}, f)
    return path


def load_snapshot(
    path: Path = CTGOV_DIR / f"{SEGMENT}_snapshot.json.gz",
) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data["meta"], data["studies"]


# --------------------------------------------------------------------------- #
# Parse (pure functions)
# --------------------------------------------------------------------------- #

_UNIT_DAYS = {"day": 1.0, "week": 7.0, "month": DAYS_PER_MONTH, "year": 365.25}
_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(day|week|month|year)s?\b", re.IGNORECASE)
_DURATION_REVERSED = re.compile(r"\b(day|week|month|year)s?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def time_frame_days(text: str) -> float | None:
    """Longest duration mentioned in a time-frame string, in days.

    Understands "24 Weeks", "up to 12 months", "Baseline to Week 52",
    "Day 1 to Day 29", "2 years". Returns ``None`` if nothing parses.

    >>> time_frame_days("Baseline to Week 52")
    364.0
    >>> time_frame_days("Day 1 to Day 29")
    29.0
    """
    found = [float(n) * _UNIT_DAYS[u.lower()] for n, u in _DURATION.findall(text)]
    found += [float(n) * _UNIT_DAYS[u.lower()] for u, n in _DURATION_REVERSED.findall(text)]
    return max(found) if found else None


def _month_diff(start: str, end: str) -> float:
    """Months between two CT.gov dates ("YYYY-MM-DD" or "YYYY-MM": day 15 assumed)."""

    def parse(s: str) -> date:
        parts = [int(x) for x in s.split("-")]
        return date(parts[0], parts[1], parts[2] if len(parts) > 2 else 15)

    return (parse(end) - parse(start)).days / DAYS_PER_MONTH


def first_period_flow(results: dict | None) -> tuple[float, float] | None:
    """(started, completed) summed over groups in the first participant-flow period."""
    flow = (results or {}).get("participantFlowModule")
    if not flow or not flow.get("periods"):
        return None
    totals = {}
    for m in flow["periods"][0].get("milestones", []):
        if m.get("type") in ("STARTED", "COMPLETED"):
            try:
                totals[m["type"]] = sum(float(a["numSubjects"]) for a in m["achievements"])
            except (KeyError, ValueError):
                return None
    if "STARTED" not in totals or "COMPLETED" not in totals or totals["STARTED"] <= 0:
        return None
    return totals["STARTED"], totals["COMPLETED"]


def parse_study(study: dict) -> dict:
    """One row of the trial table (raw facts + derived quantities, no fitting)."""
    ps = study["protocolSection"]
    status, design = ps["statusModule"], ps["designModule"]
    locations = ps.get("contactsLocationsModule", {}).get("locations", [])
    frames = [
        o.get("timeFrame", "") for o in ps.get("outcomesModule", {}).get("primaryOutcomes", [])
    ]
    parsed = [d for d in (time_frame_days(t) for t in frames) if d]
    d_primary = max(parsed) / DAYS_PER_MONTH if parsed else None
    start = status.get("startDateStruct", {}).get("date")
    pcd = status.get("primaryCompletionDateStruct", {}).get("date")
    window = _month_diff(start, pcd) if start and pcd else None
    flow = first_period_flow(study.get("resultsSection"))
    countries = sorted({loc.get("country", "") for loc in locations} - {""})
    return {
        "nct_id": ps["identificationModule"]["nctId"],
        "title": ps["identificationModule"].get("briefTitle", ""),
        "conditions": "; ".join(ps.get("conditionsModule", {}).get("conditions", [])),
        "phase": "/".join(design.get("phases", [])),
        "status": status.get("overallStatus"),
        "sponsor": ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
        "start": start,
        "primary_completion": pcd,
        "enrollment": design.get("enrollmentInfo", {}).get("count"),
        "n_sites": len(locations),
        "n_countries": len(countries),
        "us_share": (
            sum(loc.get("country") == "United States" for loc in locations) / len(locations)
            if locations
            else None
        ),
        "window_months": window,
        "primary_months": d_primary,
        "recruit_months": window - d_primary if window is not None and d_primary else None,
        "flow_started": flow[0] if flow else None,
        "flow_completed": flow[1] if flow else None,
    }


def add_estimates(trials: pd.DataFrame, site_open_fraction: float) -> pd.DataFrame:
    """Per-trial rate and annual dropout, with an exclusion reason when not usable."""
    t = trials.copy()
    reason = pd.Series("", index=t.index, dtype=object)

    def exclude(mask: pd.Series, why: str) -> None:
        reason[(reason == "") & mask] = why

    in_scope = t["phase"].isin(IN_SCOPE_PHASES)
    exclude(~in_scope, "phase out of scope")
    exclude(t["n_sites"] < MIN_SITES, f"fewer than {MIN_SITES} sites listed")
    exclude(t["primary_months"].isna(), "primary time frame not parseable")
    exclude(
        ~(t["recruit_months"] >= MIN_RECRUIT_MONTHS),
        f"recruitment window < {MIN_RECRUIT_MONTHS:g} months",
    )
    exclude(~(t["enrollment"] > 0), "no enrollment")
    ok = reason == ""
    t["rate_crude"] = np.where(ok, t["enrollment"] / (t["n_sites"] * t["recruit_months"]), np.nan)
    t["rate"] = t["rate_crude"] / site_open_fraction
    t["rate_excluded"] = reason

    p = 1 - t["flow_completed"] / t["flow_started"]
    usable = in_scope & p.between(0, 1) & (t["primary_months"] >= MIN_DROPOUT_PERIOD_MONTHS)
    t["dropout_period"] = np.where(usable, p, np.nan)
    t["dropout_annual"] = np.where(
        usable, 1 - (1 - p.clip(upper=0.999)) ** (12 / t["primary_months"]), np.nan
    )
    return t


def fit_lognormal(x: pd.Series) -> tuple[float, float]:
    """(median, gsd) from the mean and SD of log values."""
    logs = np.log(x.dropna())
    return float(math.exp(logs.mean())), float(math.exp(logs.std(ddof=1)))


def fit_beta(x: pd.Series) -> tuple[float, float]:
    """(mean, kappa) by method of moments; kappa = m (1 - m) / var - 1."""
    v = x.dropna()
    m, var = float(v.mean()), float(v.var(ddof=1))
    return m, m * (1 - m) / var - 1


def priors_table(t: pd.DataFrame, meta: dict, site_open_fraction: float) -> pd.DataFrame:
    """Priors for the segment (all trials, and by phase / status as sensitivity rows)."""
    source = (
        f"Derived: ClinicalTrials.gov API v2 snapshot {meta['fetched']} "
        f"(data/ctgov/{SEGMENT}_snapshot.json.gz); method in src/trialcost/ctgov.py"
    )
    rows = []

    def add(subset_name: str, sub: pd.DataFrame):
        rates = sub["rate"].dropna()
        if len(rates) >= 5:
            med, gsd = fit_lognormal(rates)
            rows.append(
                {
                    "segment": SEGMENT,
                    "subset": subset_name,
                    "parameter": "rate_mean",
                    "kind": "lognormal",
                    "p1_name": "median",
                    "p1": round(med, 4),
                    "p2_name": "gsd",
                    "p2": round(gsd, 3),
                    "n_trials": len(rates),
                    "evidence": "C",
                    "source": source,
                    "notes": f"rate = N / (sites x recruitment months x {site_open_fraction}); "
                    f"crude median {sub['rate_crude'].median():.3f} pts/site/month",
                }
            )
        drops = sub["dropout_annual"].dropna()
        if len(drops) >= 5:
            mean, kappa = fit_beta(drops)
            rows.append(
                {
                    "segment": SEGMENT,
                    "subset": subset_name,
                    "parameter": "annual_dropout_rate",
                    "kind": "beta",
                    "p1_name": "mean",
                    "p1": round(mean, 4),
                    "p2_name": "kappa",
                    "p2": round(kappa, 2),
                    "n_trials": len(drops),
                    "evidence": "C",
                    "source": source,
                    "notes": "first participant-flow period, annualised over the primary "
                    f"time frame; median per-period dropout {sub['dropout_period'].median():.3f}",
                }
            )

    t = t[t["phase"].isin(IN_SCOPE_PHASES)]
    add("all", t)
    for phase, sub in t.groupby("phase"):
        add(f"phase={phase}", sub)
    for status, sub in t.groupby("status"):
        add(f"status={status}", sub)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m trialcost.ctgov")
    p.add_argument("command", choices=["fetch", "estimate"])
    args = p.parse_args(argv)
    if args.command == "fetch":
        terms = conditions()
        studies = fetch(terms)
        path = save_snapshot(studies, terms)
        print(f"Fetched {len(studies)} studies -> {path}")
        return 0
    meta, studies = load_snapshot()
    f = load_catalogue().constant("site_open_fraction").central()
    trials = add_estimates(pd.DataFrame([parse_study(s) for s in studies]), f)
    trials.to_csv(CTGOV_DIR / f"{SEGMENT}_trials.csv", index=False)
    priors = priors_table(trials, meta, f)
    priors.to_csv(DATA_DIR / "recruitment_priors.csv", index=False)
    print(priors[["subset", "parameter", "p1_name", "p1", "p2_name", "p2", "n_trials"]].to_string())
    print(f"\nExcluded from rate fit: {trials['rate_excluded'].ne('').sum()} of {len(trials)}")
    print(trials.loc[trials["rate_excluded"].ne(""), "rate_excluded"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
