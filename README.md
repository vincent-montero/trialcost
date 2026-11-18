# Clinical Trial Cost Estimator

A design-driven, probabilistic estimator of the **total sponsor budget** of a clinical trial.
You describe a design (population, visit schedule, procedures, countries, sites, operational options); the tool returns a **cost distribution** (P10 / P50 / P90), an **estimated duration distribution**, and a **breakdown** by cost category, resource type and country.

> Status (2026-11-18): design phase complete, v0 cost catalogue built, throwaway sanity check run, package scaffolding in place (engine step 1/6). **No engine yet.**
> Personal project, built exclusively from public sources.

---

## 1. Goals

| # | Goal | v0 |
|---|------|----|
| G1 | Anticipate the cost of a trial from its design ("what would a trial with this design cost?") | ✅ |
| G2 | Give a ~10-minute estimate for proposals and feasibility work | ✅ |
| G3 | Decompose cost into items (procedures, staff, infrastructure, CRO services, fees) | ✅ |
| G4 | Express uncertainty honestly as distributions, not point values | ✅ |
| G5 | Judge whether a design is feasible **financially** and **operationally** (recruitment, duration) | ✅ (cost + time) |
| G6 | Judge **technical** feasibility (power as a function of N, endpoint variance, dropout) | ❌ v1 |
| G7 | Detailed, line-by-line budgets | ❌ later ("advanced mode") |

Non-goals for v0: site-level budgets (what a hospital invoices), academic-sponsor budgets, IMP manufacturing cost, sponsor internal costs.

---

## 2. Methodology

### 2.1 Core idea: design, not phase

Published figures such as "a phase 3 costs €X M" are **averages over a mixture of designs**. Phase correlates with N, duration and number of sites but does not cause cost. The estimator conditions on **design primitives** instead. Per-phase benchmarks then become a validation target: feeding typical designs should approximately reproduce them.

### 2.2 Decomposition

```
Design ──► Quantities ──► × Unit prices(country) ──► Cost
   └──► Time (recruitment) ──┘
```

```
C = C_fixed                                               # study-level items
  + Σ_sites [ c_activation + c_site/month × T_active ]    # site-level items
  + N_screened × c_screen                                 # screening visits incl. screen failures
  + Σ_patients Σ_visits Σ_procedures c_proc(country)      # per-patient, dropout-adjusted
  + T_total × c_management/month                          # time-driven items

N_screened = N / (1 − SF)
T_total    = T_startup + T_recruit + T_treatment + T_closeout
```

Four cost layers, each with its own driver:

| Layer | Examples | Driver |
|---|---|---|
| Fixed (study) | protocol, CTA, EDC build, SAP/CSR, IRT set-up, imaging charter | per study / per country |
| Per site | feasibility, activation, close-out, imaging qualification | number of sites |
| Per patient | visits × procedures, investigator fees, patient travel | N, N_screened, schedule, dropout |
| Time-driven | project management, site management, monitoring, PV, IRT hosting | duration (recruitment speed) |

### 2.3 Design primitives (inputs)

- **Population**: screen-failure rate, annual dropout rate, recruitment rate per site.
- **Endpoint**: the procedures it requires (MRI, clinician scales, PRO, labs, PK, lumbar puncture…), central reading, rater certification.
- **Arms / ratio / blinding**: drug supply, IRT, placebo.
- **Schedule**: visits with days and procedures; patient duration is **derived** from the last visit (single source of truth).
- **Adaptive features**: DMC, interim analyses.
- **Geography**: countries, sites per country, activation delays.
- **Operations**: sponsor model (full CRO / functional / in-house), monitoring strategy and interval, central lab, central imaging, IRT, DMC, patient travel support.

### 2.4 Time model

- **Site activation**: per-site delay from study start, drawn per country (triangular).
- **Recruitment**: Poisson–gamma model (Anisimov & Fedorov): site rate λᵢ ~ Gamma(shape, shape / rate_mean); a site recruits only once active. Gives a distribution of T_recruit.
- **Treatment / follow-up**: derived from the visit schedule.
- **Close-out**: database lock, analysis and CSR, assumed 6 months after LPLV (assumption).

Time feeds every time-driven item, so **recruitment assumptions drive a large part of total cost** (see §6).

### 2.5 Dropout

Constant hazard derived from the annual dropout rate. Expected visits completed are computed per visit; dropouts stop generating per-visit costs. Requirement from Vincent: the tool must **explicitly report the cost impact of dropout** (e.g. cost per completer, delta vs. a no-dropout scenario), on top of its effect on interpretability.

### 2.6 Uncertainty

- Every numeric parameter can be a distribution: `fixed`, `triangular`, `lognormal` (median, geometric SD), `beta` (mean, concentration κ) for proportions.
- Monte Carlo: each run simulates one possible trial → joint outcome (duration, cost).
- Outputs: P10 / P50 / P90, breakdowns, tornado chart (which assumptions matter most), P(cost ≤ budget), P(recruitment complete by T).
- Distribution width reflects evidence grade (§4).

### 2.7 Prices

- **Single base currency: EUR**, fixed exchange rate **1 EUR = 1.15 USD** (mid-market ~1.15–1.16 in Nov 2026), **price year 2026**.
- **France is the reference country** for site procedure prices (public tariffs exist: convention unique + CCAM).
- Other countries via a **site price index** (triangular) and a **site overhead rate**; a procedure can carry a country-specific override (e.g. US research MRI).
- The catalogue contains ~20 generic procedures (v0); finer granularity (e.g. MRI with/without contrast) later.

---

## 3. Design schema (agreed draft)

Pydantic v2. A plain number in YAML is coerced to `Fixed`.

```python
from __future__ import annotations
from typing import Annotated, Literal, Union
from pydantic import BaseModel, BeforeValidator, Field, model_validator

class Fixed(BaseModel):
    kind: Literal["fixed"] = "fixed"
    value: float

class Triangular(BaseModel):
    kind: Literal["triangular"] = "triangular"
    low: float
    mode: float
    high: float

    @model_validator(mode="after")
    def _check(self):
        if not self.low <= self.mode <= self.high:
            raise ValueError("need low <= mode <= high")
        return self

class LogNormal(BaseModel):
    kind: Literal["lognormal"] = "lognormal"
    median: float = Field(gt=0)
    gsd: float = Field(gt=1)

class Beta(BaseModel):
    kind: Literal["beta"] = "beta"
    mean: float = Field(gt=0, lt=1)
    kappa: float = Field(gt=0)

def _coerce(v):
    return {"kind": "fixed", "value": v} if isinstance(v, (int, float)) else v

Dist = Annotated[Union[Fixed, Triangular, LogNormal, Beta],
                 Field(discriminator="kind"), BeforeValidator(_coerce)]

class Arm(BaseModel):
    name: str
    ratio: int = Field(ge=1)
    placebo: bool = False

class Visit(BaseModel):
    id: str
    day: int                      # relative to randomisation; screening < 0
    procedures: list[str]         # IDs from data/procedures.csv
    screening: bool = False       # applied to N_screened
    remote: bool = False          # no on-site costs

class CountryAllocation(BaseModel):
    country: str = Field(pattern=r"^[A-Z]{2}$")
    n_sites: int = Field(ge=1)
    activation_months: Dist

class Population(BaseModel):
    screen_failure_rate: Dist
    annual_dropout_rate: Dist
    rate_mean: Dist               # randomised patients / site / month
    rate_shape: float = Field(gt=0)

class Operations(BaseModel):
    sponsor_model: Literal["full_cro", "functional", "in_house"] = "full_cro"
    monitoring: Literal["full_sdv", "risk_based"] = "risk_based"
    monitoring_interval_months: Dist = 2.0
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
```

Planned helper for quick mode: `visits_every(weeks=4, until=52, procedures=[...])`.

### First archetype: `rare-neuro-ph2-mri-scales`

Phase 2, rare neurological disease, N = 50 (1:1 placebo), 52-week treatment + 4-week follow-up, 8 visits (2 remote), brain MRI at screening/W24/W52, clinician scale and PRO, PK at BL/W12; 15 sites: FR 4, DE 3, ES 3, IT 2, US 3; SF 35 %, dropout 15 %/year, recruitment median 0.25 pt/site/month (gsd 1.6, shape 1.5); central imaging reading and DMC on. Population and recruitment values are **placeholders**, to be replaced by ClinicalTrials.gov-derived priors.

---

## 4. Data catalogue (v0)

Files in `data/` (EUR, price year 2026):

| File | Content |
|---|---|
| `procedures.csv` | 20 per-occurrence site procedures (France reference) |
| `study_items.csv` | 39 study-, country-, site-, time- and patient-driven items |
| `countries.csv` | 6 countries: site price index, overhead, labour cost, CTA fees, activation delays |
| `benchmarks.csv` | 9 published benchmarks, used for validation only |

### Evidence grades

| Grade | Meaning | procedures | study_items |
|---|---|---|---|
| **A** sourced | Read directly from an official tariff or regulation | 12 | 4 |
| **B** secondary | Aggregator / planning guide citing industry data | 0 | 10 |
| **C** derived | Computed from sourced inputs + a stated assumption | 4 | 4 |
| **D** assumption | Expert-judgement placeholder, to be reviewed | 4 | 21 |

Rule agreed with Vincent: **assumptions and derivations are acceptable as long as they are transparent and traceable** (each row has `evidence`, `source`, `notes`).

---

## 5. Decision log

| # | Decision | Rationale |
|---|---|---|
| D1 | Perspective = **sponsor total budget** | The tool is for anticipation ("what would this design cost?") |
| D2 | Primary use = **quick ~10-min estimate**; detailed budgets later | Proposals and feasibility first |
| D3 | Scope: **open**; start narrow (interventional drug trials, industry sponsor + CRO, EU + US), engine generic from day 1 | Widening scope = adding data rows, not rewriting code |
| D4 | **Personal project** → public data only; no Thelonius or OrphanDev budgets | Separation of personal and employer work |
| D5 | **Design-driven, not phase-driven** | Phase is a correlate, not a cause (§2.1) |
| D6 | Hybrid: bottom-up engine, top-down UX via **archetypes** + ~10–15 inputs | 10-minute constraint |
| D7 | Architecture: **data / engine / UI separated**; engine = pure Python package | Maintainability; Vincent reviews Python more easily than JS |
| D8 | UI (proposed, not yet built): **Streamlit** (HF Spaces / Streamlit Cloud) for v0; **stlite** for a static portfolio page; FastAPI + JS only if it becomes a product | Engine swap-free |
| D9 | **Monte Carlo** with distributions; P10/P50/P90 + tornado | Uncertainty is large and accepted |
| D10 | Recruitment = **Poisson–gamma** | Standard, links to Vincent's trial-simulation work |
| D11 | **No technical feasibility (power) in v0**; N is an input | Validate cost and time engine first |
| D12 | **EUR**, fixed FX 1.15 USD/EUR, price year 2026 | Simplicity |
| D13 | ~20–30 **generic procedures** in v0 | Granularity later |
| D14 | **Dropout** via constant hazard; report cost impact of dropout explicitly | Vincent wants dropout cost quantified |
| D15 | Quick-mode **visit generator** helper | 10-minute constraint |
| D16 | **France = reference country** for unit prices | Public tariffs available (convention unique, CCAM) |
| D17 | **Evidence grades A–D** on every value | Traceability |
| D18 | First archetype = **rare neuro phase 2, MRI + scales** | Close to Vincent's domain, exercises most modules |

---

## 6. Key assumptions (to review)

Ranked by expected impact. All are in the CSVs with grade C or D.

1. **CRO bill-rate multiplier 2.5–3.5 × salary hourly rate** (feeds `monitoring_visit`, `project_management`). No public CRO rate card found. Base salaries: CRA ~$85.5k/y (ZipRecruiter), clinical PM ~$117k/y (Salary.com). For comparison, a staffing-agency markup on contractor pay is quoted at 25–40 % (Freelance Clinical); CRO rates are assumed much higher because they include overhead, utilisation and margin. **Most important parameter to validate.**
2. **Monitoring visit** = ~12 CRA hours (on-site day + prep + report) + travel €300–900; interval 2 months per active site.
3. **Site management** €1,000–2,500 per active site-month (Tufts/PharmExec ~$1,500–2,000). May partly double-count with monitoring visits: to be checked.
4. **Project management** 40–120 h/month.
5. **Site price indices** vs France: DE 1.1–1.6, ES 0.7–1.0, IT 0.8–1.1, NL 1.0–1.5, US 2.5–4.0. US anchored on research MRI (€900–1,300 vs ~€300 in FR). Labour costs are informative, not determinant.
6. **Site overhead**: FR 0 (convention unique includes management fees), EU 10–25 %, US 25–35 % (published institutional rates 31–35 %).
7. **Activation delays**: 3–12 months by country (Tufts: ~6 months existing sites, ~8.4 months new sites).
8. **Close-out**: 6 months from LPLV to CSR.
9. **Investigator fee / contreparties**: €500–3,000 per patient (country-scaled).
10. **Raters**: 2 per site; **DMC**: one meeting per 6 months of conduct.
11. **US MRI override**: €900–1,300 per scan.
12. **German BfArM fee** not found: CTA range = ethics fees + assumed €3–10k.
13. **Italy**: labour cost, fees and site index not verified.
14. **Excluded**: IMP manufacturing, sponsor internal staff, VAT, inflation indexing, explicit CRO margin (embedded in rates).

---

## 7. Sanity check (throwaway prototype, not validation)

`prototype/sanity_check.py`, 5,000 runs on the archetype:

| Bucket | P50 (€M) | Share |
|---|---|---|
| Monitoring | 1.11 | 23.6 % |
| Management (time-driven) | 0.97 | 20.5 % |
| Site start-up & close-out | 0.61 | 12.9 % |
| Site procedures | 0.54 | 11.5 % |
| Supply, safety, other | 0.51 | 10.9 % |
| Data & analysis | 0.37 | 7.8 % |
| Central imaging | 0.23 | 4.9 % |
| Regulatory | 0.22 | 4.8 % |
| Investigator fees | 0.12 | 2.5 % |
| **Total** | **4.71** (P10 4.15, P90 5.74) | |

- Duration P10/P50/P90: 32 / 38 / 52 months.
- Cost per randomised patient P50: ~€94k.
- **Observation**: ~45 % of cost is time-driven (monitoring + management) → recruitment assumptions dominate; these buckets also hold the most grade-D values.
- **Benchmarks (not directly comparable)**: ASPE CNS phase 2 ≈ €42k/patient (2018 $, ~243 patients/trial → fixed costs diluted); median phase 2 trial ≈ €7.5M (Martin 2017, larger trials); Tufts phase 2 direct cost ≈ €20.6k/day vs ~€4k/day here. Consistent in direction, possibly low; inconclusive.
- Prototype runtime ~2.4 min (pure Python loops) → the engine must be vectorised.

Published component shares for context (phase 3, US): administrative staff 20 %, clinical procedures 20 %, clinical staff 15 %, site monitoring 14 %, site retention 11 %, central lab 7 % (Sertkaya 2016 via BMC Health Serv Res 2024). On-site monitoring cited at 15–30 % of total costs in industry RCTs (Griessbach 2024).

---

## 8. Roadmap

1. Engine v0: vectorised NumPy Monte Carlo, YAML design loader, breakdowns, tornado, dropout-cost report, deterministic "all-at-mode" mode for testing.
2. Recruitment priors from ClinicalTrials.gov (enrolment ÷ (sites × recruitment duration) by indication/phase).
3. Fix known gaps: Italy, BfArM fee, possible double counting site management vs monitoring.
4. One-hour review of defaults by an operational professional (CRA or clinical project manager): CRO rates, monitoring frequency, investigator fees.
5. Validation: reconstruct per-phase benchmarks from typical designs.
6. Streamlit UI (quick mode), then stlite static page.
7. v1: technical feasibility (power), academic sponsor parameter, inflation indexing, advanced mode.

---

## 9. Sources

Accessed 2026-11-18. Grade in brackets = how the source is used.

### Official tariffs and regulations
- [A] Arrêté du 29 juillet 2024 fixant le modèle de convention unique, annexe coûts et surcoûts (JORF 03/08/2024) — https://www.fhf.fr/sites/default/files/2024-08/joe_20240803_0184_0004.pdf
- [context] FHF, convention unique: publication de l'arrêté du 29/07/2024 — https://www.fhf.fr/expertises/finances/budget-eprd/convention-unique-publication-de-larrete-du-29/07/2024-et-note-dinformation-relative-son-utilisation
- [context] Instruction DGOS/PF4 n° 2014-195 (contrat unique, grille de surcoûts) — https://solidarites-sante.gouv.fr/fichiers/bo/2014/14-09/ste_20140009_0000_0030.pdf
- [A] KPBV Anlage 3, German ethics committee fees — https://www.buzer.de/gesetz/12670/b29590.htm
- [A] AEMPS, relación de tasas — https://www.aemps.gob.es/la-aemps/tasas/relaciontasas/?lang=ca
- [A] Parc de Salut Mar CEIm fees (AEMPS single fee amounts) — https://researchmar.net/media/upload//arxius/tarifes/Tasas%20CEIm-PSMAR_cast.pdf
- [A] Citeline disclosure roundup, July 2026 (Dutch CTA fees) — https://www.citeline.com/en/disclosure-monthly-roundups/jul-2026
- [context] CCMO, Dutch review fees — https://english.ccmo.nl/investigators/fees-for-reviewing-research-with-a-medicinal-product-and-research-with-a-medical-device
- [A] ECRIN, CTCG fees for clinical trials under the CTR (France: no fees) — https://icss.ecrin.org/en/node/423
- [not accessed] HMA/CTCG consolidated fee table (blocked from this environment) — https://www.hma.eu/fileadmin/dateien/HMA_joint/00-_About_HMA/03-Working_Groups/CTCG/2025-1-Fees-for-clinical-trials-submitted-under_CTR.pdf

### French imaging tariffs (CCAM)
- [C] Groupama, prix d'une IRM (forfait technique 28.04–203.83 €, honoraires 69 €) — https://www.groupama.fr/mutuelle-sante/remboursement/prix-irm/
- [C] Selectra, remboursement IRM — https://selectra.info/assurance/mutuelle-sante/remboursement/irm

### Cost benchmarks and literature
- [benchmark] ASPE (2024), Examination of clinical trial costs and barriers for drug development (Table 6) — https://aspe.hhs.gov/sites/default/files/documents/ba3c3c99877862bac0b1d04460ad01b1/drug-development.pdf
- [context] ASPE/ERG (2014) report page — https://aspe.hhs.gov/report/examination-clinical-trial-costs-and-barriers-drug-development
- [context] Sertkaya et al. (2016), Key cost drivers of pharmaceutical clinical trials in the US, Clin Trials — https://pubmed.ncbi.nlm.nih.gov/26908540/
- [context] BMC Health Services Research (2024), per-patient phase III costs and ICER (component shares) — https://link.springer.com/article/10.1186/s12913-024-11638-0
- [context] Griessbach et al. (2024), Resource use and costs of investigator-sponsored RCTs, J Clin Epidemiol — https://real.mtak.hu/249106/1/Griessbach_Schandelmaier_J%20of%20Clin%20Epid_2024.pdf
- [benchmark] Moore et al. (2018), JAMA Intern Med, via RAPS — https://www.raps.org/resource/phase-3-trial-costs-estimated-at-19m-study-finds.html
- [benchmark] Moore et al. (2018), via AJMC — https://www.ajmc.com/focus-of-the-week/more-than-100fold-cost-differences-found-for-pivotal-trials
- [benchmark] Moore et al. (2018), via Global Genes (orphan trials) — https://globalgenes.org/blog/ferris-bueller-and-the-cost-of-pivotal-clinical-trials/
- [benchmark] Nature Reviews Drug Discovery news (2018), citing Martin et al. 2017 median phase costs — https://www.nature.com/articles/nrd.2018.197.pdf
- [benchmark] Clinical Leader (2026), per-patient cost analytics (Tufts cost per day) — https://www.clinicalleader.com/doc/clinical-trial-per-patient-cost-analytics-for-procurement-and-sponsor-strategy-0001
- [C] Lille ILIS master's thesis (2021), costs of academic French trials (insurance, DM, statistics) — https://pepite-depot.univ-lille.fr/LIBRE/Mem_ILIS/2021/LILU_SMIS_2021_022.pdf

### Operational cost aggregators (secondary)
- [B] IntuitionLabs, Clinical trial start-up costs (Tufts, ClinicalTrialZone, PharmExec figures) — https://www.intuitionlabs.ai/articles/clinical-trial-start-up-costs
- [B] IntuitionLabs, EDC software pricing 2026 (ClinCapture ranges) — https://intuitionlabs.ai/articles/edc-software-pricing-2026
- [B] IntuitionLabs, Clinical trial site payment benchmarks 2026 (overhead 31–35 %) — https://intuitionlabs.ai/articles/clinical-trial-site-payment-benchmarks
- [B] MedDeviceGuide, Imaging core labs: budget breakdown — https://meddeviceguide.com/blog/imaging-core-lab-medical-device-clinical-trials-guide
- [context] CASRAI, Clinical trial budget example (budget structure) — https://casrai.org/guides/clinical-trial-budget-example

### US research procedure prices (anchors for US index)
- [C] Penn CAMRIS MRI rates FY27 — https://www.med.upenn.edu/camris/rates.html
- [C] Baylor College of Medicine CAMRI pricing FY27 — https://www.bcm.edu/sites/default/files/pricing-sheet-fy27_camri.pdf
- [C] UCSD radiology research core services — https://radiology.ucsd.edu/research/core-services/index.html
- [context] ACR, CY2026 MPFS final rule impact tables (CPT 70551) — https://edge.sitecorecloud.io/americancoldf5f-acrorgf92a-productioncb02-3650/media/ACR/Files/Medicare-Medicaid/MPFS/CY2026-MPFS-Final-Rule-Impact-Tables-7xxxx-series.pdf
- [context] URMC Clinical Research Center fee schedule 2024–25 — https://urmc.rochester.edu/MediaLibraries/URMCMedia/ctsi/clinical-research/documents/CRCFees-1July2024-to-30June2025-Updated-10022024-Final.pdf

### Salaries and labour costs (inputs to derived rates and indices)
- [C] ZipRecruiter, CRA salary (US) — https://www.ziprecruiter.com/Salaries/Cra-Clinical-Research-Associate-Salary
- [C] Salary.com, Clinical project manager (US) — https://www.salary.com/research/salary/listing/clinical-project-manager-hourly-wages
- [context] Freelance Clinical, CRO vs clinical contractors (agency markup 25–40 %) — https://freelanceclinical.com/cro-vs-clinical-contractors
- [A] INSEE, Salaires et coût du travail en Europe 2025 (FR 44.7 €/h, DE 45.7 €/h) — https://www.insee.fr/fr/statistiques/8733109
- [A] Bolsamanía, citing Eurostat 2025 (ES 26.4 €/h) — https://www.bolsamania.com/noticias/economia/economia--los-costes-laborales-en-espana-suben-un-35-en-2025-pero-siguen-lejos-de-la-media-de-la-ue--22217718.html
- [A] BTA, citing Eurostat 2025 (NL 47.9 €/h) — https://www.bta.bg/en/news/1096069-bulgaria-has-lowest-labour-costs-in-eu-in-2025-but-highest-annual-increase
- [context] Eurostat news release, hourly labour costs 2025 — https://ec.europa.eu/eurostat/product?code=ddn-20260331-2

### Exchange rate
- [A] Wise, EUR to USD (≈1.16, Nov 2026) — https://wise.com/in/currency-converter/eur-to-usd-rate

### Searched but not usable
- Apices CRO, clinical trial cost in Europe (fetch failed) — https://apicescro.com/clinical-trial-cost-europe/
- EPM Scientific, Clinical operations rates guide Europe (gated) — https://www.epmscientific.com/en-de/industry-insights/compensation-guides/clinical-operations-rates-guide-europe

---

## 10. Work log

Newest first. Never edit past entries; add corrections as new entries.

### 2026-11-18 — Engine step 1: repository and package scaffolding
- Done: `git init`; uv library package `trialcost` (`pyproject.toml`, `uv.lock`, `.python-version` = 3.13, `requires-python >= 3.11`); runtime deps pydantic, numpy, pandas, pyyaml; dev deps pytest, ruff; ruff + pytest config; `.gitignore`; `src/trialcost/__init__.py` (`DATA_DIR`); `tests/test_smoke.py`; empty `designs/`.
- Decisions: none (tooling only). Ruff excludes `*.md` and `prototype/` so it never reformats the agreed schema in §3 or the throwaway prototype.
- Assumptions: none.
- Results: no change (no engine yet).
- Next: step 2 — `distributions.py`, `design.py`, archetype YAML. Open questions still pending: override file format, site_management vs monitoring_visit double counting, tornado parameters.

### 2026-11-18 — Design phase and v0 catalogue (chat with Claude)
- Done: brainstorm and design decisions (§5), design schema (§3), v0 catalogue in `data/` (4 CSVs), prototype Monte Carlo (`prototype/sanity_check.py`), README and CLAUDE.md.
- Decisions: D1–D18.
- Assumptions: §6 items 1–14 created.
- Results: archetype P50 €4.71M (P10 4.15, P90 5.74); duration P50 38 months; ~45 % of cost time-driven.
- Next: vectorised engine v0, recruitment priors from ClinicalTrials.gov, operational review of CRO rates.
