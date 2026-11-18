# Clinical Trial Cost Estimator

A design-driven, probabilistic estimator of the **total sponsor budget** of a clinical trial.
You describe a design (population, visit schedule, procedures, countries, sites, operational options); the tool returns a **cost distribution** (P10 / P50 / P90), an **estimated duration distribution**, and a **breakdown** by cost category, resource type and country.

> Status (2026-11-18): design phase complete, v0 cost catalogue built, throwaway sanity check run, package scaffolding, distributions, design schema, catalogue loader, recruitment, quantities and costing in place (engine step 5/6); no CLI / report yet. **No engine yet.**
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
  - `rate_mean` is drawn **once per simulated trial** and shared by its sites (uncertainty about the indication moves all sites together); `rate_shape` sets the between-site spread (CV = 1/√shape).
  - **Exact computation, no time grid**: expected randomised by month t is M(t) = Σᵢ λᵢ·(t − aᵢ)⁺ (piecewise linear). By the time-change theorem, T_LPI = M⁻¹(S) with S ~ Gamma(N, 1); M⁻¹ is solved segment by segment. Checked against a brute-force simulation of the arrival times in the tests.
  - **Patients per site** = exact conditional expectation given T_LPI: the first N − 1 patients split in proportion to λᵢ·(T_LPI − aᵢ)⁺, the N-th in proportion to λᵢ of sites open at T_LPI. Fractional counts, each simulation sums to N (costs are linear in patients, so integer draws would only add noise).
  - Deterministic mode: activation at mode, all λᵢ = median rate_mean, S = N, so M(T_LPI) = N (archetype: 18.93 months by hand).
  - No upper cap on recruitment duration (the prototype stopped at 120 months): long tails are reported as they are.
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
- Other countries via a **site price index** (triangular) and a **site overhead rate**; a procedure can carry a country-specific price in `procedure_overrides.csv` (e.g. US research MRI), which replaces *FR price × site index*; the country overhead is added on top only if `apply_overhead` is true (D25).
- The catalogue contains ~20 generic procedures (v0); finer granularity (e.g. MRI with/without contrast) later.

### 2.8 Quantities and costing (implementation, `quantities.py`, `costing.py`)

What each `study_items.csv` driver multiplies the unit price by, per simulation:

| Driver | Quantity | Level |
|---|---|---|
| `per_study` | 1 | study |
| `per_month` | T_total (study start → CSR) | study |
| `per_meeting` | DMC meetings = ⌈(T_LPLV − first activation) / interval⌉ (D27) | study |
| `per_country` | 1 per country | country |
| `per_site` | sites in the country (all planned sites, recruiting or not) | country |
| `per_site_month` | Σ sites max(0, T_LPLV − activation) | country |
| `per_site_year` | site-months / 12 | country |
| `per_patient` | randomised patients | country |
| `per_patient_visit` | expected patient-visits (screening visits on screened patients) | country |
| `per_scan` | occurrences of `imaging`-category procedures (dropout-adjusted) | country |
| `per_monitoring_visit` | site-months / monitoring interval | country |
| `per_rater` | raters per site × sites | country |

- Procedures: expected occurrences = Σ visits (patients attending × times listed), plus the automatic on-site procedures once per on-site visit.
- Unit price in country c: override → override (× (1 + overhead) if `apply_overhead`); `site_price` → FR price × index_c × (1 + overhead_c); `none` → FR price. Same `country_scaling` rule for study items (D23). CTA fees per country from `countries.csv`.
- Draw sharing: country index and overhead drawn once per simulation and shared by all lines of that country; each catalogue price drawn once per simulation and shared across countries; all catalogue rows drawn in file order on every run (common random numbers across scenarios, D29).
- Output: `CostLines` = metadata table (item, source, country, category, resource, driver) + `(n_lines, n_sims)` cost array; any breakdown is a groupby on the metadata; `to_frame()` gives the tidy long table (D28).

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
    activation_months: Dist | None = None   # None -> data/countries.csv (D21, provisional)

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
| `procedure_overrides.csv` | Country-specific procedure prices replacing *FR price × site index* (1 row: US brain MRI) |
| `constants.csv` | 4 engine-level parameters (FX rate, close-out months, raters per site, DMC meeting interval), same `low/mode/high` + grade columns |

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
| D19 | Deterministic mode uses each distribution's **stated parameter**: Fixed → value, Triangular → mode, LogNormal → median, Beta → mean | Golden-test totals stay hand-checkable from the YAML; a Beta mode would depend on κ |
| D20 | Design validation adds consistency checks **without changing the schema**: unique arm / visit / country IDs, `screening` ⇔ `day < 0`, ≥ 1 arm, country and post-randomisation visit | Catch typos early; procedure IDs are checked against the catalogue (`catalogue.py`) |
| D21 | **Provisional — revisit.** `CountryAllocation.activation_months` is **optional**: when omitted, the country's range in `countries.csv` is used; a value in the design overrides it | Avoids duplicating catalogue values in every design. To revisit once recruitment priors exist (activation may become indication- or site-type-specific, and the design may need to carry it explicitly) |
| D22 | Engine-level numbers (FX, close-out, raters per site, DMC interval) live in `data/constants.csv` with the usual grade columns | "No numbers in code" applies to these too |
| D23 | `study_items.csv` gets a **`country_scaling`** column (`site_price` / `none`), like `procedures.csv`. `site_price` for site-paid items: `pharmacy_annual`, `site_monitoring_fee`, `investigator_fee` (notes said "country-scaled") and the FR-only `site_fixed_fees_fr`, `imaging_site_setup_fr` (no numeric effect: FR index 1, overhead 0). All CRO / vendor items `none` | Previously implicit in notes and prototype code; agreed with Vincent |
| D24 | Recruitment computed **exactly** (time-change inversion, §2.4) with **fractional expected patients per site** and **no duration cap** | Exact, vectorised (10k sims in ~10 ms), no grid bias; costs are linear in patients |
| D25 | Country-specific procedure prices in **`procedure_overrides.csv`** (`procedure_id, country, low, mode, high, apply_overhead, evidence, source, notes`). An override is a local EUR price: never multiplied by the site index; overhead added only if `apply_overhead`. **US brain MRI = €687 / 1,000 / 1,261, `apply_overhead = false`** | Vincent delegated the choice. Reasoning: (1) the US site index was itself calibrated on MRI, so *FR × index* would reuse the same evidence with extra noise: a direct US price is stronger; (2) per scan = 1 h research slot + local read, matching the FR definition; low = Baylor CAMRI FY27 single tier ($790 all-in), mode = Penn CAMRIS FY27 industry tier ($1,150 incl. read), high = Penn outside-entity tier ($1,450); (3) no overhead because industry/outside tiers are full-cost-recovery rates (Penn industry = 1.31 × government tier, ≈ US overhead 31–35 %), so adding it would double count. Inference, grade C. Impact small (~€9k on the archetype either way) |
| D26 | **Keep both `site_management` and `monitoring_visit`**: `site_management` = remote / in-house CRA work per active site-month (CTA, TMF, site contact), `monitoring_visit` = on-site visits | Item definitions do not overlap on paper; possible overlap in real CRO quotes stays flagged in §6 #3 for the operational review (roadmap #4) |
| D27 | **DMC meetings over the conduct window = first site activation → LPLV**; number of meetings = ⌈window / `dmc_meeting_interval_months`⌉ | DMCs review accrued data during conduct, not only during one patient's treatment period. First activation slightly precedes first patient in (archetype: by ~0.4 month on average), well under one meeting; ceiling counts the partial final period as a meeting |
| D28 | Cost lines stored as **metadata table + `(n_lines, n_sims)` array**, tidy long table on demand | Breakdowns stay a groupby while 10k sims × ~200 lines remain fast (full pipeline ≈ 45 ms) |
| D29 | **Common random numbers**: every catalogue row drawn on every run in file order; country factors shared within a country, prices shared across countries | Toggling an option changes only the lines it affects (tested), needed for the Δ-vs-no-dropout report and for a clean tornado |
| D30 | **Labels reworded** to match their drivers (`edc_hosting`: whole duration; `pharmacy_annual`: every active site-year) rather than adding offsets; `resource` corrected for software / fee items (`edc_build_licence`, `edc_hosting` → `it_vendor`; `irb_local` → `authority_fee`) | Vincent's choice; small overlaps documented in §6 #16. Resource fix keeps the resource breakdown honest and keeps software / fees out of the sponsor-model labour factors |

**Provisional decisions to revisit:** D21 (optional `activation_months`).

---

## 6. Key assumptions (to review)

Ranked by expected impact. All are in the CSVs with grade C or D.

1. **CRO bill-rate multiplier 2.5–3.5 × salary hourly rate** (feeds `monitoring_visit`, `project_management`). No public CRO rate card found. Base salaries: CRA ~$85.5k/y (ZipRecruiter), clinical PM ~$117k/y (Salary.com). For comparison, a staffing-agency markup on contractor pay is quoted at 25–40 % (Freelance Clinical); CRO rates are assumed much higher because they include overhead, utilisation and margin. **Most important parameter to validate.**
2. **Monitoring visit** = ~12 CRA hours (on-site day + prep + report) + travel €300–900; interval 2 months per active site.
3. **Site management** €870–2,174 per active site-month ($1,000–2,500; Tufts/PharmExec ~$1,500–2,000). Kept alongside monitoring visits (D26: remote vs on-site work); real CRO quotes may overlap: to check in the operational review.
4. **Project management** 40–120 h/month.
5. **Site price indices** vs France: DE 1.1–1.6, ES 0.7–1.0, IT 0.8–1.1, NL 1.0–1.5, US 2.5–4.0. US anchored on research MRI (€900–1,300 vs ~€300 in FR). Labour costs are informative, not determinant.
6. **Site overhead**: FR 0 (convention unique includes management fees), EU 10–25 %, US 25–35 % (published institutional rates 31–35 %).
7. **Activation delays**: 3–12 months by country (Tufts: ~6 months existing sites, ~8.4 months new sites).
8. **Close-out**: 6 months from LPLV to CSR.
9. **Investigator fee / contreparties**: €500–3,000 per patient (country-scaled).
10. **Raters**: 2 per site; **DMC**: one meeting per 6 months of conduct, conduct = first site activation → LPLV (`constants.csv`, D27). The `dmc_meeting` note in `study_items.csv` ("of treatment period") predates D27.
11. **US brain MRI**: €687 / 1,000 / 1,261 per scan, no overhead added (`procedure_overrides.csv`, D25). Assumes one research scan = 1 h scanner slot + $100 local read; the "no overhead" choice is inferred from Penn's tiered rates, not stated by the source.
12. **German BfArM fee** not found: CTA range = ethics fees + assumed €3–10k.
13. **Italy**: labour cost, fees and site index not verified.
14. **Excluded**: IMP manufacturing, sponsor internal staff, VAT, inflation indexing, explicit CRO margin (embedded in rates).
15. **Beta concentration κ = 20** for screen-failure and dropout rates in the archetype (SD ≈ 0.10 and 0.08), inherited from the prototype. Placeholder until ClinicalTrials.gov priors.
16. **Driver vs label overlaps (accepted, D30)**: `edc_hosting` is charged over the whole study, so it may overlap the year-1 licence in `edc_build_licence` (≤ ~€18k at mode); `pharmacy_annual` counts every active site-year, so FR year 1 overlaps `site_fixed_fees_fr` (≤ ~€500 per FR site). Labels reworded to say what is counted.
17. **Operations fields not costed yet**: `sponsor_model`, `monitoring` (full SDV vs risk-based) and `n_interim_analyses` are accepted in the design but no catalogue item depends on them in v0.
18. **All planned sites cost start-up and close-out**, even when a simulated site recruits nobody (late activation); site-months still stop at LPLV.

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
3. Fix known gaps: Italy, BfArM fee; check site management vs monitoring overlap against real CRO quotes (D26).
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
- [C] Penn CAMRIS MRI rates FY27 (government $800/h, industry $1,050/h, outside entity $1,350/h; re-read 2026-11-18) — https://www.med.upenn.edu/camris/rates.html
- [C] Baylor College of Medicine CAMRI pricing FY27 ($600/h, technologist $90, limited radiologist read $100; re-read 2026-11-18) — https://www.bcm.edu/sites/default/files/pricing-sheet-fy27_camri.pdf
- [not used] UCSD radiology research core services (no rates on the page when re-read 2026-11-18; not used for D25) — https://radiology.ucsd.edu/research/core-services/index.html
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

### 2026-11-18 — Engine step 5: quantities and costing
- Done: `src/trialcost/quantities.py` (SF, dropout hazard and visit attendance, automatic on-site procedures, procedure counts as visit × incidence matrix product, scans, site-months, monitoring visits, DMC meetings, `Quantities.driver()`); `src/trialcost/costing.py` (country factors, procedure / override / study-item / CTA pricing, `CostLines` with `total()`, `by()`, `select()`, `to_frame()`); `catalogue.py` rejects `site_price` on study-level items; `tests/conftest.py` (shared pipeline helper), `tests/test_quantities.py`, `tests/test_costing.py` (golden deterministic total, hand-checked lines, breakdown consistency, invariants, common-random-numbers check, timing, P50 regression band). 132 tests pass. Methodology written up in §2.8.
- Decisions: D28 (cost-line storage), D29 (common random numbers).
- Assumptions: §6 #16 (driver/label mismatches for `edc_hosting`, `pharmacy_annual`), #17 (operations fields not costed), #18 (all planned sites pay start-up).
- Results: deterministic (all at mode/median) total **€4,182,180**, T_total 37.8 months. 10k sims (seed 42): total P10/P50/P90 **€4.20M / 4.81M / 5.85M** (prototype 4.15 / 4.71 / 5.74), duration 31.6 / 38.4 / 51.7 months. Largest categories at P50: monitoring €1.13M, management €0.70M, site start-up €0.55M, imaging €0.31M. Full pipeline 10k sims ≈ 45 ms.
- Next: step 6, `simulate.py` CLI + `report.py` (quantiles, breakdowns, dropout-cost report, tornado). Decide the two label mismatches (§6 #16).

### 2026-11-18 — Pre-costing decisions: US MRI price, site management, DMC window
- Done: new `data/procedure_overrides.csv` (1 row: US brain MRI); `catalogue.py` loads and validates it (known procedure and country, unique pair, range, grade, boolean `apply_overhead`) and exposes `procedure_override(procedure_id, country)`; 5 new tests (108 pass). Penn and Baylor sources re-read; the UCSD page has no rates and is now marked not used.
- Decisions: D25 (override format; US MRI €687 / 1,000 / 1,261, no overhead; full reasoning in §5), D26 (keep `site_management` and `monitoring_visit`), D27 (DMC window = first activation → LPLV, ⌈window / 6⌉ meetings).
- Assumptions: §6 #3 reworded to match the CSV (€870–2,174 = $1,000–2,500; the old text gave the dollar range with a € sign), #10 and #11 updated.
- Corrections to earlier entries: none. The US MRI range moves from €900–1,300 (mode 1,100, prototype code) to €687–1,261 (mode 1,000), without overhead in either case.
- Results: no cost output yet (expected impact on the archetype ≈ −€3k).
- Next: step 5, `quantities.py` + `costing.py`.

### 2026-11-18 — Engine step 4: recruitment model (+ D23 catalogue column)
- Done: `data/study_items.csv` gains `country_scaling` (D23, separate commit; validated in `catalogue.py`; no other value changed). `src/trialcost/recruitment.py`: per-country site activation (design or `countries.csv`), Poisson–gamma site rates with `rate_mean` shared within a simulated trial, exact T_LPI via `time_to_intensity()` (inverse of the piecewise-linear cumulative intensity at S ~ Gamma(N, 1)), expected patients per site; `Recruitment` result dataclass. `tests/test_recruitment.py` (15 tests: hand examples, brute-force Poisson comparison of T_LPI quantiles and per-site counts, deterministic archetype by hand, monotonicity, reproducibility, timing). 103 tests pass.
- Decisions: D23 (`country_scaling` in study items), D24 (exact recruitment computation, fractional patients, no cap). §2.4 updated.
- Assumptions: none new (`rate_mean` shared by sites within a trial, as in the prototype, now stated in §2.4).
- Results: archetype T_LPI P10/P50/P90 = 12.7 / 19.5 / 32.8 months (prototype implied 13.1 / 19.1 / 33.1); T_total = LPI + 12.88 + 6 → 31.6 / 38.4 / 51.7 (prototype 32 / 38 / 52). P(T_LPI > 60 months) ≈ 0.5 %. Deterministic T_LPI = 18.93 months. Runtime 10k sims ≈ 8 ms.
- Next: step 5, `quantities.py` + `costing.py`. Open before costing: override file format (US MRI); site_management vs monitoring_visit double counting; DMC exposure window (§6 #10).

### 2026-11-18 — Engine step 3: catalogue loader and validation
- Done: `src/trialcost/catalogue.py` (load the 5 CSVs; validate columns, unique IDs, `low <= mode <= high`, non-negative, grades incl. mixed `A/D`, non-empty sources, drivers, scalings, FR reference = index 1 / overhead 0, required constants; all problems reported together in `CatalogueError`; `applies_if` parser with explicit atom classes, no `eval`; `Catalogue` accessors returning `Triangular`; `applicable_items(design, country)`; `check_design()`); new `data/constants.csv`; `activation_months` made optional in `design.py`, removed from the archetype YAML; `tests/test_catalogue.py`. 86 tests pass.
- Decisions: D21 (optional `activation_months`, **provisional**, listed under "to revisit" in §5), D22 (`constants.csv`).
- Assumptions: no new values. Existing assumptions §6 #8 and #10 moved from prototype code into `constants.csv`; §6 #10 notes the DMC exposure-window mismatch.
- Results: no cost output yet. Archetype resolves to 36 study-level items, plus 2 France-only (`site_fixed_fees_fr`, `imaging_site_setup_fr`) and 1 US-only (`irb_local`).
- Next: step 4, `recruitment.py`. Open before step 5: (a) `study_items.csv` has no `country_scaling` column, but notes say `pharmacy_annual`, `site_monitoring_fee` and `investigator_fee` are country-scaled (the prototype scaled `investigator_fee` in code); (b) override file format; (c) site_management vs monitoring_visit double counting.

### 2026-11-18 — Engine step 2: distributions, design schema, archetype YAML
- Done: `src/trialcost/distributions.py` (Fixed / Triangular / LogNormal / Beta with vectorised `sample(rng, size)`, `central()`, `expectation()`, and `draw(..., deterministic=)` as the single sampling entry point); `src/trialcost/design.py` (§3 schema field for field, consistency checks, `n_sites`, `patient_duration_months`, `visits_every()`, `load_design()`); `designs/rare-neuro-ph2-mri-scales.yaml` (same design as the prototype); `tests/test_distributions.py`, `tests/test_design.py` (41 tests incl. one doctest). Pytest now also collects doctests from `src/`.
- Decisions: D19 (central values for deterministic mode), D20 (extra design validation, schema unchanged). Implementation details that do not change §3: `Operations.monitoring_interval_months` uses `validate_default=True` (otherwise the default 2.0 stays a bare float instead of `Fixed`); `Union[...]` written as `A | B | ...` (equivalent).
- Assumptions: §6 #15 (Beta κ = 20, placeholder).
- Results: no cost output yet. Archetype patient duration = 392 d = 12.88 months; all procedure IDs in the YAML exist in `procedures.csv`.
- Next: step 3 — `catalogue.py` (load + validate CSVs, `applies_if` parser). Open question: per-country `activation_months` is required in the schema and currently duplicates `countries.csv`; should it become optional with the catalogue as default?

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
