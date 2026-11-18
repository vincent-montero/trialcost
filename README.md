# Clinical Trial Cost Estimator

A design-driven, probabilistic estimator of the **total sponsor budget** of a clinical trial.
You describe a design (population, visit schedule, procedures, countries, sites, operational options); the tool returns a **cost distribution** (P10 / P50 / P90), an **estimated duration distribution**, and a **breakdown** by cost category, resource type and country.

> Status (2026-11-18): design phase complete, v0 cost catalogue built, throwaway sanity check run, package scaffolding, distributions, design schema, **engine v0 complete** (design → recruitment → quantities → costs → report, CLI) and **first benchmark validation done** (§7.2); no UI yet. **No engine yet.**
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
- **Dropout-cost report** (`report.dropout_report`): the design is simulated twice with the same seed, as given and with dropout = 0. Reported: completers = N·exp(−h·patient duration), cost per completer, cost per completer without dropout, total-cost change and the completer cost premium.
- **Tornado** (`report.tornado`, one-way sensitivity on the **mean** total cost, same seed for every run): each input goes to its P10 then P90 while all other inputs keep their uncertainty; fixed inputs go to 0.5× / 1.5×; groups (all countries' site index / overhead / activation, all CRO-labour prices) move together, members without spread unchanged. Inputs: recruitment rate, between-site spread, activation delays, screen failure, dropout, monitoring interval, site price index, site overhead, CRO labour prices, close-out duration.
- **Random streams**: every random input draws from its own child generator (`Generator.spawn`), so scenarios that pin or change one input share all other random numbers exactly (D32). That makes the dropout delta and tornado bars free of Monte Carlo noise from unrelated inputs.
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
| `per_conduct_month` | conduct window = T_LPLV − first activation (D27) | study |
| `per_interim_analysis` | `operations.n_interim_analyses` | study |

- Procedures: expected occurrences = Σ visits (patients attending × times listed), plus the automatic on-site procedures once per on-site visit.
- Unit price in country c: override → override (× (1 + overhead) if `apply_overhead`); `site_price` → FR price × index_c × (1 + overhead_c); `none` → FR price. Same `country_scaling` rule for study items (D23). CTA fees per country from `countries.csv`.
- Draw sharing: country index and overhead drawn once per simulation and shared by all lines of that country; each catalogue price drawn once per simulation and shared across countries; all catalogue rows drawn in file order on every run (common random numbers across scenarios, D29).
- Output: `CostLines` = metadata table (item, source, country, category, resource, driver) + `(n_lines, n_sims)` cost array; any breakdown is a groupby on the metadata; `to_frame()` gives the tidy long table (D28).

### 2.9 Operational options (D31)

| Option | What changes in the cost | Evidence |
|---|---|---|
| `monitoring: risk_based` (default) | `monitoring_visit` (targeted SDV, ~12 h CRA) + `central_monitoring` per conduct month (8/16/32 h) + `rbm_setup` (40/80/160 h) + `rbm_extra_queries` per patient (0.5/2.4/4 h) | Extra queries: MONITORING study (Fougerou-Leurent 2019) 2.4 h per participant; hours otherwise assumptions × CRA bill rate |
| `monitoring: full_sdv` | `monitoring_visit_full_sdv` instead: CRA time per visit × 1.17/1.35/1.49, travel unchanged | ADAMON (Brosteanu 2017): on-site time ratio / visit ratio, extensive vs risk-based |
| `monitoring_interval_months` | visit **frequency**, for both strategies; strategy does not change it | ADAMON: 1.8–3.5× more visits per participant with extensive on-site monitoring, so a full-SDV design should normally set a shorter interval |
| `sponsor_model: functional` / `in_house` | CRO-delivered labour items (`country_scaling = none`, resource in the factor file) × 0.65/0.80/0.95 (FSP) or 0.42/0.59/0.93 (in-house); vendors, fees, site payments and procedures unchanged | FSP: analyst / vendor claims only (D). In-house: fully burdened cost 1.25–1.40 × salary ÷ CRO multiplier 2.5–3.5 ÷ utilisation 0.60–0.85 (C) |
| `n_interim_analyses` | per interim: `interim_analysis` (data cut + independent statistician) and, if DMC, `interim_dmc_review` (extra meeting) | Fractions of `db_lock` / `biostatistics`, DMC meeting price (C/D); no public price found |

Deterministic impact on the archetype (base €4.249M, risk-based, full CRO): full SDV at the same 2-month interval +€43k (+1.0 %); full SDV monthly +€663k (+15.6 %, on-site monitoring ≈ 2.2× risk-based, vs 2.7–3.4× in OPTIMON); FSP −€509k (−12 %); in-house −€1.04M (−24.5 %); 2 interim analyses +€106k (+2.5 %).

`applies_if` also accepts `operations.<choice field>==<value>` (e.g. `operations.monitoring==full_sdv`), validated against the allowed values in the schema.

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
| `study_items.csv` | 45 study-, country-, site-, time- and patient-driven items |
| `countries.csv` | 6 countries: site price index, overhead, labour cost, CTA fees, activation delays |
| `benchmarks.csv` | 13 published benchmarks (costs and phase durations), used for validation only |
| `benchmark_checks.csv` | Which design and metric each benchmark is compared with, statistic, comparability (validation only) |
| `price_indices.csv` | US CPI-U Medical Care annual values, to inflate USD benchmarks to 2026 (validation only) |
| `sponsor_model_factors.csv` | Multipliers on CRO-delivered labour for `functional` (FSP) and `in_house` sponsor models (D31) |
| `procedure_overrides.csv` | Country-specific procedure prices replacing *FR price × site index* (1 row: US brain MRI) |
| `constants.csv` | 4 engine-level parameters (FX rate, close-out months, raters per site, DMC meeting interval), same `low/mode/high` + grade columns |

### Evidence grades

| Grade | Meaning | procedures | study_items |
|---|---|---|---|
| **A** sourced | Read directly from an official tariff or regulation | 12 | 4 |
| **B** secondary | Aggregator / planning guide citing industry data | 0 | 10 |
| **C** derived | Computed from sourced inputs + a stated assumption | 4 | 6 (+3 mixed C/D) |
| **D** assumption | Expert-judgement placeholder, to be reviewed | 4 | 22 |

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
| D31 | **Operational options now drive cost** (§2.9): monitoring strategy selects risk-based items (central monitoring, set-up, extra queries) or longer full-SDV visits, frequency stays `monitoring_interval_months`; sponsor model multiplies CRO-delivered labour via `sponsor_model_factors.csv`; interim analyses add analysis + DMC review items. New drivers `per_conduct_month`, `per_interim_analysis`; `applies_if` gains `operations.<field>==<value>` | Vincent asked for the options to be costed from public evidence. Frequency kept separate from strategy so the two effects are not counted twice. In-house counts sponsor staff (otherwise the work would vanish); full CRO / FSP exclude sponsor oversight (§1) |
| D32 | **One child random stream per random input** (`Generator.spawn` per stage, per country, per catalogue row), refining D29 | Pinning or removing one input (`Fixed`, point mass) consumes no random numbers; with a single shared stream that shifted every later draw. Now verified: pinning one procedure price changes only that procedure's lines |
| D33 | **Tornado on the mean total cost**, P10/P90 of uncertain inputs, 0.5×/1.5× of fixed ones, groups moved together (§2.6) | Mean is stable with common random numbers and 2,000 sims per bar; group swings answer "what if all our CRO rates are high", which is how the uncertainty is correlated in practice |
| D34 | No SciPy: the Beta quantile (tornado only) is computed in pure Python (incomplete beta continued fraction + bisection), tested against 1M samples | Adding the dependency was blocked in this session; the function is short and isolated, SciPy can replace it later |
| D35 | **Benchmark validation by reference designs**: each benchmark compared with a design resembling the trials it summarises (`benchmark_checks.csv`), USD inflated with the US CPI-U Medical Care index (`price_indices.csv`) then converted at the fixed FX; results reported in `reports/benchmark-validation.md`, never asserted | Follows D5 (benchmarks are mixtures of designs); ASPE inflated with the same index; CLAUDE.md: benchmarks are checked in a separate report |
| D36 | Reference designs live in `designs/ref-*.yaml`; they are **validation fixtures, not archetypes** | Keeps archetypes (user-facing starting points) separate from designs tuned to match published averages |

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
17. **Operational options (D31, §2.9)**: risk-based hours (central monitoring 8/16/32 h per conduct month, set-up 40/80/160 h) are assumptions; full-SDV visit length ratio 1.17–1.49 from ADAMON; FSP factor 0.65/0.80/0.95 rests on analyst/vendor claims only (weakest number of the set); in-house factor depends on §6 #1 (CRO multiplier) and an assumed utilisation 0.60–0.85; interim analysis = 40 % of `db_lock` + 20 % of `biostatistics` (assumed fractions); the site's own monitoring fee does not grow with longer full-SDV visits.
18. **All planned sites cost start-up and close-out**, even when a simulated site recruits nobody (late activation); site-months still stop at LPLV.
19. **Reference designs for validation** (`designs/ref-cns-ph2-typical.yaml`, `ref-cns-ph3-typical.yaml`): N and phase duration from ASPE 2024 Table 2; everything else assumed (1:1 placebo; 24 / 52 weeks of treatment; generic CNS assessments without imaging; 40 / 90 sites, US-weighted; SF 40 %; dropout 30 / 25 % per year; 0.30 / 0.25 patients per site-month).
20. **Benchmark price years**: Moore 2018 (no price year stated) taken as 2016 USD; Martin 2017 as 2015 USD; the 2026 index value is H1 2026 (latest available). A one-year shift changes the converted values by about 2–4 %.

---

## 7. Sanity check and validation

### 7.1 Prototype sanity check (throwaway, not validation)

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

### 7.2 Benchmark validation (engine v0, 2026-11-18)

Full table: `reports/benchmark-validation.md` (regenerate with `uv run python -m trialcost.validation --out reports`). Method (D35): each benchmark is compared with a **reference design resembling the trials it summarises**, not with the archetype; USD benchmarks inflated to 2026 with the US CPI-U Medical Care index (as ASPE did) and converted at 1.15. Reported, never asserted in tests.

| Benchmark (EUR 2026) | Compared with | Model | Ratio |
|---|---|---|---|
| ASPE CNS phase 2, mean per patient €51.8k | `ref-cns-ph2-typical` (243 pts) | mean €51.1k | **0.99** |
| ASPE all areas phase 2, mean per patient €62.3k | same (CNS proxy) | €51.1k | 0.82 |
| Martin phase II, median trial €9.9M | same | median €12.1M | 1.22 (poor match: median trial size unknown) |
| ASPE CNS phase 3, mean per patient €42.1k | `ref-cns-ph3-typical` (529 pts) | mean €53.8k | **1.28** |
| Moore pivotal, median trial €21.1M / per patient €45.7k / per patient-visit €4.0k | same (Moore median N = 488) | €27.7M / €52.4k / €5.0k | **1.31 / 1.15 / 1.27** |
| Moore pivotal ≤ 100 patients, mean trial €6.55M | archetype (50 pts) | mean €5.03M | **0.77** |
| ASPE phase durations: CNS ph2 40.1, CNS ph3 45.3 months | reference designs | 41.4 / 51.3 months | 1.03 / 1.13 |
| Tufts phase 2 direct cost per day €21.7k | `ref-cns-ph2-typical` | €10.2k | 0.47 (poor match: day base undefined) |

**Reading**
- **Mid-size phase 2: in range.** Per-patient cost matches ASPE CNS almost exactly; all-areas is higher because it includes costlier areas (haematology, oncology). Durations are consistent.
- **Large / long trials: model ~15–30 % high.** ASPE has phase 3 CNS **19 % cheaper per patient** than phase 2; the model has it 5 % dearer. The difference sits in time-driven costs: the 52-week reference keeps sites open 6.7 site-months per patient (4.9 in phase 2) and every site-month is charged at the same rate (site management, monitoring visits, central monitoring).
- **Small trials: model ~20–25 % low.** The archetype is below Moore's mean for pivotal trials with ≤ 100 patients (a mean, pulled up by expensive trials, so the gap may be smaller against a median).
- **Together**: the model looks **too flat across trial size**: it may under-cost fixed / small-trial overheads and over-cost time-driven site costs in long, large trials. Candidate causes, to check in the operational review (roadmap #4): monitoring frequency that falls during long maintenance phases (risk-based monitoring), CRO site-management rates per site-month, and study-level fixed costs (protocol, regulatory, data management set-up) for small trials.
- **Not conclusive**: benchmarks are averages over mixtures of designs, the reference designs are assumptions (§6 #19), and all benchmarks are US-centric.

---

## 8. Roadmap

1. ~~Engine v0: vectorised NumPy Monte Carlo, YAML design loader, breakdowns, tornado, dropout-cost report, deterministic "all-at-mode" mode for testing.~~ Done 2026-11-18.
2. Recruitment priors from ClinicalTrials.gov (enrolment ÷ (sites × recruitment duration) by indication/phase).
3. Fix known gaps: Italy, BfArM fee; check site management vs monitoring overlap against real CRO quotes (D26).
4. One-hour review of defaults by an operational professional (CRA or clinical project manager): CRO rates, monitoring frequency, investigator fees.
5. ~~Validation: reconstruct per-phase benchmarks from typical designs.~~ First pass done 2026-11-18 (§7.2). Follow-ups: per-site-month time-driven costs in long trials, fixed costs of small trials, more benchmarks (EU, rare disease, phase 2 medians with trial sizes).
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
- [benchmark] ASPE (2024), Examination of clinical trial costs and barriers for drug development (Table 6 per-patient costs, 2018 $; Table 2 patients per trial and phase durations; inflation with the Medical Care Price Index) — https://aspe.hhs.gov/sites/default/files/documents/ba3c3c99877862bac0b1d04460ad01b1/drug-development.pdf
- [context] ASPE/ERG (2014) report page — https://aspe.hhs.gov/report/examination-clinical-trial-costs-and-barriers-drug-development
- [context] Sertkaya et al. (2016), Key cost drivers of pharmaceutical clinical trials in the US, Clin Trials — https://pubmed.ncbi.nlm.nih.gov/26908540/
- [context] BMC Health Services Research (2024), per-patient phase III costs and ICER (component shares) — https://link.springer.com/article/10.1186/s12913-024-11638-0
- [context] Griessbach et al. (2024), Resource use and costs of investigator-sponsored RCTs, J Clin Epidemiol — https://real.mtak.hu/249106/1/Griessbach_Schandelmaier_J%20of%20Clin%20Epid_2024.pdf
- [benchmark] Moore et al. (2018), JAMA Intern Med, via RAPS — https://www.raps.org/resource/phase-3-trial-costs-estimated-at-19m-study-finds.html
- [benchmark] Moore et al. (2018), via AJMC (median N = 488; mean $5.9M for trials with ≤ 100 patients) — https://www.ajmc.com/focus-of-the-week/more-than-100fold-cost-differences-found-for-pivotal-trials
- [benchmark] Moore et al. (2018), via Global Genes (orphan trials) — https://globalgenes.org/blog/ferris-bueller-and-the-cost-of-pivotal-clinical-trials/
- [benchmark] Nature Reviews Drug Discovery news (2018), citing Martin et al. 2017 median phase costs — https://www.nature.com/articles/nrd.2018.197.pdf
- [not accessed] Martin L, Hutchens M, Hawkins C (2017), How much do clinical trials cost? Nat Rev Drug Discov 16:381–382 (paywalled; median trial sizes not obtained) — https://www.nature.com/articles/nrd.2017.70
- [A] US BLS CPI-U Medical Care, U.S. city average, semiannual (CUUS0000SAM), via FRED (accessed 2026-11-18) — https://fred.stlouisfed.org/series/CUUS0000SAM
- [benchmark] Clinical Leader (2026), per-patient cost analytics (Tufts cost per day) — https://www.clinicalleader.com/doc/clinical-trial-per-patient-cost-analytics-for-procurement-and-sponsor-strategy-0001
- [C] Lille ILIS master's thesis (2021), costs of academic French trials (insurance, DM, statistics) — https://pepite-depot.univ-lille.fr/LIBRE/Mem_ILIS/2021/LILU_SMIS_2021_022.pdf

### Monitoring strategies and operating models (D31)
- [C] Klatte K et al. (2021), Monitoring strategies for clinical intervention studies, Cochrane Database Syst Rev MR000051 (ADAMON, OPTIMON, MONITORING, START resource results) — https://discovery.ucl.ac.uk/id/eprint/10140800/1/Klatte_et_al-2021-Cochrane_Database_of_Systematic_Reviews.pdf
- [context] TransCelerate, Risk-based monitoring initiative — https://www.transceleratebiopharmainc.com/initiatives/risk-based-monitoring/
- [context] IntuitionLabs, FSP vs full-service CRO models (PPD FSP case study, operating costs down 20 %+) — https://intuitionlabs.ai/articles/fsp-vs-full-service-cro-model
- [context] Clinical Leader, The role of the CRO is changing (Deutsche Bank: 20–30 % first-year savings, FSP vs insourcing) — https://www.clinicalleader.com/doc/the-role-of-the-cro-is-changing-why-0001
- [context] Pharmaceutical Technology, sponsored interview (KPS: up to 40 % below full-service; vendor claim) — https://www.pharmaceutical-technology.com/sponsored/outsourcing-trends-in-clinical-trials-how-one-company-is-reimagining-the-fsp-model/
- [C] HiBob, Fully burdened labor rate (1.25–1.4 × salary rule of thumb) — https://www.hibob.com/financial-metrics/fully-burdened-labor-rate/
- [C] Payoneer, Labor burden guide 2026 — https://www.payoneer.com/resources/workforce-management/hr-and-compliance/labor-burden/
- [context] FDA (2024, draft guidance), Use of Data Monitoring Committees in Clinical Trials (interim comparative data restricted to the DMC and the statisticians producing the analysis; basis for the independent-statistician component of `interim_analysis`) — https://www.fda.gov/media/176107/download
- [searched, nothing usable] Public price of an interim analysis / unscheduled DMC review (Nov 2026)

### Operational cost aggregators (secondary)
- [B] IntuitionLabs, Clinical trial start-up costs (Tufts, ClinicalTrialZone, PharmExec figures) — https://www.intuitionlabs.ai/articles/clinical-trial-start-up-costs
- [B] IntuitionLabs, EDC software pricing 2026 (ClinCapture ranges) — https://intuitionlabs.ai/articles/edc-software-pricing-2026
- [B] IntuitionLabs, Clinical trial site payment benchmarks 2026 (overhead 31–35 %) — https://intuitionlabs.ai/articles/clinical-trial-site-payment-benchmarks
- [B] MedDeviceGuide, Imaging core labs: budget breakdown — https://meddeviceguide.com/blog/imaging-core-lab-medical-device-clinical-trials-guide
- [context] CASRAI, Clinical trial budget example (budget structure) — https://casrai.org/guides/clinical-trial-budget-example

### US research procedure prices (US brain MRI price, D25; anchors for the US site index)
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
- Cochrane Library page of Klatte 2021 (HTTP 403 from this environment; the UCL repository copy above was used instead) — https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.MR000051.pub2/full
- Journot 2017 OPTIMON primary paper (not located; OPTIMON figures taken from the Cochrane review)
- Apices CRO, clinical trial cost in Europe (fetch failed) — https://apicescro.com/clinical-trial-cost-europe/
- EPM Scientific, Clinical operations rates guide Europe (gated) — https://www.epmscientific.com/en-de/industry-insights/compensation-guides/clinical-operations-rates-guide-europe

---

## 10. Work log

Newest first. Never edit past entries; add corrections as new entries.

### 2026-11-18 — Benchmark validation report (roadmap #5, first pass)
- Done: `src/trialcost/validation.py` (CLI `python -m trialcost.validation`), `reports/benchmark-validation.md` / `.csv`; reference designs `designs/ref-cns-ph2-typical.yaml`, `ref-cns-ph3-typical.yaml`; `data/benchmark_checks.csv`, `data/price_indices.csv`; 4 benchmark rows added (Moore ≤ 100 patients mean, 3 ASPE phase durations); `tests/test_validation.py` (runs and converts correctly; no agreement asserted). 175 tests pass. Re-read ASPE 2024 (Table 2 / Table 6) and Moore 2018 (AJMC); medical-care CPI from FRED.
- Decisions: D35 (validation by reference designs, CPI medical care), D36 (reference designs are fixtures).
- Assumptions: §6 #19 (reference designs), #20 (benchmark price years).
- Correction: while writing the checks I first described Moore's sample as "~1/3 uncontrolled or active-controlled"; the source gives no proportions, so the note was corrected before commit.
- Results: phase 2 per patient matches ASPE CNS (ratio 0.99); large / long trials 15–31 % above ASPE phase 3 and Moore; archetype 23 % below Moore's mean for trials ≤ 100 patients; durations consistent (1.03 / 1.13). Reading: the model is probably too flat across trial size (time-driven site costs high in long trials, fixed costs low in small ones). Details §7.2.
- Next: take the §7.2 hypotheses into the operational review (roadmap #4); recruitment priors (#2).

### 2026-11-18 — Sources completed in §9
- Done: §9 gains the FDA DMC guidance (cited in `interim_analysis` notes), a clearer heading for the Penn / Baylor MRI sources (D25), and the Cochrane Library page and OPTIMON primary paper under "Searched but not usable". All D25 and D31 sources were already listed (commit 5686052).
- Decisions: none. Assumptions: none. Results: unchanged.
- Next: unchanged (see previous entry).

### 2026-11-18 — Engine step 6: CLI and reports (engine v0 complete)
- Done: `src/trialcost/simulate.py` (`run()` → `SimulationResult`, CLI with `--n-sims`, `--seed`, `--deterministic`, `--budget`, `--recruit-by`, `--tornado-sims`, `--no-tornado`, `--out`, `--lines`, `--data-dir`); `src/trialcost/report.py` (summary, probabilities, breakdowns with shares, dropout-cost report, tornado, text report, CSV outputs); `quantile()` / `is_point_mass` on every distribution; one child random stream per random input in recruitment, quantities and costing; ruff ignores PD011 (false positives on `CostLines.values`); `tests/test_simulate_report.py` + quantile tests (168 pass). P50 regression band moved deliberately from 4.6–5.0 to 4.7–5.1 M€ after D31.
- Decisions: D32 (child random streams), D33 (tornado definition), D34 (pure-Python Beta quantile).
- Assumptions: none new.
- Results (10k sims, seed 42, archetype): total P10/P50/P90 **€4.27M / 4.89M / 5.96M**; duration 31.6 / 38.5 / 51.8 months; LPI 12.7 / 19.6 / 32.9 months; P50 per randomised patient €98k; P(cost ≤ €5M) 57 %; P(LPI ≤ 24 months) 70 %. Largest shares: monitoring 25.8 %, management 14.7 %, site start-up 11.3 %; CRA labour alone 35.8 %. Dropout: 42.6 completers of 50 (P50), cost per completer €117k vs €99k without dropout (+16.6 %), total cost −€31k (−0.6 %). Tornado (mean €5.04M): CRO labour prices €1.89M swing, recruitment rate €1.36M, monitoring interval (1 vs 3 months) €0.81M, everything else < €0.16M. CLI run incl. dropout comparison and tornado: 0.7 s.
- Next: recruitment priors from ClinicalTrials.gov (roadmap #2), operational review of CRO rates (#4, now confirmed as the top driver), Streamlit UI (#6). Open question: strategy-specific default for `monitoring_interval_months`.

### 2026-11-18 — Label rewording (D30) and costed operational options (D31)
- Done: `study_items.csv`: `edc_hosting` and `pharmacy_annual` labels reworded, `resource` fixed for EDC (→ `it_vendor`) and `irb_local` (→ `authority_fee`); 6 new items (`monitoring_visit_full_sdv`, `central_monitoring`, `rbm_setup`, `rbm_extra_queries`, `interim_analysis`, `interim_dmc_review`), `monitoring_visit` now conditional on risk-based monitoring; new `data/sponsor_model_factors.csv`. Code: `applies_if` supports `operations.<field>==<value>`; drivers `per_conduct_month`, `per_interim_analysis`; sponsor-model factor in `costing.py`; study-level lines with zero quantity dropped. 14 new tests (146 pass). Literature search: Cochrane review of monitoring strategies (ADAMON, OPTIMON, MONITORING), FSP / insourcing claims, burdened labour cost; nothing public on interim-analysis prices.
- Decisions: D30, D31.
- Assumptions: §6 #16 and #17 rewritten.
- Results: deterministic archetype €4,182,180 → **€4,249,359** (+€67k: central monitoring, risk-based set-up and extra queries now costed). Option impacts in §2.9.
- Next: step 6. Question for Vincent: should `monitoring_interval_months` default to a strategy-specific value (e.g. 1 month for full SDV) instead of 2.0 for both? That is a schema change.

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
