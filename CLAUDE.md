# CLAUDE.md — Clinical Trial Cost Estimator

Read `README.md` first: goals, methodology, design schema, decision log, assumptions and sources live there. This file tells you how to work on the code.

## Project in one paragraph

Design-driven Monte Carlo estimator of a clinical trial's **total sponsor budget**. Input: a trial design (YAML, validated by Pydantic). Output: distributions of cost and duration (P10/P50/P90), breakdowns by category / resource / country, tornado chart, dropout-cost report. v0 = cost + time only; N is an input (no power calculation).

## Hard constraints

- **Personal project. Public data only.** Never add numbers from Thelonius Mind, OrphanDev/F-CRIN or any client/employer budget, even if they appear in local files.
- **No prices in code.** Every unit cost, rate, index or fee lives in `data/*.csv` with `evidence` (A/B/C/D), `source` and `notes`.
- **Never invent a number silently.** A new value without a source is added as grade `D`, `source = "Assumption"`, with the rationale in `notes`. A derived value is grade `C` with the derivation in `source`. Then add it to README §6 if it may matter.
- **Keep README in sync**: any change to methodology → README §2; any decision → §5 (decision log); any new assumption → §6; any new source → §9 with URL.
- Do not change the design schema (README §3) or the CSV column sets without asking first.
- Reviewer input reaches `data/` only through `python -m trialcost.suggestions apply`, run by Vincent (D40). The app must never write to `data/`.

## Stack and conventions

- Python ≥ 3.11, managed with **uv** (`uv add`, `uv run`). No conda.
- Pydantic v2, NumPy, pandas, PyYAML; pytest; ruff (lint + format). Streamlit + Altair for the app (optional `ui` extra). After any dependency change, regenerate `requirements.txt` (`uv export --extra ui --no-dev --no-hashes --format requirements-txt -o requirements.txt`): the hosted app installs from it.
- Code, comments, identifiers and docs in **English**.
- Units: money in **EUR (price year 2026)**, FX fixed at 1 EUR = 1.15 USD (`data/` constants, not code); time in **months** internally (days only in visit definitions, converted with 30.44 d/month).
- Randomness: always via a `numpy.random.Generator` passed in explicitly, seeded in tests. No global state. Every random input draws from its own child stream (`rng.spawn`, D32): keep it that way when adding inputs, or scenario comparisons (dropout delta, tornado) pick up noise.
- The engine must be **vectorised over simulations** (arrays of shape `(n_sims,)` or `(n_sims, n_sites)`); no Python loop over simulations. Target: 10,000 sims of the archetype in < 5 s.
- Small, readable functions with type hints. Vincent reviews the code himself and is more fluent in Python than JS: favour clarity over cleverness, and explain non-obvious maths in docstrings.

## Target layout

```
data/                    # source of truth for all numbers
  procedures.csv
  study_items.csv
  countries.csv
  benchmarks.csv         # validation only, never used in estimates
  benchmark_checks.csv   # benchmark -> reference design + metric (validation only)
  price_indices.csv      # US CPI medical care, to inflate USD benchmarks (validation only)
  recruitment_priors.csv # CT.gov-derived rate / dropout priors (copied into designs by hand)
  ctgov/                 # condition list, raw API snapshot, per-trial table
  procedure_overrides.csv  # country-specific procedure prices (D25)
  sponsor_model_factors.csv  # labour multipliers for functional / in_house sponsor models (D31)
  constants.csv          # engine-level numbers: FX, close-out months, raters/site, DMC interval
designs/
  rare-neuro-ph2-mri-scales.yaml
  ref-*.yaml             # reference designs for benchmark validation (fixtures, not archetypes)
src/trialcost/
  distributions.py       # Fixed / Triangular / LogNormal / Beta + vectorised sampling
  design.py              # Pydantic schema (README §3) + visits_every() helper
  catalogue.py           # load + validate CSVs (low <= mode <= high, known drivers, grades)
  recruitment.py         # site activation + Poisson-gamma recruitment
  quantities.py          # design -> quantities (screened, visits completed, scans, site-months...)
  costing.py             # quantities x unit prices -> cost lines
  simulate.py            # Monte Carlo orchestration, returns tidy results
  report.py              # quantiles, breakdowns, tornado, dropout-cost report
  validation.py          # benchmark validation report (never asserted in tests)
  ctgov.py               # ClinicalTrials.gov fetch (network) + priors estimation (offline)
  editing.py             # design <-> editable tables for the app (pure)
  suggestions.py         # reviewer suggestions; `apply` is the only reviewer path into data/ (D40)
reports/                 # generated reports (benchmark-validation.md), regenerate, don't hand-edit
app/                     # Streamlit UI only (streamlit_app.py + one module per tab, charts.py)
suggestions/             # reviewer workflow README; inbox/ is git-ignored
tests/
prototype/sanity_check.py  # throwaway loop-based prototype; reference only, do not extend
README.md
CLAUDE.md
```

## Data model of the catalogue

- `procedures.csv`: per-occurrence site cost, France reference. `country_scaling` = `site_price` (× country site price index × (1 + overhead)) or `none` (vendor price, not scaled).
- `study_items.csv`: `country_scaling` as in `procedures.csv` (site-paid items scaled, CRO/vendor items `none`, D23); `driver` ∈ `per_study | per_country | per_site | per_site_month | per_site_year | per_month | per_patient | per_patient_visit | per_scan | per_monitoring_visit | per_meeting | per_rater | per_conduct_month | per_interim_analysis`. README §2.8 says exactly what each driver counts; keep it in sync. `applies_if` is a small condition language: `always`, `country==XX`, `operations.<flag>`, `operations.<choice>==<value>`, `uses:<procedure_id>`, combined with `and`. Implement it as a tiny explicit parser, not `eval`.
- `countries.csv`: site price index and overhead (triangular), labour cost (informative only), CTA fee range, activation months range (default when a design omits `activation_months`, D21).
- `constants.csv`: engine-level parameters with the same `low/mode/high` + grade columns; read them via `Catalogue.constant()`, never hard-code.
- Automatic per-visit procedures on every **on-site** visit: `coordinator_visit`, `logistics_visit`, `vital_signs`, `patient_travel` (if `operations.patient_travel_support`). Remote visits get none of them.
- `procedure_overrides.csv`: country-specific procedure prices (D25). An override is a local EUR price: never multiplied by the site index; country overhead added only if `apply_overhead`. Read via `Catalogue.procedure_override()`.

## Modelling rules (summary; details in README §2)

- `N_screened = N / (1 − SF)`; screening visits apply to `N_screened`, others to randomised patients.
- Dropout: constant hazard `h = −ln(1 − p_annual)/12`; expected attendance at visit day `d` = `exp(−h · d/30.44)`.
- Recruitment: site `i` active from `activation_i`; rate `λ_i ~ Gamma(shape, shape/rate_mean)`; simulate until N randomised; `T_LPI`, then `T_LPLV = T_LPI + patient_duration`, `T_total = T_LPLV + closeout (6 months)`.
- Time-driven items × `T_total`; site-month items × Σ max(0, T_LPLV − activation_i); monitoring visits = site-months / interval.
- Report the **cost impact of dropout**: cost per completer and Δ cost vs. the same design with dropout = 0.
- Keep the cost lines tidy: one row per (sim, item, country, category, resource) so any breakdown is a groupby.

## Testing

- **Deterministic mode**: every distribution at its mode / median → golden test on the archetype. Hand-calculated tests use the frozen `tests/fixtures/hand-check.yaml` (D39), never the live archetype, so data updates only move the golden total and the P50 band (update both deliberately and log before → after).
- Invariants: cost increases with N, number of sites, visit count, monitoring frequency; zero dropout ≥ visits of positive dropout; removing central imaging removes all `imaging_*` lines.
- Catalogue validation tests: grades ∈ {A,B,C,D}, sources non-empty, low ≤ mode ≤ high, all procedure IDs in designs exist.
- Regression: P50 of the archetype stays within a tolerance band unless data changes (update the band deliberately).
- Benchmarks are checked in a separate report, never asserted as tests.

## Commands

```bash
uv sync --extra ui
uv run --extra ui pytest            # app tests are skipped without the ui extra
uv run --extra ui streamlit run app/streamlit_app.py
uv run python -m trialcost.suggestions review suggestions/inbox/<file>.csv   # then: apply <file> --accept 1,3 [--grade C]
uv run ruff check . && uv run ruff format .   # *.md and prototype/ are excluded in pyproject.toml
uv run python -m trialcost.simulate designs/rare-neuro-ph2-mri-scales.yaml --n-sims 10000 --seed 42
uv run python -m trialcost.validation --out reports   # benchmark validation report
uv run python -m trialcost.ctgov fetch && uv run python -m trialcost.ctgov estimate   # refresh CT.gov priors (network)
# simulate options: --deterministic --budget 5e6 --recruit-by 24 --tornado-sims 2000 --no-tornado --out results/ --lines
```

## Keeping the project log

Vincent wants a running record of the work as the project moves forward. At the end of every working session (or meaningful change), before the final commit:

- **README.md**: append an entry to a `## 10. Work log` section (create it if missing), newest first:
  ```markdown
  ### YYYY-MM-DD — <short title>
  - Done: <what changed, files touched>
  - Decisions: <new D# entries added to §5, or "none">
  - Assumptions: <added/changed in §6 and data/, or "none">
  - Results: <key numbers if the archetype output moved, e.g. P50 before → after>
  - Next: <open items>
  ```
  Also update §5 (decisions), §6 (assumptions), §9 (sources) and the status line at the top when they change.
- **CLAUDE.md**: amend this file when a working convention, layout, command or constraint changes, so the next session starts from the current rules. Don't log progress here; progress goes in the README work log.
- Never rewrite or delete past log entries. If an earlier entry turns out to be wrong, add a correction in a new entry.

## Working style with Vincent

- Propose a short plan before multi-file changes; implement in small, reviewable commits.
- Be honest and precise: say when a number is weak, when results look off against benchmarks, or when a modelling shortcut is taken.
- When showing code in chat, prefer short inline copy-pasteable snippets.
- Open questions to raise rather than decide alone: how to expose the tornado (which parameters, ranges); any change to the ClinicalTrials.gov query (D37) or estimation rules (D38).
