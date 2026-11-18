"""Throwaway Monte Carlo of the rare-neuro phase 2 archetype against the v0 catalogue.
Not the engine: just a check that the orders of magnitude are plausible."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
P = pd.read_csv("data/procedures.csv").set_index("procedure_id")
C = pd.read_csv("data/countries.csv").set_index("country")
I = pd.read_csv("data/study_items.csv").set_index("item_id")
NSIM = 5000
FX = 1 / 1.15

def tri(row, n, prefix=""):
    lo, mo, hi = row[prefix + "low"], row[prefix + "mode"], row[prefix + "high"]
    return np.full(n, mo, float) if lo == hi else rng.triangular(lo, mo, hi, n)

# design (same as the YAML example)
N = 50
alloc = {"FR": 4, "DE": 3, "ES": 3, "IT": 2, "US": 3}
visits = [  # (day, procedures, screening, remote)
    (-28, ["consent", "inclusion_medical", "labs_central_panel", "blood_draw", "sample_handling", "ecg", "mri_brain", "scale_clinician"], True, False),
    (0, ["physical_exam", "labs_central_panel", "blood_draw", "sample_handling", "scale_clinician", "scale_pro", "pk_sample", "pk_bioanalysis", "drug_dispensing"], False, False),
    (28, ["physical_exam", "labs_central_panel", "blood_draw", "sample_handling", "scale_clinician"], False, False),
    (84, ["physical_exam", "labs_central_panel", "blood_draw", "sample_handling", "scale_clinician", "scale_pro", "pk_sample", "pk_bioanalysis", "drug_dispensing"], False, False),
    (168, ["physical_exam", "labs_central_panel", "blood_draw", "sample_handling", "ecg", "mri_brain", "scale_clinician", "scale_pro", "drug_dispensing"], False, False),
    (252, ["scale_pro", "ae_review"], False, True),
    (364, ["physical_exam", "labs_central_panel", "blood_draw", "sample_handling", "ecg", "mri_brain", "scale_clinician", "scale_pro"], False, False),
    (392, ["ae_review"], False, True),
]
onsite_auto = ["coordinator_visit", "logistics_visit", "vital_signs", "patient_travel"]
sites = [c for c, k in alloc.items() for _ in range(k)]
S = len(sites)

out = {k: np.zeros(NSIM) for k in ["site_procedures", "site_startup", "monitoring", "management", "data_analysis",
                                    "regulatory", "imaging_central", "supply_safety_other", "investigator_fees"]}
dur = np.zeros(NSIM)
for s in range(NSIM):
    sf = rng.beta(0.35 * 20, 0.65 * 20)
    drop = rng.beta(0.15 * 20, 0.85 * 20)
    hazard = -np.log(1 - drop) / 12  # per month
    # site activation and Poisson-gamma recruitment
    act = np.array([rng.triangular(*C.loc[c, ["activation_months_low", "activation_months_mode", "activation_months_high"]]) for c in sites])
    rate_mean = rng.lognormal(np.log(0.25), np.log(1.6))
    lam = rng.gamma(1.5, rate_mean / 1.5, S)
    # simulate monthly until N randomised
    t, n, per_site = 0.0, 0, np.zeros(S)
    while n < N and t < 120:
        t += 0.25
        active = act <= t
        new = rng.poisson(lam * active * 0.25)
        per_site += new
        n += new.sum()
    t_lpi = t
    t_lplv = t_lpi + 392 / 30.44
    t_total = t_lplv + 6  # db lock, analysis, CSR
    dur[s] = t_total
    n_rand = N
    n_scr = n_rand / (1 - sf)
    share = per_site / per_site.sum()

    # ---- site procedures (per country price index + overhead)
    proc_cost = 0.0
    for c in alloc:
        idx = rng.triangular(*C.loc[c, ["site_price_low", "site_price_mode", "site_price_high"]]) if c != "FR" else 1.0
        ov = rng.triangular(*C.loc[c, ["site_overhead_low", "site_overhead_mode", "site_overhead_high"]]) if c != "FR" else 0.0
        frac = share[[i for i, x in enumerate(sites) if x == c]].sum()
        for day, procs, scr, remote in visits:
            n_pat = n_scr * frac if scr else n_rand * frac * np.exp(-hazard * max(day, 0) / 30.44)
            plist = procs + ([] if remote else onsite_auto)
            for p in plist:
                row = P.loc[p]
                u = tri(row, 1)[0]
                if row.country_scaling == "site_price":
                    u *= idx * (1 + ov)
                if p == "mri_brain" and c == "US":
                    u = rng.triangular(900, 1100, 1300)  # US research MRI override
                proc_cost += u * n_pat
        inv = tri(I.loc["investigator_fee"], 1)[0] * idx * n_rand * frac
        out["investigator_fees"][s] += inv
    out["site_procedures"][s] = proc_cost

    # ---- site start-up & close-out
    su = 0.0
    for c in sites:
        su += tri(I.loc["site_feasibility"], 1)[0] + tri(I.loc["site_activation"], 1)[0] + tri(I.loc["closeout_visit"], 1)[0]
        if c == "FR":
            su += tri(I.loc["site_fixed_fees_fr"], 1)[0]
        if c == "US":
            su += tri(I.loc["irb_local"], 1)[0]
    out["site_startup"][s] = su

    # ---- monitoring: each site active from activation to LPLV
    site_months = np.clip(t_lplv - act, 0, None).sum()
    n_mv = site_months / 2.0
    out["monitoring"][s] = (site_months * tri(I.loc["site_management"], 1)[0]
                            + n_mv * (tri(I.loc["monitoring_visit"], 1)[0] + tri(I.loc["site_monitoring_fee"], 1)[0])
                            + S * (t_lplv / 12) * tri(I.loc["pharmacy_annual"], 1)[0])

    # ---- management (time-driven)
    out["management"][s] = t_total * (tri(I.loc["project_management"], 1)[0] + tri(I.loc["medical_monitoring"], 1)[0]
                                      + tri(I.loc["pharmacovigilance"], 1)[0] + tri(I.loc["irt_monthly"], 1)[0]
                                      + tri(I.loc["edc_hosting"], 1)[0] * max(t_total - 12, 0) / t_total)

    # ---- data & analysis
    out["data_analysis"][s] = (tri(I.loc["edc_build_licence"], 1)[0] + n_scr * tri(I.loc["data_management"], 1)[0]
                               + tri(I.loc["db_lock"], 1)[0] + tri(I.loc["biostatistics"], 1)[0] + tri(I.loc["csr_writing"], 1)[0])

    # ---- regulatory
    reg = tri(I.loc["protocol_writing"], 1)[0] + tri(I.loc["cta_preparation"], 1)[0]
    for c in alloc:
        reg += tri(I.loc["country_regulatory"], 1)[0] + tri(I.loc["insurance"], 1)[0]
        lo, mo, hi = C.loc[c, ["cta_fee_eur_low", "cta_fee_eur_mode", "cta_fee_eur_high"]]
        reg += mo if lo == hi else rng.triangular(lo, mo, hi)
    out["regulatory"][s] = reg

    # ---- central imaging (MRI at SCR for screened, W24 and W52 for retained)
    n_scans = n_scr + n_rand * (np.exp(-hazard * 168 / 30.44) + np.exp(-hazard * 364 / 30.44))
    out["imaging_central"][s] = (tri(I.loc["imaging_charter"], 1)[0] + S * tri(I.loc["imaging_site_qualif"], 1)[0]
                                 + n_scans * (tri(I.loc["imaging_qc"], 1)[0] + tri(I.loc["imaging_read"], 1)[0])
                                 + tri(I.loc["imaging_lock"], 1)[0] + alloc["FR"] * tri(I.loc["imaging_site_setup_fr"], 1)[0])

    # ---- supply, safety, other
    n_dmc = max(1, int((t_lplv - act.min()) / 6))
    out["supply_safety_other"][s] = (tri(I.loc["irt_setup"], 1)[0] + tri(I.loc["drug_supply"], 1)[0]
                                     + tri(I.loc["central_lab_setup"], 1)[0] + 2 * S * tri(I.loc["rater_certification"], 1)[0]
                                     + tri(I.loc["epro_setup"], 1)[0] + tri(I.loc["dmc_setup"], 1)[0]
                                     + n_dmc * tri(I.loc["dmc_meeting"], 1)[0] + tri(I.loc["investigator_meeting"], 1)[0]
                                     + tri(I.loc["tmf_archiving"], 1)[0])

df = pd.DataFrame(out)
df["total"] = df.sum(axis=1)
q = df.quantile([0.1, 0.5, 0.9]).T / 1e6
q["share_of_median_%"] = (df.median() / df["total"].median() * 100).round(1)
print(q.round(2).to_string())
print(f"\nDuration months P10/P50/P90: {np.percentile(dur, [10, 50, 90]).round(1)}")
tot = df["total"]
print(f"Cost per randomised patient P50: {tot.median() / N:,.0f} EUR  (P10 {tot.quantile(.1)/N:,.0f}, P90 {tot.quantile(.9)/N:,.0f})")
print(f"Cost per day P50: {tot.median() / (np.median(dur) * 30.44):,.0f} EUR")
print("\nBenchmarks in EUR (no inflation adjustment):")
print(f"  ASPE CNS ph2 per patient (2018$): {48767 * FX:,.0f}")
print(f"  Moore pivotal median per patient (2015-16$): {41117 * FX:,.0f}")
print(f"  Martin median ph2 trial (2010-15$): {8.6e6 * FX / 1e6:,.2f} M")
print(f"  Tufts ph2 direct cost/day (2024$): {23737 * FX:,.0f}")
