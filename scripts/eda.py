"""
ASA DataFest 2026 — Stormont Vail Health
Exploratory Data Analysis Script
==============================================
Outputs all charts and summary CSVs to ./eda_output/
Run: python3 eda.py
"""

import os
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "2026-ASA-DataFest-Data-Files")
DATA = os.environ.get("DATAFEST_DATA_DIR", DEFAULT_DATA)
OUT  = os.path.join(PROJECT_ROOT, "eda_output")
os.makedirs(OUT, exist_ok=True)

FILES = {
    "encounters":          "encounters.csv",
    "patients":            "patients.csv",
    "diagnosis":           "diagnosis.csv",
    "providers":           "providers.csv",
    "departments":         "departments.csv",
    "social_determinants": "social_determinants.csv",
    "tigercensus":         "tigercensuscodes.csv",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def savefig(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  saved → {name}")

def savecsv(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path)
    print(f"  saved → {name}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def null_audit(df, label):
    """Return a DataFrame with null/unspecified counts per column."""
    rows = []
    for col in df.columns:
        n_null = df[col].isna().sum()
        # DataFest convention: *Unspecified means field was asked but not answered
        n_unspec = 0
        if df[col].dtype == object:
            n_unspec = df[col].astype(str).str.startswith("*").sum()
        pct_missing = (n_null + n_unspec) / len(df) * 100
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_null": n_null,
            "n_unspecified": n_unspec,
            "pct_missing_or_unspec": round(pct_missing, 1),
            "n_unique": df[col].nunique(),
        })
    result = pd.DataFrame(rows).set_index("column")
    savecsv(result, f"null_audit_{label}.csv")
    return result

def top_counts(series, n=20):
    return series.value_counts().head(n)

def wrap_labels(ax, width=18):
    labels = [textwrap.fill(str(t.get_text()), width) for t in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL TABLES
# ══════════════════════════════════════════════════════════════════════════════
section("1. LOADING DATA")
print(f"  data directory: {DATA}")

dfs = {}
for key, fname in FILES.items():
    path = os.path.join(DATA, fname)
    print(f"  loading {fname} ...", end=" ", flush=True)
    df = pd.read_csv(path, low_memory=False)
    dfs[key] = df
    print(f"{len(df):,} rows × {df.shape[1]} cols")

enc   = dfs["encounters"]
pat   = dfs["patients"]
diag  = dfs["diagnosis"]
prov  = dfs["providers"]
dept  = dfs["departments"]
sdoh  = dfs["social_determinants"]
tiger = dfs["tigercensus"]

# ── Summary table ──
summary_rows = []
for key, df in dfs.items():
    mem = df.memory_usage(deep=True).sum() / 1e6
    summary_rows.append({"table": key, "rows": len(df), "cols": df.shape[1], "mem_mb": round(mem, 1)})
summary_df = pd.DataFrame(summary_rows)
savecsv(summary_df.set_index("table"), "00_table_summary.csv")
print(summary_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 2. NULL / MISSINGNESS AUDIT — ALL TABLES
# ══════════════════════════════════════════════════════════════════════════════
section("2. NULL / MISSINGNESS AUDIT")

for key, df in dfs.items():
    print(f"\n  [{key}]")
    audit = null_audit(df, key)
    # Only show columns with any missingness
    problem_cols = audit[audit["pct_missing_or_unspec"] > 0]
    if problem_cols.empty:
        print("    — no missing values")
    else:
        print(problem_cols[["pct_missing_or_unspec", "n_null", "n_unspecified"]].to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 3. ENCOUNTERS — CORE FACT TABLE
# ══════════════════════════════════════════════════════════════════════════════
section("3. ENCOUNTERS")

# 3a. Encounter volume by year
if "Date" in enc.columns:
    enc["Date"] = pd.to_datetime(enc["Date"], errors="coerce")
    enc["Year"] = enc["Date"].dt.year
    enc["Month"] = enc["Date"].dt.month
    enc["YearMonth"] = enc["Date"].dt.to_period("M")

    vol_year = enc["Year"].value_counts().sort_index()
    print(f"\n  Encounter volume by year:\n{vol_year.to_string()}")

    fig, ax = plt.subplots(figsize=(7, 4))
    vol_year.plot(kind="bar", ax=ax, color=sns.color_palette("muted")[0])
    ax.set_title("Encounters by Year")
    ax.set_xlabel("Year"); ax.set_ylabel("Encounters")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    plt.tight_layout(); savefig("enc_01_volume_by_year.png")

    # Monthly trend
    vol_month = enc.groupby("YearMonth").size().reset_index(name="count")
    vol_month["YearMonth_str"] = vol_month["YearMonth"].astype(str)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(vol_month["YearMonth_str"], vol_month["count"], marker="o", markersize=3)
    ax.set_title("Monthly Encounter Volume (2022–2025)")
    ax.set_xlabel("Month"); ax.set_ylabel("Encounters")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout(); savefig("enc_02_monthly_trend.png")

# 3b. Encounter type distribution
enc_type_counts = top_counts(enc["Type"], 20)
print(f"\n  Top encounter types:\n{enc_type_counts.to_string()}")

fig, ax = plt.subplots(figsize=(10, 5))
enc_type_counts.plot(kind="bar", ax=ax, color=sns.color_palette("muted"))
ax.set_title("Encounter Type Distribution (top 20)")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
wrap_labels(ax, 20)
plt.tight_layout(); savefig("enc_03_type_distribution.png")

# 3c. Boolean encounter flags
bool_flags = [c for c in enc.columns if c.startswith("Is")]
if bool_flags:
    flag_counts = {f: enc[f].sum() for f in bool_flags if enc[f].dtype in [bool, np.bool_, int, float]}
    flag_df = pd.Series(flag_counts).sort_values(ascending=False)
    print(f"\n  Boolean encounter flags:\n{flag_df.to_string()}")

    fig, ax = plt.subplots(figsize=(10, 4))
    flag_df.plot(kind="bar", ax=ax, color=sns.color_palette("Set2"))
    ax.set_title("Encounter Types by Boolean Flag")
    ax.set_ylabel("True count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    wrap_labels(ax, 22)
    plt.tight_layout(); savefig("enc_04_boolean_flags.png")

# 3d. Visit type descriptions (top 30)
vt_counts = top_counts(enc["VisitTypeDescription"], 30) if "VisitTypeDescription" in enc.columns else None
if vt_counts is not None:
    savecsv(vt_counts.rename("count").to_frame(), "enc_visit_type_desc_top30.csv")

# 3e. Admissions: length of stay distribution (hospital encounters)
if {"AdmissionInstant", "DischargeInstant"}.issubset(enc.columns):
    hosp = enc[enc["IsHospitalAdmission"] == True].copy() if "IsHospitalAdmission" in enc.columns else enc.copy()
    hosp["AdmissionInstant"] = pd.to_datetime(hosp["AdmissionInstant"], errors="coerce")
    hosp["DischargeInstant"]  = pd.to_datetime(hosp["DischargeInstant"],  errors="coerce")
    hosp["LOS_hours"] = (hosp["DischargeInstant"] - hosp["AdmissionInstant"]).dt.total_seconds() / 3600
    hosp_valid = hosp[(hosp["LOS_hours"] > 0) & (hosp["LOS_hours"] < 8760)]  # filter <1yr
    if len(hosp_valid) > 0:
        los_days = hosp_valid["LOS_hours"] / 24
        print(f"\n  Hospital LOS (days) — n={len(los_days):,}")
        print(los_days.describe().round(2).to_string())

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(los_days.clip(upper=30), bins=60, color=sns.color_palette("muted")[2], edgecolor="white")
        ax.set_title("Hospital Length of Stay Distribution (capped at 30 days)")
        ax.set_xlabel("Days"); ax.set_ylabel("Encounters")
        plt.tight_layout(); savefig("enc_05_length_of_stay.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PATIENTS — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
section("4. PATIENTS")

# 4a. Birth year bins
if "PatientBirthYearBin" in pat.columns:
    birth_counts = pat["PatientBirthYearBin"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    birth_counts.plot(kind="bar", ax=ax, color=sns.color_palette("muted")[1])
    ax.set_title("Patient Age Distribution (Birth Year Bin = upper bound, e.g. 1970 = born 1966–1970)")
    ax.set_xlabel("Birth Year Bin (5-yr, upper bound)"); ax.set_ylabel("Patients")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout(); savefig("pat_01_birth_year_bins.png")

# 4b. Race / ethnicity
for col in ["FirstRace", "OmbRace", "OmbEthnicity"]:
    if col in pat.columns:
        rc = top_counts(pat[col], 15)
        print(f"\n  {col}:\n{rc.to_string()}")
        fig, ax = plt.subplots(figsize=(9, 4))
        rc.plot(kind="barh", ax=ax)
        ax.set_title(f"Patient {col}")
        ax.invert_yaxis()
        plt.tight_layout(); savefig(f"pat_02_{col.lower()}.png")

# 4c. Marital status, smoking, vital status
for col in ["MaritalStatus", "SmokingStatus", "VitalStatus", "MyChartStatus"]:
    if col in pat.columns:
        vc = top_counts(pat[col], 15)
        print(f"\n  {col}:\n{vc.to_string()}")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, col in zip(axes, ["MaritalStatus", "SmokingStatus", "VitalStatus", "MyChartStatus"]):
    if col in pat.columns:
        vc = top_counts(pat[col], 10)
        ax.barh(vc.index, vc.values)
        ax.set_title(col, fontsize=10)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)
plt.suptitle("Patient Demographic Variables", y=1.01)
plt.tight_layout(); savefig("pat_03_demographic_vars.png")

# 4d. Geographic coverage
if "CensusBlockGroupFipsCode" in pat.columns:
    geo_known = pat[~pat["CensusBlockGroupFipsCode"].astype(str).str.startswith("*")]
    geo_unspec = pat[pat["CensusBlockGroupFipsCode"].astype(str).str.startswith("*")]
    pct_known = len(geo_known) / len(pat) * 100
    print(f"\n  Geographic coverage: {pct_known:.1f}% of patients have a known census block")
    print(f"  Known: {len(geo_known):,}  |  Suppressed/Unspecified: {len(geo_unspec):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DIAGNOSIS — ICD-10 CODES
# ══════════════════════════════════════════════════════════════════════════════
section("5. DIAGNOSIS")

# 5a. Top diagnosis groups
if "GroupName" in diag.columns:
    group_counts = top_counts(diag["GroupName"], 25)
    print(f"\n  Top 25 Diagnosis Groups:\n{group_counts.to_string()}")
    savecsv(group_counts.rename("count").to_frame(), "diag_group_counts.csv")

    fig, ax = plt.subplots(figsize=(12, 7))
    group_counts.plot(kind="barh", ax=ax, color=sns.color_palette("muted")[3])
    ax.set_title("Top 25 Diagnosis Groups (ICD-10 GroupName)")
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    plt.tight_layout(); savefig("diag_01_top_groups.png")

# 5b. Top specific diagnoses
if "DiagnosisName" in diag.columns:
    dx_counts = top_counts(diag["DiagnosisName"], 30)
    savecsv(dx_counts.rename("count").to_frame(), "diag_top30_specific.csv")
    print(f"\n  Top 10 specific diagnoses:")
    print(dx_counts.head(10).to_string())

# 5c. Unique codes
if "DiagnosisValue" in diag.columns:
    n_unique_codes = diag["DiagnosisValue"].nunique()
    n_unique_groups = diag["GroupCode"].nunique() if "GroupCode" in diag.columns else "N/A"
    print(f"\n  Unique ICD-10 codes: {n_unique_codes:,}")
    print(f"  Unique ICD-10 groups: {n_unique_groups}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════
section("6. PROVIDERS")

# 6a. Provider type and credentials
for col in ["Type", "ClinicianTitle", "PrimarySpecialty"]:
    if col in prov.columns:
        vc = top_counts(prov[col], 20)
        print(f"\n  {col}:\n{vc.head(10).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, col in zip(axes, ["Type", "PrimarySpecialty"]):
    if col in prov.columns:
        vc = top_counts(prov[col], 20)
        ax.barh(range(len(vc)), vc.values)
        ax.set_yticks(range(len(vc)))
        ax.set_yticklabels([textwrap.fill(str(v), 30) for v in vc.index], fontsize=7)
        ax.set_title(f"Provider {col} (top 20)")
        ax.invert_yaxis()
plt.tight_layout(); savefig("prov_01_type_and_specialty.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. DEPARTMENTS
# ══════════════════════════════════════════════════════════════════════════════
section("7. DEPARTMENTS")

for col in ["DepartmentType", "DepartmentSpecialty"]:
    if col in dept.columns:
        vc = top_counts(dept[col], 20)
        print(f"\n  {col}:\n{vc.to_string()}")

if "DepartmentType" in dept.columns:
    vc = top_counts(dept["DepartmentType"], 15)
    fig, ax = plt.subplots(figsize=(8, 4))
    vc.plot(kind="bar", ax=ax, color=sns.color_palette("Set2"))
    ax.set_title("Department Types")
    ax.set_ylabel("Count")
    wrap_labels(ax, 15)
    plt.tight_layout(); savefig("dept_01_types.png")

# Encounter volume per department type (join enc + dept)
if "DepartmentKey" in enc.columns and "DepartmentKey" in dept.columns:
    enc_dept = enc.merge(dept[["DepartmentKey", "DepartmentType"]], on="DepartmentKey", how="left")
    dept_type_enc = enc_dept["DepartmentType"].value_counts().head(15)
    print(f"\n  Encounter volume by DepartmentType:\n{dept_type_enc.to_string()}")

    fig, ax = plt.subplots(figsize=(9, 4))
    dept_type_enc.plot(kind="bar", ax=ax, color=sns.color_palette("muted")[4])
    ax.set_title("Encounter Volume by Department Type")
    ax.set_ylabel("Encounters")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    wrap_labels(ax, 15)
    plt.tight_layout(); savefig("dept_02_encounter_volume_by_type.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SOCIAL DETERMINANTS OF HEALTH (SDOH)
# ══════════════════════════════════════════════════════════════════════════════
section("8. SOCIAL DETERMINANTS")

# 8a. Domain coverage
if "Domain" in sdoh.columns:
    domain_counts = top_counts(sdoh["Domain"], 20)
    print(f"\n  SDOH Domain distribution:\n{domain_counts.to_string()}")

# 8b. Question distribution (DisplayName)
if "DisplayName" in sdoh.columns:
    q_counts = top_counts(sdoh["DisplayName"], 30)
    print(f"\n  Top 15 survey questions (by response count):")
    print(q_counts.head(15).to_string())
    savecsv(q_counts.rename("count").to_frame(), "sdoh_question_counts.csv")

    fig, ax = plt.subplots(figsize=(12, 8))
    q_counts.head(20).plot(kind="barh", ax=ax, color=sns.color_palette("muted")[5])
    ax.set_title("Top 20 SDOH Survey Questions by Response Count")
    ax.invert_yaxis()
    ax.set_xlabel("Response Count")
    ax.yaxis.set_tick_params(labelsize=7)
    plt.tight_layout(); savefig("sdoh_01_question_counts.png")

# 8c. Answer distribution for top questions
if "DisplayName" in sdoh.columns and "AnswerText" in sdoh.columns:
    top_questions = q_counts.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, q in zip(axes.flatten(), top_questions):
        subset = sdoh[sdoh["DisplayName"] == q]["AnswerText"].value_counts().head(10)
        ax.barh(range(len(subset)), subset.values)
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels([textwrap.fill(str(v), 25) for v in subset.index], fontsize=7)
        ax.set_title(textwrap.fill(q, 40), fontsize=8)
        ax.invert_yaxis()
    plt.suptitle("Answer Distributions — Top 6 SDOH Questions", y=1.01)
    plt.tight_layout(); savefig("sdoh_02_answer_distributions.png")

# 8d. SDOH coverage over time (rollout analysis)
if "EncounterKey" in sdoh.columns:
    sdoh_enc = sdoh.merge(enc[["EncounterKey", "Date"]].dropna(), on="EncounterKey", how="left")
    if "Date" in sdoh_enc.columns:
        sdoh_enc["Date"] = pd.to_datetime(sdoh_enc["Date"], errors="coerce")
        sdoh_enc["YearMonth"] = sdoh_enc["Date"].dt.to_period("M")
        sdoh_monthly = sdoh_enc.groupby("YearMonth").size().reset_index(name="responses")
        sdoh_monthly = sdoh_monthly[sdoh_monthly["YearMonth"].notna()]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(sdoh_monthly["YearMonth"].astype(str), sdoh_monthly["responses"], marker="o", markersize=3, color=sns.color_palette("muted")[5])
        ax.set_title("SDOH Survey Response Volume Over Time (Rollout Tracking)")
        ax.set_xlabel("Month"); ax.set_ylabel("Survey Responses")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.tight_layout(); savefig("sdoh_03_rollout_timeline.png")

# 8e. Unique patients with any SDOH response
n_sdoh_patients = sdoh["PatientDurableKey"].nunique()
n_all_patients  = pat["DurableKey"].nunique()
pct_sdoh = n_sdoh_patients / n_all_patients * 100
print(f"\n  Patients with ≥1 SDOH response: {n_sdoh_patients:,} ({pct_sdoh:.1f}% of all patients)")


# ══════════════════════════════════════════════════════════════════════════════
# 9. JOIN VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
section("9. JOIN VALIDATION")

join_checks = [
    ("encounters.PatientDurableKey", enc["PatientDurableKey"].nunique(),
     "patients.DurableKey",          pat["DurableKey"].nunique(),
     enc["PatientDurableKey"].isin(pat["DurableKey"]).mean() * 100),

    ("encounters.DepartmentKey",     enc["DepartmentKey"].nunique(),
     "departments.DepartmentKey",    dept["DepartmentKey"].nunique(),
     enc["DepartmentKey"].isin(dept["DepartmentKey"]).mean() * 100),

    ("encounters.PrimaryDiagnosisKey", enc["PrimaryDiagnosisKey"].nunique(),
     "diagnosis.DiagnosisKey",         diag["DiagnosisKey"].nunique(),
     enc[enc["PrimaryDiagnosisKey"] != -1]["PrimaryDiagnosisKey"].isin(diag["DiagnosisKey"]).mean() * 100),

    ("encounters.ProviderDurableKey",  enc["ProviderDurableKey"].nunique(),
     "providers.DurableKey",           prov["DurableKey"].nunique(),
     enc["ProviderDurableKey"].isin(prov["DurableKey"]).mean() * 100),

    ("social_determinants.PatientDurableKey", sdoh["PatientDurableKey"].nunique(),
     "patients.DurableKey",                    pat["DurableKey"].nunique(),
     sdoh["PatientDurableKey"].isin(pat["DurableKey"]).mean() * 100),
]

jdf = pd.DataFrame(join_checks, columns=["from_key", "from_unique", "to_key", "to_unique", "match_pct"])
print(jdf.to_string(index=False))
savecsv(jdf.set_index("from_key"), "join_validation.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 10. PATIENT JOURNEY SETUP
# ══════════════════════════════════════════════════════════════════════════════
section("10. PATIENT JOURNEY ANALYSIS — SETUP")

# Build a journey-ready dataset: one row per encounter, enriched with diagnosis group
if "PrimaryDiagnosisKey" in enc.columns and "DiagnosisKey" in diag.columns:
    enc_diag = enc.merge(
        diag[["DiagnosisKey", "GroupCode", "GroupName", "DiagnosisValue", "DiagnosisName"]],
        left_on="PrimaryDiagnosisKey", right_on="DiagnosisKey", how="left"
    )

    # Encounters per patient
    enc_per_patient = enc.groupby("PatientDurableKey").size()
    print(f"\n  Encounters per patient — descriptive stats:")
    print(enc_per_patient.describe().round(1).to_string())
    savecsv(enc_per_patient.describe().rename("value").to_frame(), "journey_enc_per_patient_stats.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(enc_per_patient.clip(upper=100), bins=50, color=sns.color_palette("muted")[0], edgecolor="white")
    ax.set_title("Encounters per Patient Distribution (capped at 100)")
    ax.set_xlabel("Encounters"); ax.set_ylabel("Patients")
    plt.tight_layout(); savefig("journey_01_enc_per_patient.png")

    # Journey length by top diagnosis groups
    top_groups = enc_diag["GroupName"].value_counts().head(10).index
    journey_by_dx = (
        enc_diag[enc_diag["GroupName"].isin(top_groups)]
        .groupby(["PatientDurableKey", "GroupName"])
        .size()
        .reset_index(name="n_encounters")
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    journey_by_dx.boxplot(
        column="n_encounters", by="GroupName", ax=ax,
        flierprops=dict(marker=".", markersize=2, alpha=0.3)
    )
    ax.set_title("Encounters per Patient Journey — Top 10 Diagnosis Groups")
    plt.suptitle("")
    ax.set_xlabel("")
    ax.set_ylabel("Encounters in Journey")
    ax.set_ylim(0, 50)
    ax.set_xticklabels(
        [textwrap.fill(str(t.get_text()), 20) for t in ax.get_xticklabels()],
        rotation=45, ha="right", fontsize=7
    )
    plt.tight_layout(); savefig("journey_02_encounters_by_diagnosis_group.png")

    # Time gap between consecutive encounters for same patient
    print("\n  Computing inter-encounter time gaps (sample: first 500K patients)...")
    sample_patients = enc["PatientDurableKey"].unique()[:500_000]
    enc_sample = enc[enc["PatientDurableKey"].isin(sample_patients)].copy()
    enc_sample = enc_sample.sort_values(["PatientDurableKey", "Date"])
    enc_sample["prev_date"] = enc_sample.groupby("PatientDurableKey")["Date"].shift(1)
    enc_sample["days_since_last"] = (enc_sample["Date"] - enc_sample["prev_date"]).dt.days
    gaps = enc_sample["days_since_last"].dropna()
    print(f"  Time gap (days) between consecutive encounters:")
    print(gaps.describe().round(1).to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(gaps.clip(upper=365), bins=73, color=sns.color_palette("muted")[2], edgecolor="white")
    ax.set_title("Days Between Consecutive Encounters (same patient, capped at 365d)")
    ax.set_xlabel("Days"); ax.set_ylabel("Count")
    plt.tight_layout(); savefig("journey_03_inter_encounter_gaps.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. CROSS-TABLE: DEMOGRAPHICS × ENCOUNTER VOLUME
# ══════════════════════════════════════════════════════════════════════════════
section("11. DEMOGRAPHICS × ENCOUNTERS")

# Merge patient demographics into encounters
enc_pat = enc.merge(
    pat[["DurableKey", "PatientBirthYearBin", "FirstRace", "SmokingStatus", "VitalStatus"]],
    left_on="PatientDurableKey", right_on="DurableKey", how="left"
)

# Encounters by race
race_enc = enc_pat["FirstRace"].value_counts().head(12)
print(f"\n  Encounter volume by patient race:\n{race_enc.to_string()}")

fig, ax = plt.subplots(figsize=(10, 4))
race_enc.plot(kind="barh", ax=ax, color=sns.color_palette("muted")[1])
ax.set_title("Encounter Volume by Patient Race")
ax.set_xlabel("Encounters")
ax.invert_yaxis()
plt.tight_layout(); savefig("demo_01_encounters_by_race.png")

# Encounters per patient by race (normalized)
enc_count_per_patient = enc_pat.groupby(["PatientDurableKey", "FirstRace"]).size().reset_index(name="n_enc")
race_median = enc_count_per_patient.groupby("FirstRace")["n_enc"].median().sort_values(ascending=False).head(10)
print(f"\n  Median encounters per patient by race:\n{race_median.to_string()}")

fig, ax = plt.subplots(figsize=(10, 4))
race_median.plot(kind="bar", ax=ax, color=sns.color_palette("muted")[1])
ax.set_title("Median Encounters per Patient by Race")
ax.set_ylabel("Median Encounters")
wrap_labels(ax, 20)
plt.tight_layout(); savefig("demo_02_median_enc_by_race.png")


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
section("COMPLETE")
print(f"\n  All outputs saved to: {OUT}")
print(f"  Files written: {len(os.listdir(OUT))}")
print("""
  Output index:
    00_table_summary.csv             — row/col/memory for all tables
    null_audit_*.csv                 — missingness per table
    enc_0*                           — encounter charts
    pat_0*, pat_03                   — patient demographic charts
    diag_0*, diag_*.csv             — diagnosis charts + top codes
    prov_01                          — provider type/specialty
    dept_0*                          — department charts
    sdoh_0*                          — SDOH coverage + answer charts
    join_validation.csv              — referential integrity check
    journey_0*                       — patient journey charts
    demo_0*                          — demographics × encounters
""")
