"""
Build a mixed-type correlation / association matrix for the DataFest dataset.

Outputs:
- eda_output/corr_01_mixed_association_matrix.png
- eda_output/corr_01_mixed_association_matrix.csv
- eda_output/corr_02_top_associations.csv

The matrix is built on a sampled encounter-level analytic dataset joined to
patient, diagnosis, provider, department, geography, and SDOH features.
"""

import math
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.9)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "2026-ASA-DataFest-Data-Files")
DATA = os.environ.get("DATAFEST_DATA_DIR", DEFAULT_DATA)
OUT = os.path.join(PROJECT_ROOT, "eda_output")
os.makedirs(OUT, exist_ok=True)

RANDOM_SEED = 42
TARGET_SAMPLE = 50000
ENCOUNTER_ROWS_EST = 7_675_801
ENCOUNTER_SAMPLE_FRAC = min(1.0, TARGET_SAMPLE / ENCOUNTER_ROWS_EST)


def clean_category(series):
    s = series.astype("string").str.strip()
    bad = (
        s.isna()
        | (s == "")
        | s.str.lower().isin({"nan", "none", "null", "unknown"})
        | s.str.startswith("*", na=False)
        | s.isin({"Asked but No Answer", "Choose not to disclose"})
    )
    return s.mask(bad)


def cap_categories(series, top_n=10):
    s = clean_category(series)
    top = s.value_counts(dropna=True).head(top_n).index
    return s.where(s.isin(top), other="Other")


def correlation_ratio(categories, values):
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df["cat"].nunique() < 2 or df["val"].nunique() < 2:
        return np.nan
    groups = df.groupby("cat")["val"]
    counts = groups.size().astype(float)
    means = groups.mean()
    grand_mean = df["val"].mean()
    numerator = ((means - grand_mean) ** 2 * counts).sum()
    denominator = ((df["val"] - grand_mean) ** 2).sum()
    if denominator == 0:
        return np.nan
    return math.sqrt(numerator / denominator)


def cramers_v(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan
    table = pd.crosstab(df["x"], df["y"])
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0:
        return np.nan
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)
    r, k = observed.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return np.nan
    return math.sqrt(phi2corr / denom)


def association(x, y):
    x_numeric = pd.api.types.is_numeric_dtype(x)
    y_numeric = pd.api.types.is_numeric_dtype(y)
    if x_numeric and y_numeric:
        df = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
            return np.nan
        return df["x"].corr(df["y"], method="spearman")
    if x_numeric and not y_numeric:
        return correlation_ratio(y, x)
    if not x_numeric and y_numeric:
        return correlation_ratio(x, y)
    return cramers_v(x, y)


def savefig(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"saved -> {name}")


def savecsv(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path)
    print(f"saved -> {name}")


def load_lookup_tables():
    patients = pd.read_csv(
        os.path.join(DATA, "patients.csv"),
        usecols=[
            "DurableKey",
            "OmbRace",
            "OmbEthnicity",
            "MaritalStatus",
            "MyChartStatus",
            "SmokingStatus",
            "VitalStatus",
            "PatientBirthYearBin",
            "CensusBlockGroupFipsCode",
        ],
    ).rename(columns={"DurableKey": "PatientDurableKey"})

    tiger = pd.read_csv(
        os.path.join(DATA, "tigercensuscodes.csv"),
        usecols=["GEOID", "PopulationValue", "CENTLAT", "CENTLON"],
    )
    patients["GEOID"] = pd.to_numeric(
        clean_category(patients["CensusBlockGroupFipsCode"]),
        errors="coerce",
    ).astype("Int64")
    patients = patients.merge(tiger, on="GEOID", how="left")

    diagnosis = pd.read_csv(
        os.path.join(DATA, "diagnosis.csv"),
        usecols=["DiagnosisKey", "GroupName"],
    ).rename(columns={"DiagnosisKey": "PrimaryDiagnosisKey"})

    providers = pd.read_csv(
        os.path.join(DATA, "providers.csv"),
        usecols=["DurableKey", "Type", "PrimarySpecialty"],
    ).rename(columns={"DurableKey": "ProviderDurableKey", "Type": "ProviderType"})

    departments = pd.read_csv(
        os.path.join(DATA, "departments.csv"),
        usecols=["DepartmentKey", "DepartmentType", "DepartmentSpecialty", "County"],
    )

    return patients, tiger, diagnosis, providers, departments


def load_sdoh_counts():
    chunks = []
    path = os.path.join(DATA, "social_determinants.csv")
    for chunk in pd.read_csv(
        path,
        usecols=["EncounterKey", "DisplayName", "AnswerText"],
        chunksize=500000,
    ):
        valid_question = ~clean_category(chunk["DisplayName"]).isna()
        valid_answer = ~clean_category(chunk["AnswerText"]).isna()
        agg = (
            chunk.assign(
                sdoh_screen_count=1,
                sdoh_valid_question=valid_question.astype(int),
                sdoh_valid_answer=valid_answer.astype(int),
            )
            .groupby("EncounterKey", as_index=False)[
                ["sdoh_screen_count", "sdoh_valid_question", "sdoh_valid_answer"]
            ]
            .sum()
        )
        chunks.append(agg)
    combined = pd.concat(chunks, ignore_index=True)
    return combined.groupby("EncounterKey", as_index=False).sum()


def load_encounter_sample():
    usecols = [
        "EncounterKey",
        "PatientDurableKey",
        "ProviderDurableKey",
        "DepartmentKey",
        "PrimaryDiagnosisKey",
        "AdmitYear",
        "AdmitMonth",
        "AdmitDay",
        "AdmitHour",
        "DischargeYear",
        "DischargeMonth",
        "DischargeDay",
        "DischargeHour",
        "AdmissionSource",
        "AdmissionType",
        "Type",
        "VisitTypeDescription",
        "IsEdVisit",
        "IsHospitalAdmission",
        "IsHospitalOutpatientVisit",
        "IsInpatientAdmission",
        "IsObservation",
        "IsOutpatientFaceToFaceVisit",
    ]
    samples = []
    path = os.path.join(DATA, "encounters.csv")
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500000):
        sampled = chunk.sample(frac=ENCOUNTER_SAMPLE_FRAC, random_state=RANDOM_SEED)
        if not sampled.empty:
            samples.append(sampled)
    df = pd.concat(samples, ignore_index=True)
    if len(df) > TARGET_SAMPLE:
        df = df.sample(n=TARGET_SAMPLE, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def build_analytic_frame():
    patients, _, diagnosis, providers, departments = load_lookup_tables()
    sdoh = load_sdoh_counts()
    enc = load_encounter_sample()

    df = (
        enc.merge(patients, on="PatientDurableKey", how="left")
        .merge(diagnosis, on="PrimaryDiagnosisKey", how="left")
        .merge(providers, on="ProviderDurableKey", how="left")
        .merge(departments, on="DepartmentKey", how="left")
        .merge(sdoh, on="EncounterKey", how="left")
    )

    df["has_sdoh_screen"] = (df["sdoh_screen_count"].fillna(0) > 0).astype(int)
    df["sdoh_screen_count"] = df["sdoh_screen_count"].fillna(0)
    df["sdoh_valid_question"] = df["sdoh_valid_question"].fillna(0)
    df["sdoh_valid_answer"] = df["sdoh_valid_answer"].fillna(0)

    for col in [
        "AdmissionSource",
        "AdmissionType",
        "Type",
        "VisitTypeDescription",
        "OmbRace",
        "OmbEthnicity",
        "MaritalStatus",
        "MyChartStatus",
        "SmokingStatus",
        "VitalStatus",
        "GroupName",
        "ProviderType",
        "PrimarySpecialty",
        "DepartmentType",
        "DepartmentSpecialty",
        "County",
    ]:
        df[col] = cap_categories(df[col], top_n=10)

    numeric_cols = [
        "AdmitYear",
        "AdmitMonth",
        "AdmitDay",
        "AdmitHour",
        "DischargeYear",
        "DischargeMonth",
        "DischargeDay",
        "DischargeHour",
        "IsEdVisit",
        "IsHospitalAdmission",
        "IsHospitalOutpatientVisit",
        "IsInpatientAdmission",
        "IsObservation",
        "IsOutpatientFaceToFaceVisit",
        "PatientBirthYearBin",
        "PopulationValue",
        "CENTLAT",
        "CENTLON",
        "sdoh_screen_count",
        "sdoh_valid_question",
        "sdoh_valid_answer",
        "has_sdoh_screen",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [
        "AdmitYear",
        "AdmitMonth",
        "AdmitDay",
        "AdmitHour",
        "DischargeMonth",
        "DischargeDay",
        "DischargeHour",
        "AdmissionSource",
        "AdmissionType",
        "Type",
        "VisitTypeDescription",
        "IsEdVisit",
        "IsHospitalAdmission",
        "IsHospitalOutpatientVisit",
        "IsInpatientAdmission",
        "IsObservation",
        "IsOutpatientFaceToFaceVisit",
        "OmbRace",
        "OmbEthnicity",
        "MaritalStatus",
        "MyChartStatus",
        "SmokingStatus",
        "VitalStatus",
        "PatientBirthYearBin",
        "GroupName",
        "ProviderType",
        "PrimarySpecialty",
        "DepartmentType",
        "DepartmentSpecialty",
        "County",
        "PopulationValue",
        "CENTLAT",
        "CENTLON",
        "sdoh_screen_count",
        "sdoh_valid_question",
        "sdoh_valid_answer",
        "has_sdoh_screen",
    ]
    df = df[keep_cols]

    usable = []
    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.nunique() >= 2 and len(non_null) >= 100:
            usable.append(col)
    return df[usable]


def build_matrix(df):
    cols = list(df.columns)
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, col_i in enumerate(cols):
        matrix.loc[col_i, col_i] = 1.0
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            value = association(df[col_i], df[col_j])
            matrix.loc[col_i, col_j] = value
            matrix.loc[col_j, col_i] = value
    return matrix


def top_pairs(matrix, n=30):
    rows = []
    cols = list(matrix.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            val = matrix.loc[left, right]
            if pd.notna(val):
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "association": round(float(val), 4),
                        "abs_association": round(abs(float(val)), 4),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["abs_association", "association"], ascending=[False, False]
    )
    return result.head(n).set_index(["left", "right"])


def plot_matrix(matrix):
    plt.figure(figsize=(22, 18))
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Association strength"},
    )
    plt.title(
        "DataFest Mixed-Type Correlation Matrix\n"
        "Spearman for numeric pairs, Cramer's V for categorical pairs, eta for mixed pairs",
        pad=18,
    )
    plt.xticks(rotation=60, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    savefig("corr_01_mixed_association_matrix.png")


def main():
    print("building analytic frame...")
    df = build_analytic_frame()
    print(f"analytic sample rows: {len(df):,}")
    print(f"analytic variables: {len(df.columns)}")

    print("computing association matrix...")
    matrix = build_matrix(df)
    savecsv(matrix, "corr_01_mixed_association_matrix.csv")
    plot_matrix(matrix)

    top = top_pairs(matrix)
    savecsv(top, "corr_02_top_associations.csv")
    print("\nTop associations:")
    print(top.head(12))


if __name__ == "__main__":
    main()
