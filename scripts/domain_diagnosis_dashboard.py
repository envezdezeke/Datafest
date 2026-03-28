"""
Encounter-level domain-to-diagnosis correlation dashboard built from the raw
DataFest source files.

Outputs:
- eda_output/domain_diagnosis_dashboard.html
- eda_output/domain_diagnosis_base_corr.csv
- eda_output/domain_diagnosis_selected_summary.csv

Uses all provided source files for context cards and encounter-level joins.
"""

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline.offline import get_plotlyjs

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = PROJECT_ROOT / "2026-ASA-DataFest-Data-Files"
DATA = Path(os.environ.get("DATAFEST_DATA_DIR", str(DEFAULT_DATA)))
OUT = PROJECT_ROOT / "eda_output"
OUT.mkdir(exist_ok=True)

DOMAIN_ORDER = [
    "Alcohol Use",
    "Depression",
    "Financial Resource Strain",
    "Food insecurity",
    "Housing Stability",
    "Transportation Needs",
    "Utilities",
    "intimate partner violance",
    "physical activity",
    "social connections",
    "stress",
]

SCATTER_SAMPLE_NEG = 1200
SCATTER_SAMPLE_POS = 400
TOP_DIAGNOSES = 12

QUESTION_SPECS = {
    "Q1: How often do you have a drink containing alcohol?": {
        "domain": "Alcohol Use",
        "kind": "map",
        "values": {
            "Never": 0,
            "Monthly or less": 0.25,
            "2-4 times a month": 0.5,
            "2-3 times a week": 0.75,
            "4 or more times a week": 1.0,
        },
    },
    "Q2: How many drinks containing alcohol do you have on a typical day when you are drinking?": {
        "domain": "Alcohol Use",
        "kind": "map",
        "values": {
            "Patient does not drink": 0,
            "1 or 2": 0.2,
            "3 or 4": 0.45,
            "5 or 6": 0.65,
            "7 to 9": 0.85,
            "10 or more": 1.0,
        },
    },
    "Q3: How often do you have six or more drinks on one occasion?": {
        "domain": "Alcohol Use",
        "kind": "map",
        "values": {
            "Never": 0,
            "Less than monthly": 0.25,
            "Monthly": 0.5,
            "Weekly": 0.75,
            "Daily or almost daily": 1.0,
        },
    },
    "Edinburgh Depression Scale Total": {
        "domain": "Depression",
        "kind": "numeric_clamp",
        "min": 0,
        "max": 30,
    },
    "PHQ-2 Total Score": {
        "domain": "Depression",
        "kind": "numeric_clamp",
        "min": 0,
        "max": 6,
    },
    "Patient Health Questionnaire-2 Score": {
        "domain": "Depression",
        "kind": "numeric_clamp",
        "min": 0,
        "max": 6,
    },
    "The thought of harming myself has occurred to me.": {
        "domain": "Depression",
        "kind": "map",
        "values": {
            "Never": 0,
            "Hardly ever": 0.33,
            "Sometimes": 0.66,
            "Often": 1.0,
        },
    },
    "In the past 12 months has the electric, gas, oil, or water company threatened to shut off services in your home?": {
        "domain": "Financial Resource Strain",
        "kind": "map",
        "values": {"No": 0, "Yes": 0.7, "Already shut off": 1.0},
    },
    "In the last 12 months, was there a time when you were not able to pay the mortgage or rent on time?": {
        "domain": "Financial Resource Strain",
        "kind": "map",
        "values": {"No": 0, "Yes": 1.0},
    },
    "Within the past 12 months, you worried that your food would run out before you got the money to buy more.": {
        "domain": "Food insecurity",
        "kind": "map",
        "values": {"Never true": 0, "Sometimes true": 0.5, "Often true": 1.0},
    },
    "Within the past 12 months, the food you bought just didn't last and you didn't have money to get more.": {
        "domain": "Food insecurity",
        "kind": "map",
        "values": {"Never true": 0, "Sometimes true": 0.5, "Often true": 1.0},
    },
    "At any time in the past 12 months, were you homeless or living in a shelter (including now)?": {
        "domain": "Housing Stability",
        "kind": "map",
        "values": {"No": 0, "Yes": 1.0},
    },
    "In the past 12 months, how many times have you moved where you were living?": {
        "domain": "Housing Stability",
        "kind": "numeric_clip",
        "min": 0,
        "max": 10,
    },
    "In the past 12 months, has lack of transportation kept you from medical appointments or from getting medications?": {
        "domain": "Transportation Needs",
        "kind": "map",
        "values": {"No": 0, "Yes": 1.0},
    },
    "In the past 12 months, has lack of transportation kept you from meetings, work, or from getting things needed for daily living?": {
        "domain": "Transportation Needs",
        "kind": "map",
        "values": {"No": 0, "Yes": 1.0},
    },
    "How hard is it for you to pay for the very basics like food, housing, medical care, and heating?": {
        "domain": "Utilities",
        "kind": "map",
        "values": {
            "Not hard at all": 0,
            "Not very hard": 0.25,
            "Somewhat hard": 0.5,
            "Hard": 0.75,
            "Very hard": 1.0,
        },
    },
    "Within the last year, have you been humiliated or emotionally abused in other ways by your partner or ex-partner?": {
        "domain": "intimate partner violance",
        "kind": "map",
        "values": {"No": 0, "Yes": 1.0},
    },
    "Within the last year, have you been afraid of your partner or ex-partner?": {
        "domain": "intimate partner violance",
        "kind": "map",
        "values": {"No": 0, "Yes": 1.0},
    },
    "Within the last year, have you been kicked, hit, slapped, or otherwise physically hurt by your partner or ex-partner?": {
        "domain": "intimate partner violance",
        "kind": "map",
        "values": {"No": 0, "Yes": 1.0},
    },
    "Within the last year, have you been raped or forced to have any kind of sexual activity by your partner or ex-partner?": {
        "domain": "intimate partner violance",
        "kind": "map",
        "values": {"No": 0, "Yes": 1.0},
    },
    "On average, how many days per week do you engage in moderate to strenuous exercise (like a brisk walk)?": {
        "domain": "physical activity",
        "kind": "reverse_numeric_clip",
        "min": 0,
        "max": 7,
    },
    "On average, how many minutes do you engage in exercise at this level?": {
        "domain": "physical activity",
        "kind": "reverse_numeric_clip",
        "min": 0,
        "max": 150,
    },
    "How often do you get together with friends or relatives?": {
        "domain": "social connections",
        "kind": "inverse_map",
        "values": {
            "Never": 1.0,
            "Once a week": 0.5,
            "Twice a week": 0.25,
            "Three times a week": 0.12,
            "More than three times a week": 0.0,
        },
    },
    "In a typical week, how many times do you talk on the phone with family, friends, or neighbors?": {
        "domain": "social connections",
        "kind": "inverse_map",
        "values": {
            "Never": 1.0,
            "Once a week": 0.5,
            "Twice a week": 0.25,
            "Three times a week": 0.12,
            "More than three times a week": 0.0,
        },
    },
    "Do you belong to any clubs or organizations such as church groups, unions, fraternal or athletic groups, or school groups?": {
        "domain": "social connections",
        "kind": "map",
        "values": {"Yes": 0, "No": 1.0},
    },
    "Are you married, widowed, divorced, separated, never married, or living with a partner?": {
        "domain": "social connections",
        "kind": "map",
        "values": {
            "Married": 0.0,
            "Living with partner": 0.1,
            "Never married": 0.45,
            "Divorced": 0.6,
            "Widowed": 0.65,
            "Separated": 0.8,
        },
    },
    "How often do you attend church or religious services?": {
        "domain": "social connections",
        "kind": "inverse_map",
        "values": {
            "Never": 1.0,
            "1 to 4 times per year": 0.5,
            "More than 4 times per year": 0.0,
        },
    },
    "How often do you attend meetings of the clubs or organizations you belong to?": {
        "domain": "social connections",
        "kind": "inverse_map",
        "values": {
            "Never": 1.0,
            "1 to 4 times per year": 0.5,
            "More than 4 times per year": 0.0,
        },
    },
    "Do you feel stress - tense, restless, nervous, or anxious, or unable to sleep at night because your mind is troubled all the time - these days?": {
        "domain": "stress",
        "kind": "map",
        "values": {
            "Not at all": 0,
            "Only a little": 0.25,
            "To some extent": 0.5,
            "Rather much": 0.75,
            "Very much": 1.0,
        },
    },
}


def invalid_answer(series):
    s = series.astype("string").str.strip()
    return (
        s.isna()
        | s.str.lower().isin({"nan", "none"})
        | s.isin(["Patient declined", "Patient unable to answer", "*Unknown"])
    )


def parse_number(text):
    if pd.isna(text):
        return np.nan
    match = re.search(r"-?\d+(\.\d+)?", str(text))
    return float(match.group(0)) if match else np.nan


def normalize_numeric(values, min_value, max_value, reverse=False):
    values = values.clip(lower=min_value, upper=max_value)
    norm = (values - min_value) / (max_value - min_value)
    return 1 - norm if reverse else norm


def encode_answer(question, answers):
    spec = QUESTION_SPECS[question]
    bad = invalid_answer(answers)
    if spec["kind"] in {"map", "inverse_map"}:
        out = answers.astype("string").str.strip().map(spec["values"])
    elif spec["kind"] == "numeric_clamp":
        nums = pd.to_numeric(answers, errors="coerce")
        out = normalize_numeric(nums, spec["min"], spec["max"], reverse=False)
    elif spec["kind"] == "numeric_clip":
        nums = answers.map(parse_number)
        out = normalize_numeric(nums, spec["min"], spec["max"], reverse=False)
    else:
        nums = answers.map(parse_number)
        out = normalize_numeric(nums, spec["min"], spec["max"], reverse=True)
    out = pd.to_numeric(out, errors="coerce")
    out[bad] = np.nan
    return out


def load_all_tables():
    encounters = pd.read_csv(
        DATA / "encounters.csv",
        usecols=["EncounterKey", "PatientDurableKey", "ProviderDurableKey", "DepartmentKey", "PrimaryDiagnosisKey"],
    )
    diagnosis = pd.read_csv(
        DATA / "diagnosis.csv",
        usecols=["DiagnosisKey", "GroupName", "DiagnosisName", "DiagnosisValue"],
    ).rename(columns={"DiagnosisKey": "PrimaryDiagnosisKey"})
    providers = pd.read_csv(DATA / "providers.csv", usecols=["DurableKey", "PrimarySpecialty"]).rename(
        columns={"DurableKey": "ProviderDurableKey"}
    )
    departments = pd.read_csv(DATA / "departments.csv", usecols=["DepartmentKey", "DepartmentType"])
    patients = pd.read_csv(
        DATA / "patients.csv",
        usecols=["DurableKey", "PatientBirthYearBin", "OmbRace", "OmbEthnicity", "CensusBlockGroupFipsCode"],
    ).rename(columns={"DurableKey": "PatientDurableKey"})
    tiger = pd.read_csv(DATA / "tigercensuscodes.csv", usecols=["GEOID", "PopulationValue"])
    return encounters, diagnosis, providers, departments, patients, tiger


def load_scored_sdoh():
    records = []
    question_set = set(QUESTION_SPECS.keys())
    for chunk in pd.read_csv(
        DATA / "social_determinants.csv",
        usecols=["EncounterKey", "PatientDurableKey", "DisplayName", "AnswerText", "Domain"],
        chunksize=300000,
    ):
        chunk = chunk[chunk["DisplayName"].isin(question_set)].copy()
        if chunk.empty:
            continue
        chunk["score"] = np.nan
        for question in chunk["DisplayName"].drop_duplicates():
            mask = chunk["DisplayName"] == question
            chunk.loc[mask, "score"] = encode_answer(question, chunk.loc[mask, "AnswerText"])
        chunk = chunk.dropna(subset=["score"])
        records.append(chunk[["EncounterKey", "PatientDurableKey", "Domain", "score"]])
    return pd.concat(records, ignore_index=True)


def build_encounter_domain_frame():
    encounters, diagnosis, providers, departments, patients, tiger = load_all_tables()
    sdoh_scored = load_scored_sdoh()

    domain_scores = (
        sdoh_scored.groupby(["EncounterKey", "Domain"], as_index=False)["score"].mean()
        .pivot(index="EncounterKey", columns="Domain", values="score")
        .reset_index()
    )
    for domain in DOMAIN_ORDER:
        if domain not in domain_scores.columns:
            domain_scores[domain] = np.nan
    domain_scores = domain_scores[["EncounterKey"] + DOMAIN_ORDER]

    frame = (
        domain_scores.merge(encounters, on="EncounterKey", how="left")
        .merge(diagnosis, on="PrimaryDiagnosisKey", how="left")
        .merge(providers, on="ProviderDurableKey", how="left")
        .merge(departments, on="DepartmentKey", how="left")
        .merge(patients, on="PatientDurableKey", how="left")
    )

    patients_geo = patients.copy()
    patients_geo["GEOID"] = pd.to_numeric(
        patients_geo["CensusBlockGroupFipsCode"].replace("*Unspecified", np.nan),
        errors="coerce",
    ).astype("Int64")
    patients_geo = patients_geo.merge(tiger, on="GEOID", how="left")

    return frame, sdoh_scored, encounters, diagnosis, providers, departments, patients, patients_geo, tiger


def compute_context(frame, sdoh_scored, encounters, diagnosis, providers, departments, patients, patients_geo, tiger):
    context = {
        "encounter_rows": int(len(encounters)),
        "patient_rows": int(len(patients)),
        "diagnosis_rows": int(len(diagnosis)),
        "provider_rows": int(len(providers)),
        "department_rows": int(len(departments)),
        "sdoh_rows": int(len(sdoh_scored)),
        "tiger_rows": int(len(tiger)),
        "sdoh_encounters": int(frame["EncounterKey"].nunique()),
        "sdoh_patients": int(frame["PatientDurableKey"].nunique()),
        "sdoh_providers": int(frame["ProviderDurableKey"].nunique()),
        "sdoh_departments": int(frame["DepartmentKey"].nunique()),
        "provider_specialties": int(frame["PrimarySpecialty"].nunique(dropna=True)),
        "department_types": int(frame["DepartmentType"].nunique(dropna=True)),
        "tiger_linked_patients": int(patients_geo["PopulationValue"].notna().sum()),
    }
    return context


def select_diagnoses(frame):
    counts = (
        frame["GroupName"]
        .dropna()
        .value_counts()
        .head(TOP_DIAGNOSES)
    )
    return counts


def spearman_corr(series_a, series_b):
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if len(df) < 3 or df.iloc[:, 0].nunique() < 2 or df.iloc[:, 1].nunique() < 2:
        return np.nan
    return round(float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")), 3)


def build_base_corr(frame):
    corr = frame[DOMAIN_ORDER].corr(method="spearman").round(3)
    corr.to_csv(OUT / "domain_diagnosis_base_corr.csv")
    return corr


def build_diagnosis_payload(frame, diagnosis_counts, base_corr):
    payload = {}
    summary_rows = []
    for diagnosis_name, n_obs in diagnosis_counts.items():
        diag_flag = (frame["GroupName"] == diagnosis_name).astype(int)
        corr_vector = {domain: spearman_corr(frame[domain], diag_flag) for domain in DOMAIN_ORDER}

        matrix = base_corr.copy()
        label = f"Diagnosis: {diagnosis_name}"
        matrix[label] = [corr_vector[d] for d in DOMAIN_ORDER]
        matrix.loc[label] = [corr_vector[d] for d in DOMAIN_ORDER] + [1.0]
        matrix = matrix.reindex(DOMAIN_ORDER + [label], columns=DOMAIN_ORDER + [label])

        sorted_corrs = sorted(
            [{"domain": d, "corr": 0.0 if pd.isna(corr_vector[d]) else corr_vector[d]} for d in DOMAIN_ORDER],
            key=lambda x: abs(x["corr"]),
            reverse=True,
        )
        strongest = sorted_corrs[:3]

        pos = frame.loc[diag_flag == 1, DOMAIN_ORDER].copy()
        neg = frame.loc[diag_flag == 0, DOMAIN_ORDER].copy()
        if len(pos) > SCATTER_SAMPLE_POS:
            pos = pos.sample(SCATTER_SAMPLE_POS, random_state=42)
        if len(neg) > SCATTER_SAMPLE_NEG:
            neg = neg.sample(SCATTER_SAMPLE_NEG, random_state=42)
        scatter_df = pd.concat(
            [
                pos.assign(diagnosis_status="Selected diagnosis"),
                neg.assign(diagnosis_status="Other diagnoses"),
            ],
            ignore_index=True,
        ).dropna(subset=DOMAIN_ORDER, how="all")

        payload[diagnosis_name] = {
            "count": int(n_obs),
            "prevalence": round(float(diag_flag.mean() * 100), 2),
            "matrix": matrix.fillna(0).values.tolist(),
            "domains": DOMAIN_ORDER + [label],
            "domain_to_diagnosis": [{"domain": x["domain"], "corr": round(float(x["corr"]), 3)} for x in sorted_corrs],
            "insights": [
                f"{x['domain']} correlation: {x['corr']:.3f}" for x in strongest
            ],
            "scatter": scatter_df.to_dict(orient="records"),
        }
        for item in strongest:
            summary_rows.append(
                {
                    "diagnosis": diagnosis_name,
                    "encounter_count": int(n_obs),
                    "prevalence_pct": round(float(diag_flag.mean() * 100), 2),
                    "domain": item["domain"],
                    "correlation": round(float(item["corr"]), 3),
                }
            )

    pd.DataFrame(summary_rows).to_csv(OUT / "domain_diagnosis_selected_summary.csv", index=False)
    return payload


def top_global_insights(base_corr):
    rows = []
    for i, left in enumerate(DOMAIN_ORDER):
        for right in DOMAIN_ORDER[i + 1 :]:
            val = base_corr.loc[left, right]
            if pd.notna(val):
                rows.append((left, right, float(val)))
    rows.sort(key=lambda x: abs(x[2]), reverse=True)
    return rows[:6]


def render_html(context, diagnosis_counts, diagnosis_payload, global_insights):
    default_diag = next(iter(diagnosis_counts.index))
    stats_cards = [
        ("Encounter Rows", f"{context['encounter_rows']:,}", "raw encounters table"),
        ("Patients", f"{context['patient_rows']:,}", "patient master rows"),
        ("Diagnosis Rows", f"{context['diagnosis_rows']:,}", "diagnosis lookup rows"),
        ("Providers", f"{context['provider_rows']:,}", "provider lookup rows"),
        ("Departments", f"{context['department_rows']:,}", "department lookup rows"),
        ("SDOH Responses", f"{context['sdoh_rows']:,}", "scored screening rows"),
        ("SDOH Encounters", f"{context['sdoh_encounters']:,}", "encounters with usable domain scores"),
        ("SDOH Patients", f"{context['sdoh_patients']:,}", "unique patients in the correlation base"),
        ("Provider Specialties", f"{context['provider_specialties']:,}", "specialties touched by these encounters"),
        ("Department Types", f"{context['department_types']:,}", "department types touched by these encounters"),
        ("Tiger Rows", f"{context['tiger_rows']:,}", "census geographies available"),
        ("Tiger-Linked Patients", f"{context['tiger_linked_patients']:,}", "patients with matched census geography"),
    ]

    cards_html = "".join(
        f'<div class="mini"><span>{label}</span><strong>{value}</strong><p>{desc}</p></div>'
        for label, value, desc in stats_cards
    )
    insights_html = "".join(
        f"<li><strong>{left}</strong> vs <strong>{right}</strong>: {val:.3f}</li>"
        for left, right, val in global_insights
    )
    diagnosis_options = "".join(
        f'<option value="{diag}">{diag} ({count:,})</option>'
        for diag, count in diagnosis_counts.items()
    )
    graph_suggestions = "".join(
        f"<li>{item}</li>"
        for item in [
            "Clustered heatmap of average domain scores by diagnosis to show domain bundles by clinical condition.",
            "Grouped bar chart of domain-to-diagnosis correlations for the top diagnoses side by side.",
            "Parallel-coordinates chart across domain scores for encounters with and without the selected diagnosis.",
            "Risk-profile radar chart comparing the selected diagnosis against all other encounters.",
            "Network graph where domain nodes connect to diagnosis when absolute correlation passes a threshold.",
            "Small-multiple violin plots of each domain score split by selected diagnosis vs other diagnoses.",
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Encounter-Level Domain Diagnosis Dashboard</title>
  <style>
    :root {{
      --bg: #f5efe5;
      --panel: rgba(255,251,245,.93);
      --ink: #1f2d2f;
      --muted: #617175;
      --accent: #0d6c63;
      --line: rgba(31,45,47,.14);
      --shadow: 0 18px 45px rgba(25,35,38,.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next","Segoe UI",sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(196,110,65,.14), transparent 28%),
        radial-gradient(circle at top right, rgba(13,108,99,.12), transparent 26%),
        linear-gradient(180deg,#f8f3eb 0%,#efe4d7 100%);
      line-height: 1.55;
    }}
    .wrap {{ width:min(1360px,calc(100vw - 28px)); margin:0 auto; padding:28px 0 44px; }}
    .card {{
      background: var(--panel);
      border: 1px solid color-mix(in srgb, white 55%, var(--line));
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      padding: 28px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      display:inline-block;
      margin-bottom:12px;
      color:var(--accent);
      font-size:12px;
      letter-spacing:.18em;
      text-transform:uppercase;
      font-weight:700;
    }}
    h1,h2,h3 {{ margin:0 0 10px; line-height:1.08; }}
    h1 {{ font-size: clamp(34px, 5vw, 58px); }}
    h2 {{ font-size: 28px; }}
    p {{ margin:0 0 12px; color:var(--muted); }}
    .meta {{ display:grid; grid-template-columns: repeat(4,1fr); gap:14px; margin-top:18px; }}
    .mini {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:rgba(255,255,255,.5); }}
    .mini span {{ display:block; font-size:12px; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:6px; }}
    .mini strong {{ display:block; font-size:28px; margin-bottom:4px; }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
    ul {{ margin:0; padding-left:20px; }}
    li {{ margin-bottom:8px; }}
    .control-row {{ display:flex; gap:16px; align-items:end; flex-wrap:wrap; }}
    label {{ display:block; font-size:13px; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); margin-bottom:8px; }}
    select {{
      min-width: 420px;
      max-width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      font: inherit;
      background: rgba(255,255,255,.8);
      color: var(--ink);
    }}
    .plot {{ overflow:auto; min-height: 480px; }}
    .insight-box {{
      border-left: 4px solid var(--accent);
      padding-left: 16px;
      margin-top: 10px;
    }}
    code {{ color:var(--accent); }}
    @media (max-width: 1100px) {{
      .grid, .meta {{ grid-template-columns:1fr; }}
      .card {{ padding:20px; }}
      select {{ min-width: 0; width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <div class="eyebrow">Encounter-Level Domain Analysis</div>
      <h1>Domain Names Against A Selected Diagnosis</h1>
      <p>
        This dashboard ignores the earlier templates and rebuilds the analysis directly from the raw source files.
        Domain scores are calculated per <code>EncounterKey</code> from SDOH responses, then related to a selected
        primary diagnosis group from the encounter table.
      </p>
      <div class="meta">{cards_html}</div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>What The Data Says Overall</h2>
        <p>These are the strongest domain-to-domain relationships before choosing any diagnosis.</p>
        <ul>{insights_html}</ul>
      </div>
      <div class="card">
        <h2>Graphs To Add Next</h2>
        <p>These will make the diagnosis-specific domain story easier to explain in a strategy presentation.</p>
        <ul>{graph_suggestions}</ul>
      </div>
    </section>

    <section class="card">
      <div class="control-row">
        <div>
          <label for="diagnosis-select">Diagnosis Selector</label>
          <select id="diagnosis-select">{diagnosis_options}</select>
        </div>
        <div class="insight-box">
          <h3 id="selected-diagnosis-title">Diagnosis</h3>
          <p id="selected-diagnosis-meta"></p>
          <ul id="selected-diagnosis-insights"></ul>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Correlation Matrix</h2>
        <p>
          The matrix uses the real domain labels such as <code>stress</code>, <code>Alcohol Use</code>,
          and <code>Depression</code>. The last row and column are the selected diagnosis indicator.
        </p>
        <div id="heatmap" class="plot"></div>
      </div>
      <div class="card">
        <h2>Domain To Diagnosis Correlation</h2>
        <p>Positive values mean higher domain-risk scores appear more often with the selected diagnosis.</p>
        <div id="barplot" class="plot"></div>
      </div>
    </section>

    <section class="card">
      <h2>Scatter Matrix</h2>
      <p>
        The scatter matrix uses the same domain variables and colors the points by whether the encounter belongs to the selected diagnosis.
        It is sampled for browser performance.
      </p>
      <div id="scatter" class="plot"></div>
    </section>
  </div>

  <script>{get_plotlyjs()}</script>
  <script>
    const diagnosisData = {json.dumps(diagnosis_payload)};
    const defaultDiagnosis = {json.dumps(default_diag)};
    const redWhiteGreen = [
      [0.0, "#b2182b"],
      [0.5, "#f7f7f7"],
      [1.0, "#1b7837"]
    ];

    function updateDiagnosisView(name) {{
      const payload = diagnosisData[name];
      if (!payload) return;

      document.getElementById("selected-diagnosis-title").textContent = name;
      document.getElementById("selected-diagnosis-meta").textContent =
        `${{payload.count.toLocaleString()}} encounters, ${{payload.prevalence.toFixed(2)}}% of the scored encounter base`;
      document.getElementById("selected-diagnosis-insights").innerHTML =
        payload.insights.map(item => `<li>${{item}}</li>`).join("");

      Plotly.newPlot("heatmap", [{{
        type: "heatmap",
        z: payload.matrix,
        x: payload.domains,
        y: payload.domains,
        colorscale: redWhiteGreen,
        zmin: -1,
        zmax: 1,
        text: payload.matrix.map(row => row.map(v => Number(v).toFixed(2))),
        texttemplate: "%{{text}}",
        textfont: {{size: 11}},
        hovertemplate: "<b>%{{y}}</b> vs <b>%{{x}}</b><br>r=%{{z:.3f}}<extra></extra>",
        colorbar: {{title: "Spearman r"}}
      }}], {{
        height: 900,
        margin: {{l: 180, r: 30, t: 40, b: 180}},
        yaxis: {{autorange: "reversed"}},
        xaxis: {{tickangle: 45}}
      }}, {{responsive: true, displaylogo: false}});

      const barX = payload.domain_to_diagnosis.slice().reverse().map(d => d.corr);
      const barY = payload.domain_to_diagnosis.slice().reverse().map(d => d.domain);
      const barColors = barX.map(v => v >= 0 ? "#1b7837" : "#b2182b");
      Plotly.newPlot("barplot", [{{
        type: "bar",
        orientation: "h",
        x: barX,
        y: barY,
        marker: {{color: barColors}},
        hovertemplate: "%{{y}}<br>r=%{{x:.3f}}<extra></extra>"
      }}], {{
        height: 900,
        margin: {{l: 210, r: 30, t: 40, b: 60}},
        xaxis: {{title: "Spearman correlation", range: [-1, 1]}},
        yaxis: {{title: ""}}
      }}, {{responsive: true, displaylogo: false}});

      const dims = payload.domain_to_diagnosis.map(d => d.domain);
      const scatterRecords = payload.scatter;
      Plotly.newPlot("scatter", [{{
        type: "splom",
        dimensions: dims.map(dim => ({{
          label: dim,
          values: scatterRecords.map(r => r[dim])
        }})),
        text: scatterRecords.map(r => r.diagnosis_status),
        marker: {{
          color: scatterRecords.map(r => r.diagnosis_status === "Selected diagnosis" ? "#0d6c63" : "#bf6d3e"),
          size: 5,
          opacity: 0.45
        }},
        diagonal: {{visible: false}},
        showupperhalf: false,
        hovertemplate: "%{{text}}<extra></extra>"
      }}], {{
        height: 1180,
        margin: {{l: 90, r: 30, t: 40, b: 70}},
      }}, {{responsive: true, displaylogo: false}});
    }}

    document.getElementById("diagnosis-select").addEventListener("change", (e) => {{
      updateDiagnosisView(e.target.value);
    }});

    document.getElementById("diagnosis-select").value = defaultDiagnosis;
    updateDiagnosisView(defaultDiagnosis);
  </script>
</body>
</html>
"""
    return html


def main():
    print(f"reading data from: {DATA}")
    frame, sdoh_scored, encounters, diagnosis, providers, departments, patients, patients_geo, tiger = build_encounter_domain_frame()
    context = compute_context(frame, sdoh_scored, encounters, diagnosis, providers, departments, patients, patients_geo, tiger)
    diagnosis_counts = select_diagnoses(frame)
    base_corr = build_base_corr(frame)
    diagnosis_payload = build_diagnosis_payload(frame, diagnosis_counts, base_corr)
    global_insights = top_global_insights(base_corr)

    html = render_html(context, diagnosis_counts, diagnosis_payload, global_insights)
    out_path = OUT / "domain_diagnosis_dashboard.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"saved -> {out_path.name}")
    print("top diagnosis selector choices:")
    print(diagnosis_counts.to_string())


if __name__ == "__main__":
    main()
