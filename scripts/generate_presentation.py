from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT = PROJECT_ROOT / "eda_output"
HTML_PATH = PROJECT_ROOT / "site" / "eda_presentation.html"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


def fmt_int(value: float | int) -> str:
    return f"{int(round(value)):,}"


def fmt_float(value: float | int, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def metric_card(label: str, value: str, detail: str = "") -> str:
    detail_html = f"<p>{html.escape(detail)}</p>" if detail else ""
    return (
        '<div class="metric-card">'
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"{detail_html}"
        "</div>"
    )


def image_card(src: str, title: str, caption: str) -> str:
    return f"""
    <figure class="chart-card">
      <img src="{html.escape(src)}" alt="{html.escape(title)}" />
      <figcaption>
        <h3>{html.escape(title)}</h3>
        <p>{html.escape(caption)}</p>
      </figcaption>
    </figure>
    """


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def dataframe_rows(df: pd.DataFrame, columns: list[str], limit: int = 5) -> str:
    rows = []
    for _, row in df.head(limit).iterrows():
        values = "".join(f"<td>{html.escape(str(row[col]))}</td>" for col in columns)
        rows.append(f"<tr>{values}</tr>")
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_presentation() -> str:
    table_summary = read_csv("00_table_summary.csv")
    joins = read_csv("join_validation.csv")
    journey = read_csv("journey_enc_per_patient_stats.csv")
    diag_groups = read_csv("diag_group_counts.csv")
    diag_specific = read_csv("diag_top30_specific.csv")
    sdoh_counts = read_csv("sdoh_question_counts.csv")
    null_patients = read_csv("null_audit_patients.csv")
    null_encounters = read_csv("null_audit_encounters.csv")

    encounters_row = table_summary.loc[table_summary["table"] == "encounters"].iloc[0]
    patients_row = table_summary.loc[table_summary["table"] == "patients"].iloc[0]
    sdoh_row = table_summary.loc[table_summary["table"] == "social_determinants"].iloc[0]
    total_memory = table_summary["mem_mb"].sum()

    patient_join = joins.loc[joins["from_key"] == "encounters.PatientDurableKey", "match_pct"].iloc[0]
    provider_join = joins.loc[joins["from_key"] == "encounters.ProviderDurableKey", "match_pct"].iloc[0]
    sdoh_join = joins.loc[
        joins["from_key"] == "social_determinants.PatientDurableKey", "match_pct"
    ].iloc[0]

    journey_stats = dict(zip(journey["Unnamed: 0"] if "Unnamed: 0" in journey.columns else journey.iloc[:, 0], journey.iloc[:, 1]))
    median_enc = journey_stats["50%"]
    mean_enc = journey_stats["mean"]
    max_enc = journey_stats["max"]

    patient_geo_missing = null_patients.loc[
        null_patients["column"] == "CensusBlockGroupFipsCode", "pct_missing_or_unspec"
    ].iloc[0]
    sex_missing = null_patients.loc[
        null_patients["column"] == "SexAssignedAtBirth", "pct_missing_or_unspec"
    ].iloc[0]
    admit_missing = null_encounters.loc[
        null_encounters["column"] == "AdmissionInstant", "pct_missing_or_unspec"
    ].iloc[0]
    visit_unspecified = null_encounters.loc[
        null_encounters["column"] == "VisitTypeDescription", "pct_missing_or_unspec"
    ].iloc[0]

    top_group = diag_groups.iloc[0]
    top_specific = diag_specific.iloc[2]
    unspecified_sdoh = sdoh_counts.iloc[0]
    first_real_sdoh = sdoh_counts.iloc[1]

    key_points = [
        f"The dataset spans 2022 to 2025 and centers on {fmt_int(encounters_row['rows'])} encounters across {fmt_int(patients_row['rows'])} patient records.",
        f"Patient utilization is highly skewed: median {fmt_float(median_enc, 0)} encounters per patient, mean {fmt_float(mean_enc, 1)}, and a maximum of {fmt_int(max_enc)}.",
        f"Trauma and fracture diagnosis groups dominate the diagnosis catalog, led by {top_group['GroupName']} with {fmt_int(top_group['count'])} records.",
        f"SDOH collection is informative but incomplete: the most common value is {unspecified_sdoh['DisplayName']}, so analysis should filter unspecified responses before interpretation.",
    ]

    caution_points = [
        f"Patient geography is suppressed or unspecified for {fmt_float(patient_geo_missing, 1)}% of patients.",
        f"Sex assigned at birth is unusable as a primary feature in this release because {fmt_float(sex_missing, 1)}% is missing or unspecified.",
        f"Admission/discharge timestamps are expectedly sparse outside hospital-style encounters; AdmissionInstant is missing in {fmt_float(admit_missing, 1)}% of rows.",
        f"Visit type descriptions are still helpful, but {fmt_float(visit_unspecified, 1)}% are unspecified.",
    ]

    recommendations = [
        "Define journeys with PatientDurableKey + DiagnosisValue + Date rather than encounter type alone.",
        "Filter diagnosis placeholders like IMO0001 and IMO0002 before presenting clinical patterns.",
        "Use SDOH only after excluding '*Unspecified' DisplayName rows and framing it as a non-randomly collected subset.",
        "Lean on encounter timing, department type, diagnosis group, and OMB race/ethnicity for the most stable presentation story.",
    ]

    charts = [
        image_card(
            "../eda_output/enc_01_volume_by_year.png",
            "Encounter Volume By Year",
            "The system’s observed activity grows steadily from 2022 through 2025.",
        ),
        image_card(
            "../eda_output/enc_03_type_distribution.png",
            "Encounter Type Distribution",
            "Office, lab, and hospital encounters make up the bulk of activity.",
        ),
        image_card(
            "../eda_output/enc_05_length_of_stay.png",
            "Hospital Length Of Stay",
            "Hospital stays are concentrated at short durations with a meaningful long tail.",
        ),
        image_card(
            "../eda_output/pat_03_demographic_vars.png",
            "Patient Demographics Snapshot",
            "Race, smoking, vital status, and MyChart status are more usable than several other patient fields.",
        ),
        image_card(
            "../eda_output/diag_01_top_groups.png",
            "Top Diagnosis Groups",
            "Fracture-related categories dominate the high-frequency diagnosis groups.",
        ),
        image_card(
            "../eda_output/journey_02_encounters_by_diagnosis_group.png",
            "Journey Length By Diagnosis Group",
            "Encounter counts vary sharply by diagnosis group, which is promising for journey segmentation.",
        ),
        image_card(
            "../eda_output/journey_03_inter_encounter_gaps.png",
            "Time Between Encounters",
            "Many patients have tightly clustered visits, with a long tail of delayed follow-up.",
        ),
        image_card(
            "../eda_output/sdoh_03_rollout_timeline.png",
            "SDOH Rollout Over Time",
            "SDOH collection ramps over time, suggesting operational rollout rather than full-population sampling.",
        ),
    ]

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ASA DataFest 2026 EDA Presentation</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --paper: rgba(255, 251, 245, 0.84);
      --ink: #1d2a2f;
      --muted: #5a6a6d;
      --accent: #0d6c63;
      --accent-2: #c86b3c;
      --line: rgba(29, 42, 47, 0.12);
      --shadow: 0 20px 50px rgba(35, 41, 44, 0.12);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(200, 107, 60, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(13, 108, 99, 0.18), transparent 26%),
        linear-gradient(180deg, #f7f2eb 0%, #efe5d8 100%);
      line-height: 1.5;
    }}

    .hero {{
      min-height: 100vh;
      padding: 72px 6vw 48px;
      display: grid;
      align-items: end;
    }}

    .hero-card {{
      max-width: 1120px;
      background: linear-gradient(145deg, rgba(255,255,255,0.72), rgba(255,248,240,0.9));
      border: 1px solid rgba(255,255,255,0.6);
      border-radius: 30px;
      box-shadow: var(--shadow);
      padding: 40px;
      backdrop-filter: blur(10px);
    }}

    .eyebrow {{
      display: inline-block;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      font-size: 12px;
      color: var(--accent);
      margin-bottom: 14px;
      font-weight: 700;
    }}

    h1, h2, h3, p {{
      margin-top: 0;
    }}

    h1 {{
      font-size: clamp(2.7rem, 5vw, 5.8rem);
      line-height: 0.95;
      margin-bottom: 18px;
      max-width: 10ch;
    }}

    .hero-grid, .metric-grid, .two-col, .chart-grid {{
      display: grid;
      gap: 18px;
    }}

    .hero-grid {{
      grid-template-columns: 1.2fr 0.9fr;
      align-items: start;
    }}

    .metric-grid {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-top: 28px;
    }}

    .metric-card, .panel, .chart-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}

    .metric-card {{
      padding: 18px 20px;
    }}

    .metric-card span {{
      display: block;
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 10px;
    }}

    .metric-card strong {{
      display: block;
      font-size: 2rem;
      line-height: 1;
      margin-bottom: 10px;
    }}

    .metric-card p {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 0;
    }}

    section {{
      padding: 24px 6vw 48px;
    }}

    .section-header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      margin-bottom: 20px;
    }}

    .section-header p {{
      max-width: 64ch;
      color: var(--muted);
      margin-bottom: 0;
    }}

    .two-col {{
      grid-template-columns: 1.1fr 0.9fr;
    }}

    .panel {{
      padding: 26px;
    }}

    ul {{
      margin: 0;
      padding-left: 18px;
    }}

    li + li {{
      margin-top: 10px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}

    th, td {{
      text-align: left;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}

    th {{
      color: var(--muted);
      font-weight: 600;
    }}

    .chart-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    .chart-card {{
      overflow: hidden;
    }}

    .chart-card img {{
      width: 100%;
      display: block;
      background: white;
      aspect-ratio: 16 / 10;
      object-fit: cover;
    }}

    .chart-card figcaption {{
      padding: 18px 20px 20px;
    }}

    .chart-card h3 {{
      margin-bottom: 8px;
      font-size: 1.1rem;
    }}

    .chart-card p, .lede, .small {{
      color: var(--muted);
    }}

    .lede {{
      font-size: 1.05rem;
      max-width: 56ch;
    }}

    .small {{
      font-size: 0.92rem;
    }}

    .footer {{
      padding: 0 6vw 60px;
    }}

    .footer .panel {{
      background: linear-gradient(135deg, rgba(13,108,99,0.95), rgba(24,56,58,0.96));
      color: white;
    }}

    .footer .panel p,
    .footer .panel li {{
      color: rgba(255,255,255,0.86);
    }}

    .chip {{
      display: inline-flex;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(13, 108, 99, 0.11);
      color: var(--accent);
      font-size: 0.85rem;
      font-weight: 600;
      margin-right: 8px;
      margin-bottom: 8px;
    }}

    @media (max-width: 900px) {{
      .hero-grid,
      .two-col,
      .chart-grid,
      .metric-grid {{
        grid-template-columns: 1fr;
      }}

      h1 {{
        max-width: none;
      }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-card">
      <div class="hero-grid">
        <div>
          <span class="eyebrow">ASA DataFest 2026</span>
          <h1>Exploratory Data Analysis of Patient Journeys</h1>
          <p class="lede">A presentation-ready summary of the Stormont Vail Health dataset, built from the project’s existing EDA outputs and focused on the patient-journey question at the center of the challenge.</p>
          <div style="margin-top:20px;">
            <span class="chip">2022-2025 coverage</span>
            <span class="chip">7 source tables</span>
            <span class="chip">Journey framing</span>
            <span class="chip">Operational + clinical data</span>
          </div>
        </div>
        <div class="panel">
          <h2>Opening Story</h2>
          {bullets(key_points)}
        </div>
      </div>
      <div class="metric-grid">
        {metric_card("Encounters", fmt_int(encounters_row["rows"]), "Central fact table for patient-system interactions.")}
        {metric_card("Patients", fmt_int(patients_row["rows"]), "De-identified patient records available for joins.")}
        {metric_card("SDOH Responses", fmt_int(sdoh_row["rows"]), "Useful but operationally rolled out over time.")}
        {metric_card("Approx. In-Memory Size", f"{fmt_float(total_memory / 1024, 1)} GB", "Combined dataframe footprint from the current EDA run.")}
      </div>
    </div>
  </header>

  <section>
    <div class="section-header">
      <div>
        <span class="eyebrow">1. Dataset Structure</span>
        <h2>What We’re Working With</h2>
      </div>
      <p>The data is organized around encounters, with supporting patient, diagnosis, department, provider, and SDOH context. This makes it naturally suited for longitudinal journey analysis once keys are joined cleanly.</p>
    </div>
    <div class="two-col">
      <div class="panel">
        <h3>Source Tables</h3>
        {dataframe_rows(table_summary, ["table", "rows", "cols", "mem_mb"], limit=7)}
      </div>
      <div class="panel">
        <h3>Join Reliability</h3>
        {bullets([
            f"Encounter to patient coverage: {fmt_float(patient_join, 1)}%",
            f"Encounter to provider coverage: {fmt_float(provider_join, 3)}%",
            f"SDOH to patient coverage: {fmt_float(sdoh_join, 1)}%",
            "Department and non--1 diagnosis joins are effectively complete in the precomputed validation run.",
        ])}
        <p class="small" style="margin-top:18px;">This is strong enough to support enriched journey tables, especially when excluding placeholder diagnosis keys and other clearly non-clinical records.</p>
      </div>
    </div>
  </section>

  <section>
    <div class="section-header">
      <div>
        <span class="eyebrow">2. Data Quality</span>
        <h2>Where The Data Is Strong And Where It’s Fragile</h2>
      </div>
      <p>The EDA shows a healthy relational structure, but several fields are intentionally sparse or operationally incomplete. Presentation conclusions should lean on stable variables and explicitly note the weak ones.</p>
    </div>
    <div class="two-col">
      <div class="panel">
        <h3>Practical Cautions</h3>
        {bullets(caution_points)}
      </div>
      <div class="panel">
        <h3>Best Usable Features</h3>
        {bullets([
            "Encounter date and encounter type",
            "Diagnosis group and diagnosis value after filtering placeholders",
            "Department type for care setting context",
            "Patient OMB race and OMB ethnicity",
            "Selected SDOH questions after removing unspecified rows",
        ])}
      </div>
    </div>
  </section>

  <section>
    <div class="section-header">
      <div>
        <span class="eyebrow">3. Utilization Patterns</span>
        <h2>Volume, Case Mix, And System Activity</h2>
      </div>
      <p>The system appears busier over time, with heavy concentration in a handful of encounter types. Hospital-specific fields behave as expected and should not be read as global missingness problems.</p>
    </div>
    <div class="chart-grid">
      {''.join(charts[:4])}
    </div>
  </section>

  <section>
    <div class="section-header">
      <div>
        <span class="eyebrow">4. Diagnoses And Journeys</span>
        <h2>The Core Patient-Journey Story</h2>
      </div>
      <p>The project’s journey framing becomes most persuasive when connected to diagnosis groups. Encounters are not evenly distributed across patients or conditions, which creates strong opportunities for segmentation and follow-up analysis.</p>
    </div>
    <div class="two-col" style="margin-bottom:18px;">
      <div class="panel">
        <h3>Journey Statistics</h3>
        {bullets([
            f"Median encounters per patient: {fmt_float(median_enc, 0)}",
            f"Mean encounters per patient: {fmt_float(mean_enc, 1)}",
            f"Maximum observed encounters for a single patient: {fmt_int(max_enc)}",
            f"Top diagnosis group: {top_group['GroupName']} ({fmt_int(top_group['count'])})",
            f"Example high-frequency specific diagnosis: {top_specific['DiagnosisName']} ({fmt_int(top_specific['count'])})",
        ])}
      </div>
      <div class="panel">
        <h3>Interpretation</h3>
        {bullets([
            "A median of 8 encounters suggests nontrivial longitudinal care even before narrowing to condition-specific journeys.",
            "The long right tail implies a small subset of patients account for a disproportionate amount of activity.",
            "Trauma-heavy diagnosis groups may produce short, intense journeys, while chronic conditions may produce longer, more diffuse ones.",
            "Inter-encounter gaps are a natural next lens for identifying delayed follow-up or care fragmentation.",
        ])}
      </div>
    </div>
    <div class="chart-grid">
      {''.join(charts[4:7])}
    </div>
  </section>

  <section>
    <div class="section-header">
      <div>
        <span class="eyebrow">5. Social Determinants</span>
        <h2>Useful Signal, But Not A Census</h2>
      </div>
      <p>SDOH data can strengthen a patient-journey narrative, but only when framed as a partially rolled-out screening program. The response volume and question mix appear driven by operational rollout rather than universal collection.</p>
    </div>
    <div class="two-col" style="margin-bottom:18px;">
      <div class="panel">
        <h3>What The Counts Say</h3>
        {bullets([
            f"Most frequent raw SDOH row: {unspecified_sdoh['DisplayName']} ({fmt_int(unspecified_sdoh['count'])})",
            f"Most frequent named question: {first_real_sdoh['DisplayName']} ({fmt_int(first_real_sdoh['count'])})",
            "Transportation, food insecurity, utilities, stress, and alcohol questions appear early and often.",
            "Depression-related questions exist, but counts are much lower and likely come from narrower workflows.",
        ])}
      </div>
      <div class="panel">
        <h3>Presentation Guidance</h3>
        {bullets([
            "Filter out '*Unspecified' DisplayName before any chart you show.",
            "Avoid treating SDOH prevalence as a population estimate.",
            "Use SDOH as explanatory context for selected journey subsets, not as a universal baseline.",
            "Pair SDOH timing with encounter timing to tell a care-access or support-needs story.",
        ])}
      </div>
    </div>
    <div class="chart-grid">
      {''.join(charts[7:])}
      {image_card(
          "../eda_output/sdoh_01_question_counts.png",
          "Top SDOH Questions",
          "Question volume reveals which screening items were used most frequently in practice.",
      )}
    </div>
  </section>

  <section class="footer">
    <div class="panel">
      <span class="eyebrow" style="color: rgba(255,255,255,0.72);">6. Recommended Next Step</span>
      <h2>How To Turn This Into A Strong DataFest Story</h2>
      {bullets(recommendations)}
      <p class="small" style="margin-top:18px;">This HTML was generated from the current EDA outputs in <code>eda_output/</code>. If the underlying plots or CSV summaries change, rerun <code>python3 generate_presentation.py</code> to refresh the presentation.</p>
    </div>
  </section>
</body>
</html>
"""
    return html_doc


def main() -> None:
    HTML_PATH.write_text(build_presentation(), encoding="utf-8")
    print(json.dumps({"html": str(HTML_PATH)}, indent=2))


if __name__ == "__main__":
    main()
