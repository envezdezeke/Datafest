"""
Create a geographic provider destination heatmap based on encounter volume.

Outputs:
- eda_output/provider_destination_heatmap.html
- eda_output/provider_destination_summary.csv

Method:
- standardize provider ZIP codes to ZIP5
- aggregate encounter volume by provider ZIP
- geocode ZIP centroids via Zippopotam.us
- render an interactive US map weighted by encounter counts
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = PROJECT_ROOT / "2026-ASA-DataFest-Data-Files"
DATA = Path(os.environ.get("DATAFEST_DATA_DIR", str(DEFAULT_DATA)))
OUT = PROJECT_ROOT / "eda_output"
OUT.mkdir(exist_ok=True)


def zip5(value):
    text = "" if pd.isna(value) else str(value)
    match = re.search(r"(\d{5})", text)
    return match.group(1) if match else np.nan


def geocode_zip(zip_code, retries=3, sleep_s=0.35):
    url = f"https://api.zippopotam.us/us/{zip_code}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.load(response)
            place = data["places"][0]
            return {
                "zip5": zip_code,
                "city": place["place name"],
                "state": place["state abbreviation"],
                "lat": float(place["latitude"]),
                "lon": float(place["longitude"]),
            }
        except urllib.error.HTTPError:
            return None
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(sleep_s)
    return None


def main():
    providers = pd.read_csv(
        DATA / "providers.csv",
        usecols=["DurableKey", "OfficeAddress", "OfficeCity", "OfficePostalCode", "PrimaryDepartment", "PrimarySpecialty", "Type"],
    ).rename(columns={"DurableKey": "ProviderDurableKey"})
    encounters = pd.read_csv(
        DATA / "encounters.csv",
        usecols=["EncounterKey", "PatientDurableKey", "ProviderDurableKey", "DepartmentKey", "PrimaryDiagnosisKey"],
    )
    diagnosis = pd.read_csv(
        DATA / "diagnosis.csv",
        usecols=["DiagnosisKey", "GroupName"],
    ).rename(columns={"DiagnosisKey": "PrimaryDiagnosisKey"})

    providers["zip5"] = providers["OfficePostalCode"].map(zip5)
    merged = encounters.merge(
        providers[
            [
                "ProviderDurableKey",
                "OfficeAddress",
                "OfficeCity",
                "OfficePostalCode",
                "zip5",
                "PrimaryDepartment",
                "PrimarySpecialty",
                "Type",
            ]
        ],
        on="ProviderDurableKey",
        how="left",
    ).merge(diagnosis, on="PrimaryDiagnosisKey", how="left")

    valid = merged.dropna(subset=["zip5"]).copy()
    valid["zip5"] = valid["zip5"].astype(str)

    summary = (
        valid.groupby("zip5", as_index=False)
        .agg(
            encounter_count=("EncounterKey", "count"),
            unique_patients=("PatientDurableKey", "nunique"),
            unique_providers=("ProviderDurableKey", "nunique"),
            top_specialty=("PrimarySpecialty", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
            top_department=("PrimaryDepartment", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
            top_diagnosis_group=("GroupName", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
        )
        .sort_values("encounter_count", ascending=False)
    )

    top_zip_volume = summary.head(250).copy()
    geocoded = []
    for z in top_zip_volume["zip5"]:
        item = geocode_zip(z)
        if item:
            geocoded.append(item)
    geo = pd.DataFrame(geocoded)
    result = top_zip_volume.merge(geo, on="zip5", how="inner")
    result["encounters_per_provider"] = (result["encounter_count"] / result["unique_providers"]).round(1)
    result["patients_per_provider"] = (result["unique_patients"] / result["unique_providers"]).round(1)
    result = result.sort_values("encounter_count", ascending=False)
    result.to_csv(OUT / "provider_destination_summary.csv", index=False)

    fig = px.density_map(
        result,
        lat="lat",
        lon="lon",
        z="encounter_count",
        radius=35,
        hover_name="zip5",
        hover_data={
            "city": True,
            "state": True,
            "encounter_count": ":,",
            "unique_patients": ":,",
            "unique_providers": ":,",
            "top_specialty": True,
            "top_department": True,
            "top_diagnosis_group": True,
            "lat": False,
            "lon": False,
            "encounters_per_provider": True,
            "patients_per_provider": True,
        },
        center={"lat": 39.0, "lon": -96.5},
        zoom=3.4,
        map_style="carto-positron",
        color_continuous_scale="YlOrRd",
        title="Provider Destination Heatmap Based On Encounter Volume",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_colorbar_title="Encounter volume",
    )

    top10 = result.head(10)
    insights = [
        f"Top destination ZIP by encounter volume: {top10.iloc[0]['zip5']} ({top10.iloc[0]['city']}, {top10.iloc[0]['state']}) with {int(top10.iloc[0]['encounter_count']):,} encounters."
        if len(top10) else "No mapped destinations available.",
        f"The mapped top 250 ZIPs cover {int(result['encounter_count'].sum()):,} encounters across {result['state'].nunique()} states."
        if len(result) else "No mapped destinations available.",
        f"The highest-volume cluster is concentrated around the Kansas / Midwest region, with strong pull into ZIPs like {', '.join(top10['zip5'].head(5).tolist())}."
        if len(top10) >= 5 else "No mapped destinations available.",
        f"The most common top specialties among high-volume ZIPs are led by {result['top_specialty'].mode().iloc[0]}."
        if not result['top_specialty'].dropna().empty else "No specialty insight available.",
        f"High-volume ZIPs average {result['encounters_per_provider'].median():.1f} encounters per provider, which helps distinguish dense provider hubs from single-provider outliers."
        if len(result) else "No provider density insight available.",
    ]

    summary_table = top10[
        ["zip5", "city", "state", "encounter_count", "unique_patients", "unique_providers", "top_specialty", "top_diagnosis_group"]
    ].to_html(index=False, classes="data-table")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Provider Destination Heatmap</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255,251,245,.94);
      --ink: #1f2d2f;
      --muted: #607074;
      --accent: #0d6c63;
      --line: rgba(31,45,47,.14);
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
    }}
    .wrap {{ width:min(1280px, calc(100vw - 28px)); margin:0 auto; padding:28px 0 42px; }}
    .card {{
      background: var(--panel);
      border: 1px solid color-mix(in srgb, white 55%, var(--line));
      border-radius: 28px;
      padding: 28px;
      margin-bottom: 18px;
      box-shadow: 0 18px 45px rgba(25,35,38,.10);
      backdrop-filter: blur(10px);
    }}
    h1,h2 {{ margin:0 0 12px; line-height:1.08; }}
    h1 {{ font-size: clamp(34px,5vw,56px); }}
    h2 {{ font-size: 28px; }}
    p {{ color: var(--muted); margin: 0 0 12px; line-height: 1.55; }}
    .grid {{ display:grid; grid-template-columns: 1.1fr .9fr; gap:18px; }}
    .metrics {{ display:grid; grid-template-columns: repeat(4,1fr); gap:14px; margin-top:18px; }}
    .mini {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:rgba(255,255,255,.5); }}
    .mini span {{ display:block; font-size:12px; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); margin-bottom:6px; }}
    .mini strong {{ display:block; font-size:28px; }}
    ul {{ margin:0; padding-left:20px; }}
    li {{ margin-bottom:8px; }}
    .data-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .data-table th, .data-table td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
    @media (max-width: 1000px) {{
      .grid, .metrics {{ grid-template-columns:1fr; }}
      .card {{ padding:20px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Provider Destination Heatmap</h1>
      <p>
        This map shows where patients are going most often based on encounter volume attached to provider office ZIP locations.
        Provider postal codes were standardized to ZIP5 and mapped to geographic centroids, then weighted by total encounters.
      </p>
      <div class="metrics">
        <div class="mini"><span>Mapped ZIPs</span><strong>{result['zip5'].nunique():,}</strong></div>
        <div class="mini"><span>Mapped Encounters</span><strong>{int(result['encounter_count'].sum()):,}</strong></div>
        <div class="mini"><span>Mapped Patients</span><strong>{int(result['unique_patients'].sum()):,}</strong></div>
        <div class="mini"><span>States Touched</span><strong>{result['state'].nunique():,}</strong></div>
      </div>
    </section>

    <section class="card">
      {fig.to_html(full_html=False, include_plotlyjs='cdn')}
    </section>

    <section class="grid">
      <div class="card">
        <h2>Key Insights</h2>
        <ul>
          {''.join(f'<li>{item}</li>' for item in insights)}
        </ul>
      </div>
      <div class="card">
        <h2>Top Destination ZIPs</h2>
        {summary_table}
      </div>
    </section>
  </div>
</body>
</html>
"""
    (OUT / "provider_destination_heatmap.html").write_text(html, encoding="utf-8")
    print(f"saved -> provider_destination_summary.csv ({len(result):,} mapped zips)")
    print("saved -> provider_destination_heatmap.html")
    print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
