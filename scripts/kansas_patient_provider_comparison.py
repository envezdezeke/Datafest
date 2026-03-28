"""
Kansas-only patient-home vs provider-destination comparison dashboard.

Outputs:
- eda_output/kansas_patient_provider_comparison.html
- eda_output/kansas_patient_provider_distance_summary.csv
- eda_output/kansas_patient_home_summary.csv
- eda_output/kansas_provider_destination_summary.csv

Uses:
- patients.csv
- tigercensuscodes.csv
- encounters.csv
- providers.csv
"""

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline.offline import get_plotlyjs

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


def geocode_zip(zip_code, retries=3, sleep_s=0.25):
    url = f"https://api.zippopotam.us/us/{zip_code}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.load(response)
            place = data["places"][0]
            return {
                "zip5": zip_code,
                "provider_city": place["place name"],
                "provider_state": place["state abbreviation"],
                "provider_lat": float(place["latitude"]),
                "provider_lon": float(place["longitude"]),
            }
        except urllib.error.HTTPError:
            return None
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(sleep_s)
    return None


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.7613
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def main():
    patients = pd.read_csv(
        DATA / "patients.csv",
        usecols=["DurableKey", "CensusBlockGroupFipsCode", "OmbRace", "OmbEthnicity"],
    ).rename(columns={"DurableKey": "PatientDurableKey"})
    tiger = pd.read_csv(
        DATA / "tigercensuscodes.csv",
        usecols=["GEOID", "PopulationValue", "CENTLAT", "CENTLON"],
    )
    encounters = pd.read_csv(
        DATA / "encounters.csv",
        usecols=["EncounterKey", "PatientDurableKey", "ProviderDurableKey"],
    )
    providers = pd.read_csv(
        DATA / "providers.csv",
        usecols=["DurableKey", "OfficePostalCode", "OfficeCity", "PrimarySpecialty", "PrimaryDepartment", "Type"],
    ).rename(columns={"DurableKey": "ProviderDurableKey"})

    patients["fips"] = pd.to_numeric(
        patients["CensusBlockGroupFipsCode"].replace("*Unspecified", np.nan),
        errors="coerce",
    ).astype("Int64")
    ks_patients = patients[patients["fips"].astype(str).str.startswith("20", na=False)].copy()
    ks_patients = ks_patients.merge(tiger, left_on="fips", right_on="GEOID", how="left")
    ks_patients = ks_patients.dropna(subset=["CENTLAT", "CENTLON"]).copy()

    home_summary = (
        ks_patients.groupby(["fips", "CENTLAT", "CENTLON"], as_index=False)
        .agg(
            patient_count=("PatientDurableKey", "nunique"),
            top_race=("OmbRace", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
            top_ethnicity=("OmbEthnicity", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
            population_value=("PopulationValue", "first"),
        )
        .sort_values("patient_count", ascending=False)
    )
    home_summary["heat_weight"] = np.log1p(home_summary["patient_count"])
    home_summary.to_csv(OUT / "kansas_patient_home_summary.csv", index=False)

    providers["zip5"] = providers["OfficePostalCode"].map(zip5)
    provider_usage = (
        encounters.merge(
            ks_patients[["PatientDurableKey", "CENTLAT", "CENTLON", "fips"]],
            on="PatientDurableKey",
            how="inner",
        )
        .merge(
            providers[
                [
                    "ProviderDurableKey",
                    "zip5",
                    "OfficeCity",
                    "PrimarySpecialty",
                    "PrimaryDepartment",
                    "Type",
                ]
            ],
            on="ProviderDurableKey",
            how="left",
        )
        .dropna(subset=["zip5"])
        .copy()
    )
    provider_usage["zip5"] = provider_usage["zip5"].astype(str)

    zip_geocodes = []
    for z in sorted(provider_usage["zip5"].unique()):
        item = geocode_zip(z)
        if item and item["provider_state"] == "KS":
            zip_geocodes.append(item)
    zip_geo = pd.DataFrame(zip_geocodes)

    provider_usage = provider_usage.merge(zip_geo, on="zip5", how="inner")

    provider_summary = (
        provider_usage.groupby(
            ["zip5", "provider_city", "provider_state", "provider_lat", "provider_lon"],
            as_index=False,
        )
        .agg(
            encounter_count=("EncounterKey", "count"),
            unique_patients=("PatientDurableKey", "nunique"),
            unique_providers=("ProviderDurableKey", "nunique"),
            top_specialty=("PrimarySpecialty", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
            top_department=("PrimaryDepartment", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
            top_provider_type=("Type", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
        )
        .sort_values("encounter_count", ascending=False)
    )
    provider_summary["heat_weight"] = np.log1p(provider_summary["encounter_count"])
    provider_summary.to_csv(OUT / "kansas_provider_destination_summary.csv", index=False)

    provider_usage["distance_miles"] = haversine_miles(
        provider_usage["CENTLAT"].to_numpy(),
        provider_usage["CENTLON"].to_numpy(),
        provider_usage["provider_lat"].to_numpy(),
        provider_usage["provider_lon"].to_numpy(),
    )
    provider_usage["distance_rounded"] = provider_usage["distance_miles"].round().astype(int)

    mean_distance = float(provider_usage["distance_miles"].mean())
    median_distance = float(provider_usage["distance_miles"].median())
    mode_distance = int(provider_usage["distance_rounded"].mode().iloc[0])

    distance_summary = pd.DataFrame(
        [
            {"metric": "mean_miles", "value": round(mean_distance, 2)},
            {"metric": "median_miles", "value": round(median_distance, 2)},
            {"metric": "mode_miles_rounded", "value": mode_distance},
        ]
    )
    distance_summary.to_csv(OUT / "kansas_patient_provider_distance_summary.csv", index=False)

    top_destination = provider_summary.iloc[0]
    top_home = home_summary.iloc[0]
    shared_center = {"lat": 38.5, "lon": -96.4}

    patient_map = px.density_map(
        home_summary,
        lat="CENTLAT",
        lon="CENTLON",
        z="heat_weight",
        radius=24,
        center=shared_center,
        zoom=6,
        map_style="carto-positron",
        color_continuous_scale=[
            [0.0, "#1d3557"],
            [0.25, "#457b9d"],
            [0.5, "#00b4d8"],
            [0.75, "#ffd166"],
            [1.0, "#ef476f"],
        ],
        hover_name="fips",
        hover_data={
            "patient_count": ":,",
            "population_value": ":,",
            "top_race": True,
            "top_ethnicity": True,
            "CENTLAT": False,
            "CENTLON": False,
            "heat_weight": False,
        },
        title="Kansas Patient Home Heatmap",
    )
    patient_map.update_layout(coloraxis_colorbar_title="Home density", margin=dict(l=10, r=10, t=50, b=10))

    provider_map = px.density_map(
        provider_summary,
        lat="provider_lat",
        lon="provider_lon",
        z="heat_weight",
        radius=30,
        center=shared_center,
        zoom=6,
        map_style="carto-positron",
        color_continuous_scale=[
            [0.0, "#3a0ca3"],
            [0.25, "#7209b7"],
            [0.5, "#f72585"],
            [0.75, "#ff8c42"],
            [1.0, "#ffbe0b"],
        ],
        hover_name="zip5",
        hover_data={
            "provider_city": True,
            "encounter_count": ":,",
            "unique_patients": ":,",
            "unique_providers": ":,",
            "top_specialty": True,
            "top_department": True,
            "provider_lat": False,
            "provider_lon": False,
            "heat_weight": False,
        },
        title="Kansas Provider Destination Heatmap",
    )
    provider_map.update_layout(coloraxis_colorbar_title="Destination volume", margin=dict(l=10, r=10, t=50, b=10))

    overlay = go.Figure()
    overlay.add_trace(
        go.Densitymap(
            lat=home_summary["CENTLAT"],
            lon=home_summary["CENTLON"],
            z=home_summary["heat_weight"],
            radius=20,
            colorscale=[
                [0.0, "rgba(29,53,87,0.08)"],
                [0.25, "rgba(69,123,157,0.28)"],
                [0.5, "rgba(0,180,216,0.45)"],
                [0.75, "rgba(255,209,102,0.7)"],
                [1.0, "rgba(239,71,111,0.95)"],
            ],
            showscale=True,
            colorbar={"title": "Patient home density", "x": 0.97},
            hovertemplate="Home geography<br>Patients=%{z:.2f}<extra></extra>",
            name="Patient homes",
        )
    )
    overlay.add_trace(
        go.Scattermap(
            lat=provider_summary["provider_lat"],
            lon=provider_summary["provider_lon"],
            mode="markers",
            marker={
                "size": np.clip(np.log1p(provider_summary["encounter_count"]) * 3.8, 8, 30),
                "color": provider_summary["encounter_count"],
                "colorscale": [
                    [0.0, "#6a040f"],
                    [0.25, "#9d0208"],
                    [0.5, "#dc2f02"],
                    [0.75, "#f48c06"],
                    [1.0, "#ffba08"],
                ],
                "opacity": 0.88,
                "showscale": True,
                "colorbar": {"title": "Provider destination volume", "x": 1.06},
            },
            text=provider_summary["zip5"],
            customdata=np.stack(
                [
                    provider_summary["provider_city"],
                    provider_summary["encounter_count"],
                    provider_summary["unique_patients"],
                    provider_summary["unique_providers"],
                    provider_summary["top_specialty"].fillna(""),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>ZIP %{text}</b><br>"
                "City: %{customdata[0]}<br>"
                "Encounters: %{customdata[1]:,}<br>"
                "Patients: %{customdata[2]:,}<br>"
                "Providers: %{customdata[3]:,}<br>"
                "Top specialty: %{customdata[4]}<extra></extra>"
            ),
            name="Provider destinations",
        )
    )
    overlay.update_layout(
        title="Kansas Patients Vs Provider Destinations",
        map={
            "style": "carto-positron",
            "center": shared_center,
            "zoom": 6,
        },
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
    )

    distance_hist = px.histogram(
        provider_usage.assign(distance_bin=provider_usage["distance_rounded"]),
        x="distance_miles",
        nbins=40,
        title="Encounter-Weighted Distance Distribution",
        color_discrete_sequence=["#0d6c63"],
    )
    distance_hist.update_layout(
        bargap=0.04,
        xaxis_title="Distance from patient home centroid to provider ZIP centroid (miles)",
        yaxis_title="Encounter count",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    top_provider_rows = provider_summary.head(10)[
        [
            "zip5",
            "provider_city",
            "encounter_count",
            "unique_patients",
            "unique_providers",
            "top_specialty",
            "top_department",
        ]
    ].to_html(index=False, classes="data-table")

    insights = [
        f"Patient homes are geographically dispersed across Kansas, but provider destinations are much more concentrated, especially in Topeka-area ZIPs like {', '.join(provider_summary.head(5)['zip5'].tolist())}.",
        f"The top provider destination ZIP is {top_destination['zip5']} in {top_destination['provider_city']}, with {int(top_destination['encounter_count']):,} encounters.",
        f"The densest patient home geography in the mapped Kansas set has {int(top_home['patient_count']):,} patients linked to one census block group centroid.",
        f"The mean encounter-weighted travel distance is {mean_distance:.2f} miles, while the median is {median_distance:.2f} miles, suggesting {'a long travel tail' if mean_distance > median_distance else 'a fairly symmetric travel pattern'}.",
        f"The mode of rounded travel distance is {mode_distance} miles, which suggests many visits are still fairly local even though some destination hubs pull from much farther away.",
        f"The contrast between the broad patient-home heatmap and the tighter provider-destination clusters suggests care access is geographically centralized relative to where patients live.",
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Kansas Patient Home Vs Provider Destination Comparison</title>
  <style>
    :root {{
      --bg:#f6f1e8; --panel:rgba(255,251,245,.94); --ink:#1f2d2f; --muted:#607074; --accent:#0d6c63; --line:rgba(31,45,47,.14);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      font-family:"Avenir Next","Segoe UI",sans-serif;
      color:var(--ink);
      background:
        radial-gradient(circle at top left, rgba(196,110,65,.14), transparent 28%),
        radial-gradient(circle at top right, rgba(13,108,99,.12), transparent 26%),
        linear-gradient(180deg,#f8f3eb 0%,#efe4d7 100%);
    }}
    .wrap {{ width:min(1360px, calc(100vw - 28px)); margin:0 auto; padding:28px 0 42px; }}
    .card {{
      background:var(--panel);
      border:1px solid color-mix(in srgb, white 55%, var(--line));
      border-radius:28px;
      padding:28px;
      margin-bottom:18px;
      box-shadow:0 18px 45px rgba(25,35,38,.10);
      backdrop-filter:blur(10px);
    }}
    h1,h2 {{ margin:0 0 12px; line-height:1.08; }}
    h1 {{ font-size:clamp(34px,5vw,58px); }}
    h2 {{ font-size:28px; }}
    p {{ color:var(--muted); margin:0 0 12px; line-height:1.55; }}
    .metrics {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-top:18px; }}
    .mini {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:rgba(255,255,255,.5); }}
    .mini span {{ display:block; font-size:12px; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); margin-bottom:6px; }}
    .mini strong {{ display:block; font-size:28px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    ul {{ margin:0; padding-left:20px; }}
    li {{ margin-bottom:8px; }}
    .data-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .data-table th,.data-table td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
    @media (max-width: 1100px) {{
      .grid, .metrics {{ grid-template-columns:1fr; }}
      .card {{ padding:20px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Kansas Patient Homes Vs Provider Locations</h1>
      <p>
        This dashboard uses <code>patients.csv</code> with <code>*Unspecified</code> census codes removed, converts
        Kansas census block group FIPS to geography using <code>tigercensuscodes.csv</code>, and compares where Kansas
        patients live against where their provider encounters are concentrated.
      </p>
      <div class="metrics">
        <div class="mini"><span>Kansas Patients</span><strong>{ks_patients['PatientDurableKey'].nunique():,}</strong></div>
        <div class="mini"><span>Home Geographies</span><strong>{home_summary['fips'].nunique():,}</strong></div>
        <div class="mini"><span>Kansas Provider ZIPs</span><strong>{provider_summary['zip5'].nunique():,}</strong></div>
        <div class="mini"><span>Mapped Encounters</span><strong>{provider_usage['EncounterKey'].nunique():,}</strong></div>
        <div class="mini"><span>Median Distance</span><strong>{median_distance:.1f} mi</strong></div>
      </div>
    </section>

    <section class="grid">
      <div class="card">{patient_map.to_html(full_html=False, include_plotlyjs='cdn')}</div>
      <div class="card">{provider_map.to_html(full_html=False, include_plotlyjs=False)}</div>
    </section>

    <section class="card">{overlay.to_html(full_html=False, include_plotlyjs=False)}</section>

    <section class="grid">
      <div class="card">{distance_hist.to_html(full_html=False, include_plotlyjs=False)}</div>
      <div class="card">
        <h2>Distance And Geography Insights</h2>
        <ul>{''.join(f'<li>{item}</li>' for item in insights)}</ul>
        <p><strong>Mean distance:</strong> {mean_distance:.2f} miles</p>
        <p><strong>Median distance:</strong> {median_distance:.2f} miles</p>
        <p><strong>Mode distance:</strong> {mode_distance} miles (rounded to nearest mile)</p>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Top Kansas Provider Destinations</h2>
        {top_provider_rows}
      </div>
      <div class="card">
        <h2>Interpretation</h2>
        <p>
          The patient-home layer shows where demand originates. The provider-destination layer shows where care is
          actually concentrated. When those shapes do not align, patients are likely traveling farther than necessary
          or relying on a small number of destination hubs.
        </p>
        <p>
          Because the colors use log-scaled weights and high-contrast palettes, weaker clusters remain visible instead
          of disappearing behind the main Topeka-area hubs.
        </p>
      </div>
    </section>
  </div>
</body>
</html>
"""

    (OUT / "kansas_patient_provider_comparison.html").write_text(html, encoding="utf-8")
    print("saved -> kansas_patient_home_summary.csv")
    print("saved -> kansas_provider_destination_summary.csv")
    print("saved -> kansas_patient_provider_distance_summary.csv")
    print("saved -> kansas_patient_provider_comparison.html")
    print(f"mean miles: {mean_distance:.2f}")
    print(f"median miles: {median_distance:.2f}")
    print(f"mode miles (rounded): {mode_distance}")


if __name__ == "__main__":
    main()
