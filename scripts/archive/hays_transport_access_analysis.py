"""
Hays-area transportation and provider access analysis.

Outputs:
- eda_output/hays_transport_access.html
- eda_output/hays_transport_destinations.csv
- eda_output/hays_transport_candidate_areas.csv
- eda_output/hays_transport_summary.csv

Method note:
- Patients do not have an explicit city field.
- "Hays city patients" are approximated as Kansas patients whose census block
  group centroid lies within 10 miles of the Hays, KS (ZIP 67601) provider
  centroid.
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = PROJECT_ROOT / "2026-ASA-DataFest-Data-Files"
DATA = Path(os.environ.get("DATAFEST_DATA_DIR", str(DEFAULT_DATA)))
OUT = PROJECT_ROOT / "eda_output"
OUT.mkdir(exist_ok=True)

HAYS_LAT = 38.8782
HAYS_LON = -99.3348
HAYS_RADIUS_MILES = 10
AREA_RADIUS_MILES = 10
MIN_AREA_PATIENTS = 250
MIN_AREA_SCREENED = 25

TRANSPORT_QUESTIONS = [
    "In the past 12 months, has lack of transportation kept you from medical appointments or from getting medications?",
    "In the past 12 months, has lack of transportation kept you from meetings, work, or from getting things needed for daily living?",
]


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


def rounded_mode(values):
    rounded = pd.Series(values).dropna().round().astype(int)
    return int(rounded.mode().iloc[0]) if not rounded.empty else np.nan


def load_tables():
    patients = pd.read_csv(
        DATA / "patients.csv",
        usecols=["DurableKey", "CensusBlockGroupFipsCode", "OmbRace", "OmbEthnicity"],
    ).rename(columns={"DurableKey": "PatientDurableKey"})
    tiger = pd.read_csv(DATA / "tigercensuscodes.csv", usecols=["GEOID", "CENTLAT", "CENTLON"])
    encounters = pd.read_csv(DATA / "encounters.csv", usecols=["EncounterKey", "PatientDurableKey", "ProviderDurableKey"])
    providers = pd.read_csv(
        DATA / "providers.csv",
        usecols=["DurableKey", "OfficeCity", "OfficePostalCode", "PrimarySpecialty", "PrimaryDepartment", "Type"],
    ).rename(columns={"DurableKey": "ProviderDurableKey"})
    sdoh = pd.read_csv(
        DATA / "social_determinants.csv",
        usecols=["EncounterKey", "PatientDurableKey", "DisplayName", "AnswerText", "Domain"],
    )
    return patients, tiger, encounters, providers, sdoh


def build_kansas_base():
    patients, tiger, encounters, providers, sdoh = load_tables()

    patients["fips"] = pd.to_numeric(
        patients["CensusBlockGroupFipsCode"].replace("*Unspecified", np.nan),
        errors="coerce",
    ).astype("Int64")
    ks_patients = patients[patients["fips"].astype(str).str.startswith("20", na=False)].copy()
    ks_patients = ks_patients.merge(tiger, left_on="fips", right_on="GEOID", how="left")
    ks_patients = ks_patients.dropna(subset=["CENTLAT", "CENTLON"]).copy()
    ks_patients["distance_to_hays"] = haversine_miles(
        ks_patients["CENTLAT"].to_numpy(),
        ks_patients["CENTLON"].to_numpy(),
        np.full(len(ks_patients), HAYS_LAT),
        np.full(len(ks_patients), HAYS_LON),
    )

    providers["OfficeCityClean"] = (
        providers["OfficeCity"].astype("string").str.strip().str.upper()
    )
    providers["zip5"] = providers["OfficePostalCode"].map(zip5)
    providers = providers.dropna(subset=["zip5"]).copy()

    cached_geo_path = OUT / "kansas_provider_destination_summary.csv"
    cached_geo = pd.DataFrame()
    if cached_geo_path.exists():
        cached_geo = pd.read_csv(
            cached_geo_path,
            usecols=["zip5", "provider_city", "provider_state", "provider_lat", "provider_lon"],
        ).drop_duplicates(subset=["zip5"])
        cached_geo["zip5"] = cached_geo["zip5"].astype(str)

    need_zips = sorted(set(providers["zip5"].dropna().astype(str)) - set(cached_geo["zip5"].astype(str)) if not cached_geo.empty else set(providers["zip5"].dropna().astype(str)))
    geo_rows = []
    for z in need_zips:
        item = geocode_zip(z)
        if item and item["provider_state"] == "KS":
            geo_rows.append(item)
    fresh_geo = pd.DataFrame(geo_rows)
    zip_geo = pd.concat([cached_geo, fresh_geo], ignore_index=True).drop_duplicates(subset=["zip5"])

    provider_geo = providers.merge(zip_geo, on="zip5", how="inner")

    encounter_geo = (
        encounters.merge(
            ks_patients[
                ["PatientDurableKey", "CENTLAT", "CENTLON", "distance_to_hays", "fips"]
            ],
            on="PatientDurableKey",
            how="inner",
        )
        .merge(
            provider_geo[
                [
                    "ProviderDurableKey",
                    "zip5",
                    "OfficeCityClean",
                    "provider_city",
                    "provider_lat",
                    "provider_lon",
                    "PrimarySpecialty",
                    "PrimaryDepartment",
                    "Type",
                ]
            ],
            on="ProviderDurableKey",
            how="inner",
        )
        .copy()
    )
    encounter_geo["distance_to_provider"] = haversine_miles(
        encounter_geo["CENTLAT"].to_numpy(),
        encounter_geo["CENTLON"].to_numpy(),
        encounter_geo["provider_lat"].to_numpy(),
        encounter_geo["provider_lon"].to_numpy(),
    )

    transport = sdoh[sdoh["DisplayName"].isin(TRANSPORT_QUESTIONS)].copy()
    transport["screened"] = 1
    transport["positive_issue"] = (transport["AnswerText"] == "Yes").astype(int)
    patient_transport = (
        transport.groupby("PatientDurableKey", as_index=False)
        .agg(
            transport_screen_count=("screened", "sum"),
            transport_positive_count=("positive_issue", "sum"),
        )
    )
    patient_transport["screened_any"] = (patient_transport["transport_screen_count"] > 0).astype(int)
    patient_transport["positive_any"] = (patient_transport["transport_positive_count"] > 0).astype(int)

    return ks_patients, provider_geo, encounter_geo, patient_transport


def summarize_city_areas(ks_patients, encounter_geo, patient_transport):
    city_centers = (
        encounter_geo.groupby("provider_city", as_index=False)
        .agg(
            provider_lat=("provider_lat", "mean"),
            provider_lon=("provider_lon", "mean"),
            encounter_count=("EncounterKey", "count"),
        )
        .query("provider_city == provider_city")
        .sort_values("encounter_count", ascending=False)
    )

    area_rows = []
    for row in city_centers.itertuples():
        nearby = ks_patients[
            haversine_miles(
                ks_patients["CENTLAT"].to_numpy(),
                ks_patients["CENTLON"].to_numpy(),
                np.full(len(ks_patients), row.provider_lat),
                np.full(len(ks_patients), row.provider_lon),
            )
            <= AREA_RADIUS_MILES
        ].copy()
        if nearby["PatientDurableKey"].nunique() < MIN_AREA_PATIENTS:
            continue
        patient_ids = nearby["PatientDurableKey"].drop_duplicates()
        enc = encounter_geo[encounter_geo["PatientDurableKey"].isin(patient_ids)]
        transport = patient_transport[patient_transport["PatientDurableKey"].isin(patient_ids)]
        screened = int(transport["screened_any"].sum())
        positive = int(transport["positive_any"].sum())
        if screened < MIN_AREA_SCREENED:
            continue
        mean_dist = float(enc["distance_to_provider"].mean()) if len(enc) else np.nan
        median_dist = float(enc["distance_to_provider"].median()) if len(enc) else np.nan
        mode_dist = rounded_mode(enc["distance_to_provider"])
        positive_rate = positive / screened if screened else np.nan
        far_share = float((enc["distance_to_provider"] >= 20).mean()) if len(enc) else np.nan
        score = mean_dist * positive_rate * math.log1p(nearby["PatientDurableKey"].nunique())
        area_rows.append(
            {
                "area_city": row.provider_city,
                "area_patients": int(nearby["PatientDurableKey"].nunique()),
                "area_encounters": int(len(enc)),
                "screened_patients": screened,
                "positive_transport_patients": positive,
                "positive_rate_screened": round(positive_rate, 4),
                "mean_distance": round(mean_dist, 2),
                "median_distance": round(median_dist, 2),
                "mode_distance_rounded": mode_dist,
                "far_20mi_share": round(far_share, 4),
                "provider_lat": row.provider_lat,
                "provider_lon": row.provider_lon,
                "opportunity_score": round(score, 2),
            }
        )
    areas = pd.DataFrame(area_rows).sort_values(
        ["opportunity_score", "mean_distance", "positive_rate_screened"],
        ascending=[False, False, False],
    )
    return areas


def main():
    ks_patients, provider_geo, encounter_geo, patient_transport = build_kansas_base()

    hays_patients = ks_patients[ks_patients["distance_to_hays"] <= HAYS_RADIUS_MILES].copy()
    hays_ids = hays_patients["PatientDurableKey"].drop_duplicates()
    hays_encounters = encounter_geo[encounter_geo["PatientDurableKey"].isin(hays_ids)].copy()
    hays_transport = patient_transport[patient_transport["PatientDurableKey"].isin(hays_ids)].copy()

    encounter_counts = hays_encounters.groupby("PatientDurableKey").size()
    total_hays_patients = int(hays_ids.nunique())
    total_hays_encounters = int(len(hays_encounters))
    mean_encounters = float(encounter_counts.mean()) if len(encounter_counts) else np.nan
    median_encounters = float(encounter_counts.median()) if len(encounter_counts) else np.nan
    screened_patients = int(hays_transport["screened_any"].sum())
    positive_patients = int(hays_transport["positive_any"].sum())
    positive_rate_screened = positive_patients / screened_patients if screened_patients else np.nan
    positive_rate_all = positive_patients / total_hays_patients if total_hays_patients else np.nan

    hays_mean_distance = float(hays_encounters["distance_to_provider"].mean())
    hays_median_distance = float(hays_encounters["distance_to_provider"].median())
    hays_mode_distance = rounded_mode(hays_encounters["distance_to_provider"])

    destination_summary = (
        hays_encounters.groupby(["provider_city", "zip5"], as_index=False)
        .agg(
            encounter_count=("EncounterKey", "count"),
            unique_patients=("PatientDurableKey", "nunique"),
            mean_distance=("distance_to_provider", "mean"),
            median_distance=("distance_to_provider", "median"),
            provider_lat=("provider_lat", "first"),
            provider_lon=("provider_lon", "first"),
            top_specialty=("PrimarySpecialty", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
        )
        .sort_values("encounter_count", ascending=False)
    )
    destination_summary["mean_distance"] = destination_summary["mean_distance"].round(2)
    destination_summary["median_distance"] = destination_summary["median_distance"].round(2)
    destination_summary.to_csv(OUT / "hays_transport_destinations.csv", index=False)

    areas = summarize_city_areas(ks_patients, encounter_geo, patient_transport)
    candidate_areas = areas[areas["area_city"].str.upper() != "HAYS"].head(12).copy()
    candidate_areas.to_csv(OUT / "hays_transport_candidate_areas.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "hays_patients", "value": total_hays_patients},
            {"metric": "hays_encounters", "value": total_hays_encounters},
            {"metric": "mean_encounters_per_patient", "value": round(mean_encounters, 2)},
            {"metric": "median_encounters_per_patient", "value": round(median_encounters, 2)},
            {"metric": "screened_patients", "value": screened_patients},
            {"metric": "positive_transport_patients", "value": positive_patients},
            {"metric": "positive_rate_screened_pct", "value": round(positive_rate_screened * 100, 2)},
            {"metric": "positive_rate_all_pct", "value": round(positive_rate_all * 100, 2)},
            {"metric": "mean_distance_miles", "value": round(hays_mean_distance, 2)},
            {"metric": "median_distance_miles", "value": round(hays_median_distance, 2)},
            {"metric": "mode_distance_miles_rounded", "value": hays_mode_distance},
        ]
    )
    summary_df.to_csv(OUT / "hays_transport_summary.csv", index=False)

    home_geo = (
        hays_patients.groupby(["fips", "CENTLAT", "CENTLON"], as_index=False)
        .agg(patient_count=("PatientDurableKey", "nunique"))
    )
    home_geo["heat_weight"] = np.log1p(home_geo["patient_count"])

    hays_overlay = go.Figure()
    hays_overlay.add_trace(
        go.Densitymap(
            lat=home_geo["CENTLAT"],
            lon=home_geo["CENTLON"],
            z=home_geo["heat_weight"],
            radius=18,
            colorscale=[
                [0.0, "rgba(0,48,73,0.08)"],
                [0.25, "rgba(0,119,182,0.28)"],
                [0.5, "rgba(0,180,216,0.5)"],
                [0.75, "rgba(255,183,3,0.75)"],
                [1.0, "rgba(251,86,7,0.95)"],
            ],
            showscale=True,
            colorbar={"title": "Hays-area home density", "x": 0.96},
            name="Hays-area patients",
        )
    )
    top_dest = destination_summary.head(15).copy()
    hays_overlay.add_trace(
        go.Scattermap(
            lat=top_dest["provider_lat"],
            lon=top_dest["provider_lon"],
            mode="markers",
            text=top_dest["provider_city"] + " (" + top_dest["zip5"].astype(str) + ")",
            customdata=np.stack(
                [
                    top_dest["encounter_count"],
                    top_dest["unique_patients"],
                    top_dest["mean_distance"],
                    top_dest["median_distance"],
                    top_dest["top_specialty"].fillna(""),
                ],
                axis=1,
            ),
            marker={
                "size": np.clip(np.log1p(top_dest["encounter_count"]) * 5, 10, 34),
                "color": top_dest["encounter_count"],
                "colorscale": [
                    [0.0, "#6a040f"],
                    [0.25, "#9d0208"],
                    [0.5, "#dc2f02"],
                    [0.75, "#f48c06"],
                    [1.0, "#ffba08"],
                ],
                "opacity": 0.9,
                "showscale": True,
                "colorbar": {"title": "Destination volume", "x": 1.06},
            },
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Encounters: %{customdata[0]:,}<br>"
                "Patients: %{customdata[1]:,}<br>"
                "Mean distance: %{customdata[2]} mi<br>"
                "Median distance: %{customdata[3]} mi<br>"
                "Top specialty: %{customdata[4]}<extra></extra>"
            ),
            name="Destinations used by Hays-area patients",
        )
    )
    hays_overlay.update_layout(
        title="Hays-Area Patients And Their Provider Destinations",
        map={"style": "carto-positron", "center": {"lat": 38.88, "lon": -99.33}, "zoom": 6.2},
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
    )

    candidate_map = go.Figure()
    candidate_map.add_trace(
        go.Scattermap(
            lat=candidate_areas["provider_lat"],
            lon=candidate_areas["provider_lon"],
            mode="markers+text",
            text=candidate_areas["area_city"],
            textposition="top center",
            marker={
                "size": np.clip(np.sqrt(candidate_areas["area_patients"]) / 2.5, 12, 30),
                "color": candidate_areas["mean_distance"],
                "colorscale": [
                    [0.0, "#3a0ca3"],
                    [0.25, "#4361ee"],
                    [0.5, "#4cc9f0"],
                    [0.75, "#f8961e"],
                    [1.0, "#d62828"],
                ],
                "showscale": True,
                "colorbar": {"title": "Mean distance (mi)"},
                "opacity": 0.9,
            },
            customdata=np.stack(
                [
                    candidate_areas["area_patients"],
                    candidate_areas["positive_rate_screened"],
                    candidate_areas["median_distance"],
                    candidate_areas["mode_distance_rounded"],
                    candidate_areas["far_20mi_share"],
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Patients: %{customdata[0]:,}<br>"
                "Transport issue rate (screened): %{customdata[1]:.1%}<br>"
                "Mean distance: %{marker.color:.2f} mi<br>"
                "Median distance: %{customdata[2]:.2f} mi<br>"
                "Mode distance: %{customdata[3]} mi<br>"
                "Share >=20mi: %{customdata[4]:.1%}<extra></extra>"
            ),
            name="Candidate areas",
        )
    )
    candidate_map.update_layout(
        title="Other Kansas Areas With Potential Transportation Risk From Travel Distance",
        map={"style": "carto-positron", "center": {"lat": 38.5, "lon": -98.4}, "zoom": 6},
        margin=dict(l=10, r=10, t=50, b=10),
    )

    encounter_hist = px.histogram(
        encounter_counts.reset_index(name="encounter_count"),
        x="encounter_count",
        nbins=30,
        title="How Often Hays-Area Patients Have Encounters",
        color_discrete_sequence=["#0d6c63"],
    )
    encounter_hist.update_layout(
        bargap=0.04,
        xaxis_title="Encounters per Hays-area patient",
        yaxis_title="Patient count",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    top_dest_table = destination_summary.head(10)[
        ["provider_city", "zip5", "encounter_count", "unique_patients", "mean_distance", "median_distance", "top_specialty"]
    ].to_html(index=False, classes="data-table")

    candidate_table = candidate_areas[
        [
            "area_city",
            "area_patients",
            "positive_rate_screened",
            "mean_distance",
            "median_distance",
            "mode_distance_rounded",
            "far_20mi_share",
        ]
    ].copy()
    candidate_table["positive_rate_screened"] = (candidate_table["positive_rate_screened"] * 100).round(1)
    candidate_table["far_20mi_share"] = (candidate_table["far_20mi_share"] * 100).round(1)
    candidate_table = candidate_table.rename(
        columns={
            "area_city": "Area",
            "area_patients": "Patients",
            "positive_rate_screened": "Transport Issue Rate %",
            "mean_distance": "Mean Distance",
            "median_distance": "Median Distance",
            "mode_distance_rounded": "Mode Distance",
            "far_20mi_share": "Share >=20mi %",
        }
    ).to_html(index=False, classes="data-table")

    insights = [
        f"Hays-area patients are approximated as Kansas patients whose home census block group centroid falls within {HAYS_RADIUS_MILES} miles of Hays ZIP 67601.",
        f"These patients generated {total_hays_encounters:,} encounters across {total_hays_patients:,} unique patients, with a mean of {mean_encounters:.2f} and a median of {median_encounters:.2f} encounters per patient.",
        f"{screened_patients:,} Hays-area patients had at least one transportation screening, and {positive_patients:,} of them reported a transportation issue at least once.",
        f"That is {positive_rate_screened:.1%} of screened Hays-area patients, or {positive_rate_all:.1%} of all inferred Hays-area patients.",
        f"Hays-area travel distance has a mean of {hays_mean_distance:.2f} miles, a median of {hays_median_distance:.2f} miles, and a rounded mode of {hays_mode_distance} mile.",
        f"Potential comparison areas are ranked using a combination of patient volume, transportation-issue rate, and travel distance; the strongest signals outside Hays are {', '.join(candidate_areas['area_city'].head(5).tolist())}.",
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hays Transportation And Provider Access</title>
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
    code {{ color:var(--accent); }}
    @media (max-width: 1100px) {{
      .grid, .metrics {{ grid-template-columns:1fr; }}
      .card {{ padding:20px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Hays Transportation And Provider Access</h1>
      <p>
        This analysis uses the provided raw files to estimate how often Hays-area patients use encounters, whether they
        reported transportation issues, and which other Kansas areas show a similar risk pattern. Because <code>patients.csv</code>
        does not include a city field, Hays is inferred using TIGER census block group centroids within {HAYS_RADIUS_MILES} miles
        of Hays, Kansas ZIP 67601.
      </p>
      <div class="metrics">
        <div class="mini"><span>Hays-Area Patients</span><strong>{total_hays_patients:,}</strong></div>
        <div class="mini"><span>Hays-Area Encounters</span><strong>{total_hays_encounters:,}</strong></div>
        <div class="mini"><span>Mean Encounters</span><strong>{mean_encounters:.2f}</strong></div>
        <div class="mini"><span>Transport Issue Rate</span><strong>{positive_rate_screened:.1%}</strong></div>
        <div class="mini"><span>Median Distance</span><strong>{hays_median_distance:.1f} mi</strong></div>
      </div>
    </section>

    <section class="grid">
      <div class="card">{hays_overlay.to_html(full_html=False, include_plotlyjs='cdn')}</div>
      <div class="card">{candidate_map.to_html(full_html=False, include_plotlyjs=False)}</div>
    </section>

    <section class="grid">
      <div class="card">{encounter_hist.to_html(full_html=False, include_plotlyjs=False)}</div>
      <div class="card">
        <h2>Key Findings</h2>
        <ul>{''.join(f'<li>{item}</li>' for item in insights)}</ul>
        <p><strong>Mean distance:</strong> {hays_mean_distance:.2f} miles</p>
        <p><strong>Median distance:</strong> {hays_median_distance:.2f} miles</p>
        <p><strong>Mode distance:</strong> {hays_mode_distance} mile (rounded)</p>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Where Hays-Area Patients Go Most Often</h2>
        {top_dest_table}
      </div>
      <div class="card">
        <h2>Other Areas To Watch</h2>
        <p>
          These areas combine a larger travel burden with a meaningful share of screened patients reporting transportation problems.
          They are intended as candidate locations for deeper access review, not final causal claims.
        </p>
        {candidate_table}
      </div>
    </section>
  </div>
</body>
</html>
"""
    (OUT / "hays_transport_access.html").write_text(html, encoding="utf-8")
    print("saved -> hays_transport_summary.csv")
    print("saved -> hays_transport_destinations.csv")
    print("saved -> hays_transport_candidate_areas.csv")
    print("saved -> hays_transport_access.html")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
