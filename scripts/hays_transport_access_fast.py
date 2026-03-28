"""
Fast Hays-area transportation and provider access analysis.

Outputs:
- eda_output/hays_transport_access.html
- eda_output/hays_transport_destinations.csv
- eda_output/hays_transport_candidate_areas.csv
- eda_output/hays_transport_summary.csv
"""

import math
import os
import re
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


def assign_nearest_city(patient_df, city_centers):
    city_names = city_centers["provider_city"].tolist()
    city_lats = city_centers["provider_lat"].to_numpy()
    city_lons = city_centers["provider_lon"].to_numpy()
    p_lats = patient_df["CENTLAT"].to_numpy()
    p_lons = patient_df["CENTLON"].to_numpy()

    distance_matrix = np.vstack(
        [
            haversine_miles(
                p_lats,
                p_lons,
                np.full(len(patient_df), city_lats[i]),
                np.full(len(patient_df), city_lons[i]),
            )
            for i in range(len(city_centers))
        ]
    ).T
    nearest_idx = distance_matrix.argmin(axis=1)
    nearest_dist = distance_matrix[np.arange(len(patient_df)), nearest_idx]
    patient_df = patient_df.copy()
    patient_df["area_city"] = [city_names[i] for i in nearest_idx]
    patient_df["area_city_distance"] = nearest_dist
    patient_df.loc[patient_df["area_city_distance"] > AREA_RADIUS_MILES, "area_city"] = pd.NA
    return patient_df


def main():
    cached_provider_summary = pd.read_csv(
        OUT / "kansas_provider_destination_summary.csv",
        usecols=["zip5", "provider_city", "provider_lat", "provider_lon", "encounter_count"],
    ).drop_duplicates(subset=["zip5"])
    cached_provider_summary["zip5"] = cached_provider_summary["zip5"].astype(str)

    providers = pd.read_csv(
        DATA / "providers.csv",
        usecols=["DurableKey", "OfficePostalCode", "PrimarySpecialty", "PrimaryDepartment", "Type"],
    ).rename(columns={"DurableKey": "ProviderDurableKey"})
    providers["zip5"] = providers["OfficePostalCode"].map(zip5)
    provider_map = providers.merge(cached_provider_summary, on="zip5", how="inner")

    city_centers = (
        cached_provider_summary.groupby("provider_city", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "provider_lat": np.average(g["provider_lat"], weights=g["encounter_count"]),
                    "provider_lon": np.average(g["provider_lon"], weights=g["encounter_count"]),
                    "encounter_count": g["encounter_count"].sum(),
                }
            )
        )
        .reset_index(drop=True)
        .sort_values("encounter_count", ascending=False)
    )
    city_centers = city_centers[city_centers["encounter_count"] >= 5000].copy()

    patients = pd.read_csv(
        DATA / "patients.csv",
        usecols=["DurableKey", "CensusBlockGroupFipsCode", "OmbRace", "OmbEthnicity"],
    ).rename(columns={"DurableKey": "PatientDurableKey"})
    tiger = pd.read_csv(DATA / "tigercensuscodes.csv", usecols=["GEOID", "CENTLAT", "CENTLON"])
    patients["fips"] = pd.to_numeric(
        patients["CensusBlockGroupFipsCode"].replace("*Unspecified", np.nan),
        errors="coerce",
    ).astype("Int64")
    ks_patients = patients[patients["fips"].astype(str).str.startswith("20", na=False)].copy()
    ks_patients = ks_patients.merge(tiger, left_on="fips", right_on="GEOID", how="left").dropna(subset=["CENTLAT", "CENTLON"])
    ks_patients["distance_to_hays"] = haversine_miles(
        ks_patients["CENTLAT"].to_numpy(),
        ks_patients["CENTLON"].to_numpy(),
        np.full(len(ks_patients), HAYS_LAT),
        np.full(len(ks_patients), HAYS_LON),
    )
    ks_patients["is_hays"] = ks_patients["distance_to_hays"] <= HAYS_RADIUS_MILES
    ks_patients = assign_nearest_city(ks_patients, city_centers)

    transport_parts = []
    for chunk in pd.read_csv(
        DATA / "social_determinants.csv",
        usecols=["PatientDurableKey", "DisplayName", "AnswerText"],
        chunksize=400000,
    ):
        chunk = chunk[chunk["DisplayName"].isin(TRANSPORT_QUESTIONS)]
        if chunk.empty:
            continue
        part = (
            chunk.assign(
                screened=1,
                positive_issue=(chunk["AnswerText"] == "Yes").astype(int),
            )
            .groupby("PatientDurableKey", as_index=False)[["screened", "positive_issue"]]
            .sum()
        )
        transport_parts.append(part)
    transport = pd.concat(transport_parts, ignore_index=True).groupby("PatientDurableKey", as_index=False).sum()
    transport["screened_any"] = (transport["screened"] > 0).astype(int)
    transport["positive_any"] = (transport["positive_issue"] > 0).astype(int)

    patient_map = ks_patients[
        ["PatientDurableKey", "CENTLAT", "CENTLON", "is_hays", "area_city"]
    ].drop_duplicates()
    patient_map = patient_map.merge(transport[["PatientDurableKey", "screened_any", "positive_any"]], on="PatientDurableKey", how="left")
    patient_map[["screened_any", "positive_any"]] = patient_map[["screened_any", "positive_any"]].fillna(0).astype(int)

    hays_encounter_parts = []
    patient_agg_parts = []
    for chunk in pd.read_csv(
        DATA / "encounters.csv",
        usecols=["EncounterKey", "PatientDurableKey", "ProviderDurableKey"],
        chunksize=600000,
    ):
        merged = (
            chunk.merge(patient_map, on="PatientDurableKey", how="inner")
            .merge(
                provider_map[
                    [
                        "ProviderDurableKey",
                        "provider_city",
                        "zip5",
                        "provider_lat",
                        "provider_lon",
                        "PrimarySpecialty",
                    ]
                ],
                on="ProviderDurableKey",
                how="inner",
            )
        )
        if merged.empty:
            continue
        merged["distance_to_provider"] = haversine_miles(
            merged["CENTLAT"].to_numpy(),
            merged["CENTLON"].to_numpy(),
            merged["provider_lat"].to_numpy(),
            merged["provider_lon"].to_numpy(),
        )

        hays_subset = merged[merged["is_hays"]].copy()
        if not hays_subset.empty:
            hays_encounter_parts.append(
                hays_subset[
                    [
                        "EncounterKey",
                        "PatientDurableKey",
                        "provider_city",
                        "zip5",
                        "distance_to_provider",
                        "provider_lat",
                        "provider_lon",
                        "PrimarySpecialty",
                    ]
                ]
            )

        area_subset = merged.dropna(subset=["area_city"]).copy()
        if not area_subset.empty:
            agg = (
                area_subset.groupby(["PatientDurableKey", "area_city"], as_index=False)
                .agg(
                    encounter_count=("EncounterKey", "count"),
                    distance_sum=("distance_to_provider", "sum"),
                )
            )
            patient_agg_parts.append(agg)

    hays_encounters = pd.concat(hays_encounter_parts, ignore_index=True)
    patient_enc_agg = pd.concat(patient_agg_parts, ignore_index=True).groupby(
        ["PatientDurableKey", "area_city"], as_index=False
    ).sum()
    patient_enc_agg["patient_mean_distance"] = patient_enc_agg["distance_sum"] / patient_enc_agg["encounter_count"]

    hays_patients = patient_map[patient_map["is_hays"]].copy()
    total_hays_patients = int(hays_patients["PatientDurableKey"].nunique())
    hays_patient_enc = (
        hays_encounters.groupby("PatientDurableKey").size().rename("encounter_count").reset_index()
    )
    mapped_hays_patients = int(hays_patient_enc["PatientDurableKey"].nunique())
    total_hays_encounters = int(len(hays_encounters))
    mean_encounters = float(hays_patient_enc["encounter_count"].mean())
    median_encounters = float(hays_patient_enc["encounter_count"].median())
    screened_patients = int(hays_patients["screened_any"].sum())
    positive_patients = int(hays_patients["positive_any"].sum())
    positive_rate_screened = positive_patients / screened_patients if screened_patients else np.nan
    positive_rate_all = positive_patients / total_hays_patients if total_hays_patients else np.nan
    hays_mean_distance = float(hays_encounters["distance_to_provider"].mean())
    hays_median_distance = float(hays_encounters["distance_to_provider"].median())
    hays_mode_distance = rounded_mode(hays_encounters["distance_to_provider"])

    hays_destinations = (
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
    hays_destinations["mean_distance"] = hays_destinations["mean_distance"].round(2)
    hays_destinations["median_distance"] = hays_destinations["median_distance"].round(2)
    hays_destinations.to_csv(OUT / "hays_transport_destinations.csv", index=False)

    area_patient_stats = (
        patient_map.dropna(subset=["area_city"])
        .groupby("area_city", as_index=False)
        .agg(
            area_patients=("PatientDurableKey", "nunique"),
            screened_patients=("screened_any", "sum"),
            positive_transport_patients=("positive_any", "sum"),
        )
    )
    area_enc_stats = (
        patient_enc_agg.groupby("area_city", as_index=False)
        .agg(
            area_encounters=("encounter_count", "sum"),
            mean_distance=("patient_mean_distance", "mean"),
            median_distance=("patient_mean_distance", "median"),
            mode_distance_rounded=("patient_mean_distance", rounded_mode),
            far_20mi_share=("patient_mean_distance", lambda s: float((s >= 20).mean())),
        )
    )
    area_stats = area_patient_stats.merge(area_enc_stats, on="area_city", how="inner")
    area_stats["positive_rate_screened"] = area_stats["positive_transport_patients"] / area_stats["screened_patients"]
    area_stats = area_stats.merge(city_centers[["provider_city", "provider_lat", "provider_lon"]], left_on="area_city", right_on="provider_city", how="left").drop(columns=["provider_city"])
    area_stats["opportunity_score"] = (
        area_stats["mean_distance"] * area_stats["positive_rate_screened"] * np.log1p(area_stats["area_patients"])
    )
    area_stats = area_stats[
        (area_stats["area_patients"] >= MIN_AREA_PATIENTS) & (area_stats["screened_patients"] >= MIN_AREA_SCREENED)
    ].sort_values(["opportunity_score", "mean_distance", "positive_rate_screened"], ascending=[False, False, False])
    candidate_areas = area_stats[area_stats["area_city"].str.upper() != "HAYS"].head(12).copy()
    candidate_areas.to_csv(OUT / "hays_transport_candidate_areas.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "hays_patients", "value": total_hays_patients},
            {"metric": "mapped_hays_patients", "value": mapped_hays_patients},
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
        hays_patients.groupby(["CENTLAT", "CENTLON"], as_index=False)
        .agg(patient_count=("PatientDurableKey", "nunique"))
    )
    home_geo["heat_weight"] = np.log1p(home_geo["patient_count"])

    hays_map = go.Figure()
    hays_map.add_trace(
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
            colorbar={"title": "Hays home density", "x": 0.96},
            name="Hays-area patients",
        )
    )
    top_dest = hays_destinations.head(12)
    hays_map.add_trace(
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
            name="Provider destinations",
        )
    )
    hays_map.update_layout(
        title="Hays-Area Patients And Their Provider Destinations",
        map={"style": "carto-positron", "center": {"lat": 38.88, "lon": -99.33}, "zoom": 6.2},
        margin=dict(l=10, r=10, t=50, b=10),
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
                "opacity": 0.92,
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
        )
    )
    candidate_map.update_layout(
        title="Other Kansas Areas With Potential Transportation Risk",
        map={"style": "carto-positron", "center": {"lat": 38.6, "lon": -98.2}, "zoom": 6},
        margin=dict(l=10, r=10, t=50, b=10),
    )

    encounter_hist = px.histogram(
        hays_patient_enc,
        x="encounter_count",
        nbins=28,
        title="How Often Hays-Area Patients Have Encounters",
        color_discrete_sequence=["#0d6c63"],
    )
    encounter_hist.update_layout(
        bargap=0.04,
        xaxis_title="Encounters per Hays-area patient",
        yaxis_title="Patient count",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    top_dest_table = hays_destinations.head(10)[
        ["provider_city", "zip5", "encounter_count", "unique_patients", "mean_distance", "median_distance", "top_specialty"]
    ].to_html(index=False, classes="data-table")

    candidate_table = candidate_areas[
        ["area_city", "area_patients", "positive_rate_screened", "mean_distance", "median_distance", "mode_distance_rounded", "far_20mi_share"]
    ].copy()
    candidate_table["positive_rate_screened"] = (candidate_table["positive_rate_screened"] * 100).round(1)
    candidate_table["far_20mi_share"] = (candidate_table["far_20mi_share"] * 100).round(1)
    candidate_table.columns = ["Area", "Patients", "Transport Issue Rate %", "Mean Distance", "Median Distance", "Mode Distance", "Share >=20mi %"]
    candidate_table_html = candidate_table.to_html(index=False, classes="data-table")

    insights = [
        f"Hays-area patients are approximated as Kansas patients whose TIGER census block group centroid is within {HAYS_RADIUS_MILES} miles of Hays ZIP 67601.",
        f"Of the {total_hays_patients:,} inferred Hays-area patients, {mapped_hays_patients:,} had mapped Kansas provider-location encounters in this analysis window.",
        f"Those mapped Hays-area patients generated {total_hays_encounters:,} encounters, with a mean of {mean_encounters:.2f} and median of {median_encounters:.2f} encounters per patient.",
        f"{screened_patients:,} inferred Hays-area patients had at least one transportation screening and {positive_patients:,} reported a transportation issue at least once.",
        f"That is {positive_rate_screened:.1%} of screened Hays-area patients, or {positive_rate_all:.1%} of all inferred Hays-area patients, so the Hays transportation finding should be treated as sparse-data evidence.",
        f"Hays-area travel distance has a mean of {hays_mean_distance:.2f} miles, a median of {hays_median_distance:.2f} miles, and a rounded mode of {hays_mode_distance} mile.",
        f"Based on travel burden plus transportation issue rate, the strongest comparison areas outside Hays are {', '.join(candidate_areas['area_city'].head(5).tolist())}.",
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hays Transportation And Provider Access</title>
  <style>
    :root {{ --bg:#f6f1e8; --panel:rgba(255,251,245,.94); --ink:#1f2d2f; --muted:#607074; --accent:#0d6c63; --line:rgba(31,45,47,.14); }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; font-family:"Avenir Next","Segoe UI",sans-serif; color:var(--ink);
      background: radial-gradient(circle at top left, rgba(196,110,65,.14), transparent 28%), radial-gradient(circle at top right, rgba(13,108,99,.12), transparent 26%), linear-gradient(180deg,#f8f3eb 0%,#efe4d7 100%);
    }}
    .wrap {{ width:min(1360px, calc(100vw - 28px)); margin:0 auto; padding:28px 0 42px; }}
    .card {{ background:var(--panel); border:1px solid color-mix(in srgb, white 55%, var(--line)); border-radius:28px; padding:28px; margin-bottom:18px; box-shadow:0 18px 45px rgba(25,35,38,.10); backdrop-filter:blur(10px); }}
    h1,h2 {{ margin:0 0 12px; line-height:1.08; }} h1 {{ font-size:clamp(34px,5vw,58px); }} h2 {{ font-size:28px; }}
    p {{ color:var(--muted); margin:0 0 12px; line-height:1.55; }}
    .metrics {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-top:18px; }}
    .mini {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:rgba(255,255,255,.5); }}
    .mini span {{ display:block; font-size:12px; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); margin-bottom:6px; }}
    .mini strong {{ display:block; font-size:28px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    ul {{ margin:0; padding-left:20px; }} li {{ margin-bottom:8px; }}
    .data-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .data-table th,.data-table td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
    code {{ color:var(--accent); }}
    @media (max-width: 1100px) {{ .grid, .metrics {{ grid-template-columns:1fr; }} .card {{ padding:20px; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Hays Transportation And Provider Access</h1>
      <p>
        This analysis uses the provided raw files to estimate how often Hays-area patients use encounters, whether they
        reported transportation issues, and which other Kansas areas show a similar access pattern. Because <code>patients.csv</code>
        does not include a city field, Hays is inferred from Kansas patient census block group centroids within {HAYS_RADIUS_MILES} miles of Hays ZIP 67601.
      </p>
        <div class="metrics">
          <div class="mini"><span>Hays-Area Patients</span><strong>{total_hays_patients:,}</strong></div>
          <div class="mini"><span>Mapped Hays Patients</span><strong>{mapped_hays_patients:,}</strong></div>
          <div class="mini"><span>Mapped Encounters</span><strong>{total_hays_encounters:,}</strong></div>
          <div class="mini"><span>Mean Encounters</span><strong>{mean_encounters:.2f}</strong></div>
          <div class="mini"><span>Transport Issue Rate</span><strong>{positive_rate_screened:.1%}</strong></div>
          <div class="mini"><span>Median Distance</span><strong>{hays_median_distance:.1f} mi</strong></div>
        </div>
    </section>
    <section class="grid">
      <div class="card">{hays_map.to_html(full_html=False, include_plotlyjs='cdn')}</div>
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
      <div class="card"><h2>Where Hays-Area Patients Go Most Often</h2>{top_dest_table}</div>
      <div class="card"><h2>Other Areas To Watch</h2><p>These areas combine larger travel burden with a meaningful share of screened patients reporting transportation problems.</p>{candidate_table_html}</div>
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
