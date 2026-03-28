"""
Interactive association analysis for combinedsocialencounter.csv.

Outputs:
- eda_output/combinedsocial_corr_matrix.csv
- eda_output/combinedsocial_corr_matrix.png
- eda_output/combinedsocial_scatter_matrix.html
- eda_output/combinedsocial_analysis_dashboard.html
- eda_output/combinedsocial_top_correlations.csv

The source file is long-format: one row per question-response tied to an
encounter. This script pivots key questions into encounter-level numeric
features, adds diagnosis-group indicators, then builds:
1. an annotated correlation matrix
2. an interactive scatter matrix
"""

import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT = os.path.join(PROJECT_ROOT, "eda_output")
os.makedirs(OUT, exist_ok=True)

DEFAULT_FILE = r"C:\Users\ezeki\Downloads\combinedsocialencounter.csv"
SOURCE_FILE = os.environ.get("COMBINEDSOCIAL_FILE", DEFAULT_FILE)


QUESTION_SPECS = {
    "In the past 12 months, has lack of transportation kept you from medical appointments or from getting medications?": {
        "feature": "transport_medical_barrier",
        "map": {"No": 0, "Yes": 1},
    },
    "In the past 12 months, has lack of transportation kept you from meetings, work, or from getting things needed for daily living?": {
        "feature": "transport_daily_barrier",
        "map": {"No": 0, "Yes": 1},
    },
    "Within the past 12 months, you worried that your food would run out before you got the money to buy more.": {
        "feature": "food_worry_risk",
        "map": {"Never true": 0, "Sometimes true": 1, "Often true": 2},
    },
    "Within the past 12 months, the food you bought just didn't last and you didn't have money to get more.": {
        "feature": "food_notlast_risk",
        "map": {"Never true": 0, "Sometimes true": 1, "Often true": 2},
    },
    "In the past 12 months has the electric, gas, oil, or water company threatened to shut off services in your home?": {
        "feature": "utility_shutoff_risk",
        "map": {"No": 0, "Yes": 1, "Already shut off": 2},
    },
    "In the last 12 months, was there a time when you were not able to pay the mortgage or rent on time?": {
        "feature": "rent_payment_problem",
        "map": {"No": 0, "Yes": 1},
    },
    "Q1: How often do you have a drink containing alcohol?": {
        "feature": "alcohol_frequency",
        "map": {
            "Never": 0,
            "Monthly or less": 1,
            "2-4 times a month": 2,
            "2-3 times a week": 3,
            "4 or more times a week": 4,
        },
    },
    "Q2: How many drinks containing alcohol do you have on a typical day when you are drinking?": {
        "feature": "drinks_per_day",
        "map": {
            "Patient does not drink": 0,
            "1 or 2": 1,
            "3 or 4": 2,
            "5 or 6": 3,
            "7 to 9": 4,
            "10 or more": 5,
        },
    },
    "Q3: How often do you have six or more drinks on one occasion?": {
        "feature": "binge_frequency",
        "map": {
            "Never": 0,
            "Less than monthly": 1,
            "Monthly": 2,
            "Weekly": 3,
            "Daily or almost daily": 4,
        },
    },
    "Do you feel stress - tense, restless, nervous, or anxious, or unable to sleep at night because your mind is troubled all the time - these days?": {
        "feature": "stress_level",
        "map": {
            "Not at all": 0,
            "Only a little": 1,
            "To some extent": 2,
            "Rather much": 3,
            "Very much": 4,
        },
    },
    "How hard is it for you to pay for the very basics like food, housing, medical care, and heating?": {
        "feature": "financial_strain",
        "map": {
            "Not hard at all": 0,
            "Not very hard": 1,
            "Somewhat hard": 2,
            "Hard": 3,
            "Very hard": 4,
        },
    },
    "On average, how many days per week do you engage in moderate to strenuous exercise (like a brisk walk)?": {
        "feature": "exercise_days",
        "parser": "days",
    },
    "On average, how many minutes do you engage in exercise at this level?": {
        "feature": "exercise_minutes",
        "parser": "minutes",
    },
    "At any time in the past 12 months, were you homeless or living in a shelter (including now)?": {
        "feature": "recent_homelessness",
        "map": {"No": 0, "Yes": 1},
    },
}

QUESTION_SET = set(QUESTION_SPECS.keys())
SCATTER_SAMPLE = 2500
RED_WHITE_GREEN = LinearSegmentedColormap.from_list(
    "red_white_green",
    ["#b2182b", "#f7f7f7", "#1b7837"],
    N=256,
)


def parse_days(text):
    if pd.isna(text):
        return np.nan
    match = re.search(r"(\d+)", str(text))
    return float(match.group(1)) if match else np.nan


def parse_minutes(text):
    if pd.isna(text):
        return np.nan
    match = re.search(r"(\d+)", str(text))
    return float(match.group(1)) if match else np.nan


def encode_answer(question, answers):
    spec = QUESTION_SPECS[question]
    answers = answers.astype("string").str.strip()
    invalid = answers.isna() | answers.str.lower().isin({"nan", "none"}) | answers.isin(
        ["Patient declined", "Patient unable to answer", "*Unknown"]
    )
    if "map" in spec:
        out = answers.map(spec["map"])
    elif spec["parser"] == "days":
        out = answers.map(parse_days)
    else:
        out = answers.map(parse_minutes)
    out = pd.to_numeric(out, errors="coerce")
    out[invalid] = np.nan
    return out


def slugify(text):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).lower()).strip("_")
    return re.sub(r"_+", "_", text)


def load_long_data():
    pieces = []
    usecols = ["DisplayName", "AnswerText", "EncounterKey", "GroupName"]
    for chunk in pd.read_csv(SOURCE_FILE, usecols=usecols, chunksize=250000):
        chunk = chunk[chunk["DisplayName"].isin(QUESTION_SET)].copy()
        if not chunk.empty:
            pieces.append(chunk)
    df = pd.concat(pieces, ignore_index=True)
    return df


def build_feature_table(long_df):
    feature_rows = []
    for question, spec in QUESTION_SPECS.items():
        sub = long_df.loc[long_df["DisplayName"] == question, ["EncounterKey", "AnswerText"]].copy()
        sub["feature"] = spec["feature"]
        sub["value"] = encode_answer(question, sub["AnswerText"])
        sub = sub.dropna(subset=["value"])
        feature_rows.append(sub[["EncounterKey", "feature", "value"]])

    stacked = pd.concat(feature_rows, ignore_index=True)
    wide = stacked.pivot_table(
        index="EncounterKey",
        columns="feature",
        values="value",
        aggfunc="first",
    )

    encounter_dx = (
        long_df[["EncounterKey", "GroupName"]]
        .dropna(subset=["GroupName"])
        .drop_duplicates()
        .groupby("EncounterKey")["GroupName"]
        .first()
    )
    wide = wide.join(encounter_dx.rename("GroupName"), how="left")

    top_groups = (
        wide["GroupName"]
        .dropna()
        .value_counts()
        .head(5)
        .index
        .tolist()
    )
    for group in top_groups:
        feature_name = f"dx_{slugify(group)[:36]}"
        wide[feature_name] = (wide["GroupName"] == group).astype(int)

    wide["any_dx_recorded"] = wide["GroupName"].notna().astype(int)
    return wide, top_groups


def compute_corr(df_numeric):
    return df_numeric.corr(method="spearman").round(3)


def top_pairs(corr):
    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            val = corr.loc[left, right]
            if pd.notna(val):
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "correlation": float(val),
                        "abs_correlation": abs(float(val)),
                    }
                )
    result = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)
    return result


def save_static_heatmap(corr):
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=RED_WHITE_GREEN,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        annot_kws={"size": 8},
        cbar_kws={"label": "Spearman correlation"},
    )
    plt.title("Combined Social Encounter Correlation Matrix", pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "combinedsocial_corr_matrix.png"), dpi=170, bbox_inches="tight")
    plt.close()


def build_interactive_heatmap(corr):
    labels = list(corr.columns)
    text = corr.applymap(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#b2182b"],
                [0.5, "#f7f7f7"],
                [1.0, "#1b7837"],
            ],
            zmin=-1,
            zmax=1,
            text=text.values,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar={"title": "Spearman r"},
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r=%{z:.3f}<extra></extra>",
        )
    )
    heatmap.update_layout(
        title="Interactive Correlation Matrix",
        width=1150,
        height=950,
        xaxis={"tickangle": 45, "side": "bottom"},
        yaxis={"autorange": "reversed"},
        margin={"l": 180, "r": 40, "t": 80, "b": 180},
    )
    return heatmap


def build_scatter_matrix(df_numeric):
    plot_df = df_numeric.dropna().copy()
    if len(plot_df) > SCATTER_SAMPLE:
        plot_df = plot_df.sample(SCATTER_SAMPLE, random_state=42)

    fig = px.scatter_matrix(
        plot_df,
        dimensions=list(plot_df.columns),
        opacity=0.35,
        height=1200,
        width=1200,
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker={"size": 4})
    fig.update_layout(title="Interactive Scatter Matrix (sampled for browser performance)")
    return fig


def build_dashboard_html(heatmap, scatter, corr, top_corrs, top_groups):
    heatmap_div = pio.to_html(heatmap, include_plotlyjs=True, full_html=False)
    scatter_div = pio.to_html(scatter, include_plotlyjs=False, full_html=False)

    interesting = top_corrs[
        ~(
            top_corrs["left"].str.startswith("dx_")
            & top_corrs["right"].str.startswith("dx_")
        )
    ].head(10)

    bullets = []
    for _, row in interesting.iterrows():
        bullets.append(
            f"<li><strong>{row['left']}</strong> vs <strong>{row['right']}</strong>: "
            f"{row['correlation']:.3f}</li>"
        )

    group_items = "".join(f"<li>{g}</li>" for g in top_groups)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Combined Social Encounter Correlation Analysis</title>
  <style>
    :root {{
      --bg: #f5efe5;
      --panel: rgba(255, 251, 245, 0.92);
      --ink: #1f2d2f;
      --muted: #5f7074;
      --accent: #0d6c63;
      --line: rgba(31,45,47,0.14);
      --shadow: 0 18px 45px rgba(25,35,38,0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(196,110,65,.14), transparent 28%),
        radial-gradient(circle at top right, rgba(13,108,99,.12), transparent 26%),
        linear-gradient(180deg, #f8f3eb 0%, #efe4d7 100%);
      line-height: 1.55;
    }}
    .wrap {{ width: min(1320px, calc(100vw - 28px)); margin: 0 auto; padding: 28px 0 42px; }}
    .card {{
      background: var(--panel);
      border: 1px solid color-mix(in srgb, white 55%, var(--line));
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      padding: 28px;
      margin-bottom: 18px;
    }}
    h1, h2, h3 {{ margin: 0 0 10px; line-height: 1.08; }}
    h1 {{ font-size: clamp(34px, 5vw, 58px); }}
    h2 {{ font-size: 28px; }}
    p {{ margin: 0 0 12px; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; }}
    ul {{ margin: 0; padding-left: 20px; }}
    li {{ margin-bottom: 8px; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 18px; }}
    .meta .mini {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      background: rgba(255,255,255,.5);
    }}
    .mini strong {{ display: block; font-size: 28px; margin-bottom: 4px; }}
    .plot {{ overflow: auto; }}
    code {{ color: var(--accent); }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .meta {{ grid-template-columns: 1fr; }}
      .card {{ padding: 20px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Combined Social Encounter Correlation Analysis</h1>
      <p>
        This view converts repeated social-screening questions into encounter-level numeric features,
        joins them with the most common diagnosis groups, and then computes a Spearman correlation matrix.
        The heatmap is annotated in every square, and the scatter matrix uses the same variables on a sampled set of encounters.
      </p>
      <div class="meta">
        <div class="mini"><strong>{len(corr.columns)}</strong><span>analytic variables</span></div>
        <div class="mini"><strong>{corr.shape[0]:,}</strong><span>matrix dimensions</span></div>
        <div class="mini"><strong>{len(top_corrs):,}</strong><span>unique variable pairs tested</span></div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Key Points</h2>
        <p>These are the strongest pairwise relationships after pivoting the social questions to encounter level.</p>
        <ul>
          {''.join(bullets)}
        </ul>
      </div>
      <div class="card">
        <h2>Diagnosis Indicators Included</h2>
        <p>The matrix also includes binary flags for the most common diagnosis groups observed in this file.</p>
        <ul>{group_items}</ul>
      </div>
    </section>

    <section class="card">
      <h2>Interactive Correlation Matrix</h2>
      <p>Each square shows the exact correlation value. Darker red means stronger positive association; darker blue means stronger negative association.</p>
      <div class="plot">{heatmap_div}</div>
    </section>

    <section class="card">
      <h2>Interactive Scatter Matrix</h2>
      <p>This scatter matrix uses the same variables as the heatmap and samples the encounter table for browser performance.</p>
      <div class="plot">{scatter_div}</div>
    </section>
  </div>
</body>
</html>
"""
    return html


def main():
    print(f"reading source: {SOURCE_FILE}")
    long_df = load_long_data()
    print(f"filtered rows: {len(long_df):,}")

    wide, top_groups = build_feature_table(long_df)
    numeric_df = wide.drop(columns=["GroupName"], errors="ignore")
    numeric_df = numeric_df.dropna(axis=1, how="all")

    min_non_null = max(250, int(len(numeric_df) * 0.05))
    keep_cols = [
        col
        for col in numeric_df.columns
        if numeric_df[col].notna().sum() >= min_non_null and numeric_df[col].nunique(dropna=True) >= 2
    ]
    numeric_df = numeric_df[keep_cols]
    corr = compute_corr(numeric_df)
    corr.to_csv(os.path.join(OUT, "combinedsocial_corr_matrix.csv"))
    save_static_heatmap(corr)

    top_corrs = top_pairs(corr)
    top_corrs.to_csv(os.path.join(OUT, "combinedsocial_top_correlations.csv"), index=False)

    heatmap = build_interactive_heatmap(corr)
    scatter = build_scatter_matrix(numeric_df)

    scatter.write_html(os.path.join(OUT, "combinedsocial_scatter_matrix.html"), include_plotlyjs="cdn")
    html = build_dashboard_html(heatmap, scatter, corr, top_corrs, top_groups)
    with open(os.path.join(OUT, "combinedsocial_analysis_dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"saved -> combinedsocial_corr_matrix.csv")
    print(f"saved -> combinedsocial_corr_matrix.png")
    print(f"saved -> combinedsocial_scatter_matrix.html")
    print(f"saved -> combinedsocial_analysis_dashboard.html")
    print("\nTop correlations:")
    print(top_corrs.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
