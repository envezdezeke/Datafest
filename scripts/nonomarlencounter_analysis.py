"""
Domain-level interactive correlation analysis for nonomarlencounter.csv.

Outputs:
- eda_output/nonomarl_domain_corr_matrix.csv
- eda_output/nonomarl_domain_corr_matrix.png
- eda_output/nonomarl_domain_scatter_matrix.html
- eda_output/nonomarl_domain_analysis_dashboard.html
- eda_output/nonomarl_domain_top_correlations.csv

This script converts question-level answers into normalized risk scores, then
aggregates them to one score per Domain per encounter. Higher values generally
mean higher hardship / risk or lower support.
"""

import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT = os.path.join(PROJECT_ROOT, "eda_output")
os.makedirs(OUT, exist_ok=True)

DEFAULT_FILE = r"C:\Users\ezeki\Downloads\nonomarlencounter.csv"
SOURCE_FILE = os.environ.get("NONOMARL_FILE", DEFAULT_FILE)

RED_WHITE_GREEN = LinearSegmentedColormap.from_list(
    "red_white_green",
    ["#b2182b", "#f7f7f7", "#1b7837"],
    N=256,
)

SCATTER_SAMPLE = 2500

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


def domain_slug(text):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).lower()).strip("_")
    return re.sub(r"_+", "_", text)


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


def encode_row(question, answers):
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


def load_long_data():
    questions = list(QUESTION_SPECS.keys())
    chunks = []
    for chunk in pd.read_csv(
        SOURCE_FILE,
        usecols=["EncounterKey", "DisplayName", "AnswerText", "Domain", "GroupName"],
        chunksize=250000,
    ):
        chunk = chunk[chunk["DisplayName"].isin(questions)].copy()
        if not chunk.empty:
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def build_domain_frame(df):
    rows = []
    for question, spec in QUESTION_SPECS.items():
        sub = df.loc[df["DisplayName"] == question, ["EncounterKey", "AnswerText"]].copy()
        sub["domain"] = spec["domain"]
        sub["risk_score"] = encode_row(question, sub["AnswerText"])
        sub = sub.dropna(subset=["risk_score"])
        rows.append(sub[["EncounterKey", "domain", "risk_score"]])
    stacked = pd.concat(rows, ignore_index=True)

    domain_wide = stacked.pivot_table(
        index="EncounterKey",
        columns="domain",
        values="risk_score",
        aggfunc="mean",
    )
    domain_wide.columns = [domain_slug(c) for c in domain_wide.columns]

    dx = (
        df[["EncounterKey", "GroupName"]]
        .dropna(subset=["GroupName"])
        .drop_duplicates()
        .groupby("EncounterKey")["GroupName"]
        .first()
    )
    domain_wide = domain_wide.join(dx.rename("GroupName"), how="left")

    top_groups = (
        domain_wide["GroupName"]
        .dropna()
        .value_counts()
        .head(5)
        .index
        .tolist()
    )
    for group in top_groups:
        domain_wide[f"dx_{domain_slug(group)[:38]}"] = (domain_wide["GroupName"] == group).astype(int)

    domain_wide["any_dx_recorded"] = domain_wide["GroupName"].notna().astype(int)
    return domain_wide, top_groups


def compute_corr(df_numeric):
    return df_numeric.corr(method="spearman").round(3)


def top_pairs(corr):
    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
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
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def save_static_heatmap(corr):
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=RED_WHITE_GREEN,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        annot_kws={"size": 9},
        cbar_kws={"label": "Spearman correlation"},
    )
    plt.title("Domain-Level Correlation Matrix", pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "nonomarl_domain_corr_matrix.png"), dpi=170, bbox_inches="tight")
    plt.close()


def build_interactive_heatmap(corr):
    labels = list(corr.columns)
    text = corr.applymap(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale=[[0.0, "#b2182b"], [0.5, "#f7f7f7"], [1.0, "#1b7837"]],
            zmin=-1,
            zmax=1,
            text=text.values,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar={"title": "Spearman r"},
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Interactive Domain-Level Correlation Matrix",
        width=1050,
        height=900,
        xaxis={"tickangle": 45},
        yaxis={"autorange": "reversed"},
        margin={"l": 170, "r": 40, "t": 80, "b": 170},
    )
    return fig


def build_scatter_matrix(df_numeric):
    plot_df = df_numeric.dropna().copy()
    if len(plot_df) > SCATTER_SAMPLE:
        plot_df = plot_df.sample(SCATTER_SAMPLE, random_state=42)
    fig = px.scatter_matrix(
        plot_df,
        dimensions=list(plot_df.columns),
        opacity=0.35,
        width=1150,
        height=1150,
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker={"size": 4})
    fig.update_layout(title="Interactive Scatter Matrix (sampled for browser performance)")
    return fig


def suggest_graphs():
    return [
        "Clustered heatmap of domain scores by top diagnosis groups to see which hardship bundles travel together clinically.",
        "Radar chart comparing average domain-risk profiles for the top 5 diagnosis groups.",
        "Boxplots of depression, stress, and financial strain by housing stability risk decile.",
        "Parallel-coordinates plot across domain scores to trace common multi-need patient patterns.",
        "Network graph where nodes are domains and edge thickness is correlation strength above a chosen threshold.",
        "Stacked bar chart of high-risk rate by domain, split by presence or absence of a diagnosis group like depression or sepsis.",
    ]


def build_dashboard_html(heatmap, scatter, corr, top_corrs, top_groups):
    domain_only = top_corrs[
        ~top_corrs["left"].str.startswith("dx_")
        & ~top_corrs["right"].str.startswith("dx_")
        & (top_corrs["left"] != "any_dx_recorded")
        & (top_corrs["right"] != "any_dx_recorded")
    ].head(10)
    dx_links = top_corrs[
        top_corrs["left"].str.startswith("dx_")
        | top_corrs["right"].str.startswith("dx_")
        | (top_corrs["left"] == "any_dx_recorded")
        | (top_corrs["right"] == "any_dx_recorded")
    ].head(10)

    bullets = "".join(
        f"<li><strong>{row.left}</strong> vs <strong>{row.right}</strong>: {row.correlation:.3f}</li>"
        for row in domain_only.itertuples()
    )
    dx_bullets = "".join(
        f"<li><strong>{row.left}</strong> vs <strong>{row.right}</strong>: {row.correlation:.3f}</li>"
        for row in dx_links.itertuples()
    )
    graph_bullets = "".join(f"<li>{item}</li>" for item in suggest_graphs())
    group_bullets = "".join(f"<li>{g}</li>" for g in top_groups)

    heatmap_div = pio.to_html(heatmap, include_plotlyjs=True, full_html=False)
    scatter_div = pio.to_html(scatter, include_plotlyjs=False, full_html=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nonomarl Encounter Domain Analysis</title>
  <style>
    :root {{
      --bg: #f5efe5;
      --panel: rgba(255,251,245,.92);
      --ink: #1f2d2f;
      --muted: #5f7074;
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
    .wrap {{ width:min(1320px,calc(100vw - 28px)); margin:0 auto; padding:28px 0 42px; }}
    .card {{
      background: var(--panel);
      border: 1px solid color-mix(in srgb, white 55%, var(--line));
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      padding: 28px;
      margin-bottom: 18px;
    }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
    .meta {{ display:grid; grid-template-columns: repeat(3,1fr); gap:14px; margin-top:18px; }}
    .mini {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:rgba(255,255,255,.5); }}
    .mini strong {{ display:block; font-size:28px; margin-bottom:4px; }}
    h1,h2,h3 {{ margin:0 0 10px; line-height:1.08; }}
    h1 {{ font-size: clamp(34px,5vw,58px); }}
    h2 {{ font-size: 28px; }}
    p {{ margin:0 0 12px; color:var(--muted); }}
    ul {{ margin:0; padding-left:20px; }}
    li {{ margin-bottom:8px; }}
    .plot {{ overflow:auto; }}
    @media (max-width: 980px) {{
      .grid, .meta {{ grid-template-columns:1fr; }}
      .card {{ padding:20px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Domain-Level Correlation Analysis</h1>
      <p>
        This view groups question-level responses into the categories listed in <code>Domain</code>,
        converts each answer into a normalized risk score, and then averages within each domain at the encounter level.
        Higher domain scores generally mean greater hardship, risk, or lower support.
      </p>
      <div class="meta">
        <div class="mini"><strong>{len(corr.columns)}</strong><span>analytic variables</span></div>
        <div class="mini"><strong>{corr.shape[0]:,}</strong><span>matrix dimensions</span></div>
        <div class="mini"><strong>{len(top_corrs):,}</strong><span>variable pairs tested</span></div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Key Domain Insights</h2>
        <p>These are the strongest domain-to-domain relationships after rolling up the questions.</p>
        <ul>{bullets}</ul>
      </div>
      <div class="card">
        <h2>Diagnosis-Linked Signals</h2>
        <p>These are the clearest links between domain scores and the most common diagnosis groups in the file.</p>
        <ul>{dx_bullets}</ul>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Top Diagnosis Groups Included</h2>
        <ul>{group_bullets}</ul>
      </div>
      <div class="card">
        <h2>Graphs To Add Next</h2>
        <p>These would make the correlation structure easier to explain in a strategy deck or judging presentation.</p>
        <ul>{graph_bullets}</ul>
      </div>
    </section>

    <section class="card">
      <h2>Interactive Correlation Matrix</h2>
      <p>Each square prints the exact correlation value. Red indicates negative correlation, white is near zero, and green indicates positive correlation.</p>
      <div class="plot">{heatmap_div}</div>
    </section>

    <section class="card">
      <h2>Interactive Scatter Matrix</h2>
      <p>This scatter matrix uses the same domain-level variables and a sampled set of encounters for performance.</p>
      <div class="plot">{scatter_div}</div>
    </section>
  </div>
</body>
</html>
"""
    return html


def main():
    print(f"reading source: {SOURCE_FILE}")
    df = load_long_data()
    print(f"filtered rows: {len(df):,}")

    wide, top_groups = build_domain_frame(df)
    numeric_df = wide.drop(columns=["GroupName"], errors="ignore")
    numeric_df = numeric_df.dropna(axis=1, how="all")

    min_non_null = max(200, int(len(numeric_df) * 0.05))
    keep_cols = [
        col for col in numeric_df.columns
        if numeric_df[col].notna().sum() >= min_non_null and numeric_df[col].nunique(dropna=True) >= 2
    ]
    numeric_df = numeric_df[keep_cols]
    corr = compute_corr(numeric_df)
    corr.to_csv(os.path.join(OUT, "nonomarl_domain_corr_matrix.csv"))
    save_static_heatmap(corr)

    top_corrs = top_pairs(corr)
    top_corrs.to_csv(os.path.join(OUT, "nonomarl_domain_top_correlations.csv"), index=False)

    heatmap = build_interactive_heatmap(corr)
    scatter = build_scatter_matrix(numeric_df)
    scatter.write_html(os.path.join(OUT, "nonomarl_domain_scatter_matrix.html"), include_plotlyjs="cdn")

    html = build_dashboard_html(heatmap, scatter, corr, top_corrs, top_groups)
    with open(os.path.join(OUT, "nonomarl_domain_analysis_dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("saved -> nonomarl_domain_corr_matrix.csv")
    print("saved -> nonomarl_domain_corr_matrix.png")
    print("saved -> nonomarl_domain_scatter_matrix.html")
    print("saved -> nonomarl_domain_analysis_dashboard.html")
    print("\nTop correlations:")
    print(top_corrs.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
