import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ---------- Page Config ----------
st.set_page_config(
    page_title="Pelvic Parameters Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Minimal Styling ----------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      .metric-card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fff;}
      .section {margin-top: 1rem; margin-bottom: 0.5rem;}
      .small {font-size: 0.85rem; color: #6b7280;}
      .good {color: #16a34a;}
      .warn {color: #d97706;}
      .bad {color: #dc2626;}
      .stMetric {text-align: center;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------

@st.cache_data(show_spinner=False)
def load_excel(path_or_bytes: bytes | str) -> dict:
    if isinstance(path_or_bytes, (str, os.PathLike)):
        xls = pd.ExcelFile(path_or_bytes, engine="openpyxl")
    else:
        xls = pd.ExcelFile(io.BytesIO(path_or_bytes), engine="openpyxl")
    sheets = {}
    for name in xls.sheet_names:
        df = xls.parse(name)
        sheets[name] = df
    return sheets


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    # Normalize column names
    col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # Expected columns
    expected = [
        "participant", "posture",
        "tilt_max", "tilt_min", "tilt_range",
        "obliq_max", "obliq_min", "obliq_range",
        "rot_max", "rot_min", "rot_range",
    ]
    # Try to coerce names if slightly different
    alias = {
        "participant": ["participant", "name", "id"],
        "posture": ["posture", "pose"],
        "tilt_max": ["tilt_max", "tiltmax"],
        "tilt_min": ["tilt_min", "tiltmin"],
        "tilt_range": ["tilt_range", "tiltrange", "tilt_rng"],
        "obliq_max": ["obliq_max", "obliquity_max", "obliqmax"],
        "obliq_min": ["obliq_min", "obliquity_min", "obliqmin"],
        "obliq_range": ["obliq_range", "obliquity_range", "obliqrng"],
        "rot_max": ["rot_max", "rotation_max", "rotmax"],
        "rot_min": ["rot_min", "rotation_min", "rotmin"],
        "rot_range": ["rot_range", "rotation_range", "rotrng"],
    }

    resolved: dict[str, str] = {}
    for want, choices in alias.items():
        for c in df.columns:
            if c in choices:
                resolved[want] = c
                break
    # Fill missing if exact
    for want in expected:
        if want not in resolved and want in df.columns:
            resolved[want] = want

    # Subset/rename
    selected = {resolved[k]: k for k in expected if k in resolved}
    df = df[list(selected.keys())].rename(columns=selected)

    # Standardize text columns
    if "participant" in df:
        df["participant"] = df["participant"].astype(str).str.strip()
    if "posture" in df:
        df["posture"] = (
            df["posture"].astype(str).str.strip().str.lower().map({
                "neutral": "Neutral",
                "forward": "Forward",
                "laidback": "Laidback",
                "layback": "Laidback",
            }).fillna(df["posture"].astype(str).str.title())
        )

    # Coerce numeric columns
    numeric_cols = [c for c in expected if c not in ("participant", "posture") and c in df]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop fully empty rows
    df = df.dropna(how="all")
    return df


def summarize_by(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    value_cols = [c for c in df.columns if c not in ("participant", "posture")]
    agg = df.groupby(by)[value_cols].agg(["count", "mean", "std", "median", "min", "max"])
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    return agg.reset_index()


def match_for_comparison(mocap: pd.DataFrame, opencap: pd.DataFrame) -> pd.DataFrame:
    if mocap is None or opencap is None or mocap.empty or opencap.empty:
        return pd.DataFrame()
    a = mocap.copy()
    b = opencap.copy()
    a["_p"] = a["participant"].str.lower()
    b["_p"] = b["participant"].str.lower()
    a["_post"] = a["posture"].str.lower()
    b["_post"] = b["posture"].str.lower()

    join_cols = [
        "tilt_max", "tilt_min", "tilt_range",
        "obliq_max", "obliq_min", "obliq_range",
        "rot_max", "rot_min", "rot_range",
    ]

    merged = pd.merge(
        a[["participant", "posture", "_p", "_post"] + join_cols].add_prefix("m_"),
        b[["participant", "posture", "_p", "_post"] + join_cols].add_prefix("o_"),
        left_on=["m__p", "m__post"], right_on=["o__p", "o__post"], how="inner"
    )

    # Compute differences (OpenCap - MoCap)
    for c in join_cols:
        merged[f"diff_{c}"] = merged[f"o_{c}"] - merged[f"m_{c}"]
    return merged


# ---------- Sidebar ----------
st.sidebar.title("Controls")
source_choice = st.sidebar.selectbox(
    "Data source",
    options=["Use local file pelvic-parameters.xlsx", "Upload Excel"],
    index=0,
)

uploaded_file = None
if source_choice == "Upload Excel":
    uploaded_file = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])

# Filters placeholder (populated after load)
participants_filter: list[str] | None = None
postures_filter: list[str] | None = None

# ---------- Load Data ----------
with st.spinner("Loading data..."):
    try:
        if uploaded_file is not None:
            sheets = load_excel(uploaded_file.getvalue())
        else:
            default_path = "pelvic-parameters.xlsx"
            if not os.path.exists(default_path):
                st.error("pelvic-parameters.xlsx not found in the working directory.")
                st.stop()
            sheets = load_excel(default_path)
    except Exception as e:
        st.exception(e)
        st.stop()

# Normalize sheets
normalized: dict[str, pd.DataFrame] = {}
for name, df in sheets.items():
    normalized[name] = normalize_columns(df)

sheet_names = list(normalized.keys())

st.title("Pelvic Parameters Analysis")
st.caption("Interactive report comparing MoCap and OpenCap pelvic parameters across postures and participants.")

# ---------- Global Filters ----------
# Combine participants/postures across sheets for filter options
all_participants = sorted(set(pd.concat([d for d in normalized.values()], ignore_index=True)["participant"].dropna().unique()))
all_postures = sorted(set(pd.concat([d for d in normalized.values()], ignore_index=True)["posture"].dropna().unique()))

with st.sidebar:
    participants_filter = st.multiselect("Participants", options=all_participants, default=all_participants)
    postures_filter = st.multiselect("Postures", options=all_postures, default=all_postures)

# Apply filters
for name in sheet_names:
    df = normalized[name]
    if not df.empty:
        normalized[name] = df[df["participant"].isin(participants_filter) & df["posture"].isin(postures_filter)]

# ---------- Overview Metrics ----------
col1, col2, col3, col4 = st.columns(4)
all_rows = sum(len(df) for df in normalized.values())
unique_participants = len(set().union(*[set(df["participant"]) for df in normalized.values() if not df.empty]))
unique_postures = len(set().union(*[set(df["posture"]) for df in normalized.values() if not df.empty]))

with col1:
    st.metric("Total Records", f"{all_rows}")
with col2:
    st.metric("Participants", f"{unique_participants}")
with col3:
    st.metric("Postures", f"{unique_postures}")
with col4:
    st.metric("Sheets", f"{len(sheet_names)}")

st.divider()

# ---------- Per-Sheet Summaries ----------
for name in sheet_names:
    df = normalized[name]
    st.subheader(f"Sheet: {name}")
    if df.empty:
        st.info("No data after filters.")
        st.divider()
        continue

    with st.expander("Preview data", expanded=False):
        st.dataframe(df, use_container_width=True, height=320)

    summary_by_posture = summarize_by(df, ["posture"])
    summary_by_participant = summarize_by(df, ["participant"]) if "participant" in df else pd.DataFrame()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Summary by Posture**")
        st.dataframe(summary_by_posture, use_container_width=True, height=320)
    with c2:
        st.markdown("**Summary by Participant**")
        st.dataframe(summary_by_participant, use_container_width=True, height=320)

    # Distributions per metric
    numeric_cols = [c for c in df.columns if c not in ("participant", "posture")]
    pick_metric = st.selectbox(
        f"Distribution metric • {name}", options=numeric_cols, key=f"dist_{name}")

    if pick_metric:
        fig = px.violin(df, x="posture", y=pick_metric, color="posture", box=True, points="all")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"))
        heat.update_layout(title="Correlation Heatmap", height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(heat, use_container_width=True)

    st.divider()

# ---------- Cross-Sheet Comparison (MoCap vs OpenCap) ----------
if {n.lower() for n in sheet_names} >= {"mocap", "opencap"}:
    mocap_key = [n for n in sheet_names if n.lower() == "mocap"][0]
    opencap_key = [n for n in sheet_names if n.lower() == "opencap"][0]
    mocap_df = normalized[mocap_key]
    opencap_df = normalized[opencap_key]

    st.subheader("MoCap vs OpenCap: Matched Comparison")

    matched = match_for_comparison(mocap_df, opencap_df)
    if not matched.empty:
        show_cols = [
            "m_participant", "m_posture",
            "m_tilt_max", "o_tilt_max", "diff_tilt_max",
            "m_tilt_min", "o_tilt_min", "diff_tilt_min",
            "m_tilt_range", "o_tilt_range", "diff_tilt_range",
            "m_obliq_max", "o_obliq_max", "diff_obliq_max",
            "m_obliq_min", "o_obliq_min", "diff_obliq_min",
            "m_obliq_range", "o_obliq_range", "diff_obliq_range",
            "m_rot_max", "o_rot_max", "diff_rot_max",
            "m_rot_min", "o_rot_min", "diff_rot_min",
            "m_rot_range", "o_rot_range", "diff_rot_range",
        ]
        present_cols = [c for c in show_cols if c in matched.columns]

        with st.expander("Matched table (OpenCap - MoCap deltas)", expanded=False):
            st.dataframe(matched[present_cols], use_container_width=True, height=420)

        # Bar chart of absolute mean differences per metric
        diff_cols = [c for c in matched.columns if c.startswith("diff_")]
        if diff_cols:
            mean_abs = matched[diff_cols].abs().mean().sort_values(ascending=False)
            fig = px.bar(mean_abs, title="Mean absolute differences (OpenCap - MoCap)")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: MoCap vs OpenCap for a chosen metric
        metric_choice = st.selectbox(
            "Scatter compare metric",
            options=[c.replace("diff_", "") for c in diff_cols] if diff_cols else [],
            index=0 if diff_cols else None,
        )
        if metric_choice:
            sdf = matched.dropna(subset=[f"m_{metric_choice}", f"o_{metric_choice}"])
            fig = px.scatter(
                sdf,
                x=f"m_{metric_choice}", y=f"o_{metric_choice}",
                color="m_posture", hover_data=["m_participant"],
                trendline="ols", trendline_color_override="#0ea5e9",
                title=f"MoCap vs OpenCap • {metric_choice}"
            )
            fig.add_trace(go.Scatter(x=sdf[f"m_{metric_choice}"], y=sdf[f"m_{metric_choice}"],
                                     mode="lines", name="y=x", line=dict(dash="dash", color="#9ca3af")))
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Simple paired t-test per metric across matched rows
        if diff_cols:
            st.markdown("**Paired t-tests (OpenCap vs MoCap)**")
            rows = []
            for c in [d.replace("diff_", "") for d in diff_cols]:
                sub = matched[[f"m_{c}", f"o_{c}"]].dropna()
                if len(sub) >= 2:
                    try:
                        t, p = stats.ttest_rel(sub[f"o_{c}"], sub[f"m_{c}"])
                        rows.append({"metric": c, "n": len(sub), "t_stat": t, "p_value": p})
                    except Exception:
                        pass
            if rows:
                tdf = pd.DataFrame(rows).sort_values("p_value")
                st.dataframe(tdf, use_container_width=True, height=300)
    else:
        st.info("No matched rows between MoCap and OpenCap given current filters.")

    st.divider()

# ---------- Combined Exploration ----------
st.subheader("Combined Exploration Across Sheets")
combined = pd.concat([normalized[name].assign(sheet=name) for name in sheet_names], ignore_index=True)
if combined.empty:
    st.info("No data to explore.")
else:
    # Metric selection
    metric = st.selectbox(
        "Metric to visualize",
        options=[c for c in combined.columns if c not in ("participant", "posture", "sheet")],
        index=0,
        key="combined_metric",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(combined, x="posture", y=metric, color="sheet", points="suspectedoutliers",
                     title=f"Distribution by Posture • {metric}")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            combined.groupby(["sheet", "posture"])[metric].mean().reset_index(),
            x="posture", y=metric, color="sheet", barmode="group",
            title=f"Mean by Posture and Sheet • {metric}"
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(
        range=[combined[metric].min() - 1, combined[metric].max() + 1] ))
        st.plotly_chart(fig, use_container_width=True)

    # Scatter relationships
    numeric_cols = [c for c in combined.columns if c not in ("participant", "posture", "sheet")]
    if len(numeric_cols) >= 2:
        x_opt = st.selectbox("X", options=numeric_cols, index=0, key="scx")
        y_opt = st.selectbox("Y", options=[c for c in numeric_cols if c != x_opt], index=1 if len(numeric_cols) > 1 else 0, key="scy")
        fig = px.scatter(combined, x=x_opt, y=y_opt, color="sheet", symbol="posture",
                         hover_data=["participant"], trendline="ols",
                         title=f"Relationship • {x_opt} vs {y_opt}")
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Download data"):
        st.download_button(
            label="Download combined CSV",
            data=combined.to_csv(index=False).encode("utf-8"),
            file_name="pelvic_parameters_combined.csv",
            mime="text/csv",
        )

st.caption("Report generated with Streamlit • Interactive, modern, and clean layout.")


