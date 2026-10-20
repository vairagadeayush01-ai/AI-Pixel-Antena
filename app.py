"""
AI Pixel Antenna — Inverse Design Dashboard
=============================================
Streamlit app for ML-driven 12x12 pixel antenna inverse design.
Uses the in-band fine-tuned model (13-17 GHz).
"""

import json
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).resolve().parent
DATA   = ROOT / "data"
MASKS  = ROOT / "masks"
IMAGES = ROOT / "images"

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Pixel Antenna Designer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="stApp"] { font-family: 'Inter', sans-serif; }

.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 18px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    color: #fff;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.hero-banner h1 {
    font-size: 2.4rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #00d2ff, #928dab, #ff6ec7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-banner p { color: #b8b8d0; margin-top: 0.5rem; font-size: 1.05rem; }

.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0,210,255,0.15);
}
.metric-card .label {
    font-size: 0.78rem; color: #7f8c9b;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.7rem; font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #928dab);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .sub { font-size: 0.72rem; color: #5a6577; margin-top: 0.15rem; }

.section-hdr {
    font-size: 1.35rem; font-weight: 700; color: #e0e0e0;
    margin: 2rem 0 0.8rem 0; padding-bottom: 0.4rem;
    border-bottom: 2px solid rgba(0,210,255,0.25);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #1a1a2e) !important;
}
.stTabs [data-baseweb="tab"] { font-weight: 600; color: #a0a0c0; }
.stTabs [aria-selected="true"] { color: #00d2ff !important; border-bottom-color: #00d2ff !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached data loaders ─────────────────────────────────────────────────
@st.cache_data
def load_json(name):
    return json.loads((DATA / name).read_text())

@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA / name)

@st.cache_data
def load_training():
    df = pd.read_csv(DATA / "training_history.csv")
    return df.dropna(subset=["epoch"]).reset_index(drop=True)

@st.cache_data
def load_masks():
    """Load all antenna mask CSVs → dict keyed by frequency (float)."""
    out = {}
    for f in MASKS.glob("antenna_mask_*GHz.csv"):
        freq_str = f.stem.replace("antenna_mask_", "").replace("GHz", "")
        try:
            out[float(freq_str)] = np.loadtxt(f, delimiter=",")
        except Exception:
            pass
    return out


# ── Load data ────────────────────────────────────────────────────────────
summary     = load_json("summary.json")
config      = summary["config"]
em_targets  = load_csv("inverse_design_em_targets.csv")
em_spectra  = load_csv("inverse_design_em_spectra.csv")
training    = load_training()
masks       = load_masks()
bin_stats   = load_csv("bin_stats.csv")
hist_res    = load_csv("hist_resonance.csv")
subset_bins = load_csv("subset_bin_counts.csv")
per_bin     = load_csv("per_bin_performance.csv")

# Optional files (may not be present in minimal deploy)
_ft_path = DATA / "full_test_metrics.csv"
full_test = pd.read_csv(_ft_path) if _ft_path.exists() else None
_em_path = DATA / "em_validation_results.csv"
em_val = pd.read_csv(_em_path) if _em_path.exists() else None
_mm_path = DATA / "model_metrics.csv"
model_metrics = pd.read_csv(_mm_path) if _mm_path.exists() else None
_fi_path = DATA / "failure_types_in_band.csv"
fail_in = pd.read_csv(_fi_path) if _fi_path.exists() else None
_fo_path = DATA / "failure_types_outside_band.csv"
fail_out = pd.read_csv(_fo_path) if _fo_path.exists() else None


# ── Helper ───────────────────────────────────────────────────────────────
def metric_card(label, value, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def plotly_dark(fig, height=420, **kw):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.6)",
        height=height,
        margin=dict(l=50, r=20, t=30, b=50),
        **kw,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero-banner">'
    '<h1>📡 AI Pixel Antenna Designer</h1>'
    '<p>ML-driven inverse design of 12x12 pixel antennas &middot; In-Band fine-tuned model (13 &ndash; 17 GHz)</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.radio(
        "Go to",
        [
            "🔬 Interactive Designer",
            "📊 Model Performance",
            "📈 Training Curves",
            "🎯 Per-Bin Analysis",
            "🧪 EM Validation",
            "📋 Project Report",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Model Config")
    st.markdown(
        f"**Band:** {config['BAND'][0]} – {config['BAND'][1]} GHz\n\n"
        f"**Threshold:** {config['THRESH_DB']} dB\n\n"
        f"**Tolerance:** {config['TOL_GHZ']} GHz\n\n"
        f"**Best Epoch:** {summary['best_epoch']}\n\n"
        f"**Training Subset:** {summary['subset_train_size']:,}"
    )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: Interactive Designer
# ══════════════════════════════════════════════════════════════════════════
if page == "🔬 Interactive Designer":
    st.markdown('<div class="section-hdr">🔬 Inverse Antenna Design</div>', unsafe_allow_html=True)
    st.markdown(
        "Select a **target resonant frequency** to see the ML-generated 12×12 antenna mask, "
        "its predicted S₁₁ spectrum, and key antenna properties."
    )

    available_freqs = sorted(em_targets["target_GHz"].unique())
    target_freq = st.select_slider(
        "🎚️ Target Frequency (GHz)",
        options=available_freqs,
        value=available_freqs[len(available_freqs) // 2],
    )

    mask_arr = masks.get(target_freq)
    if mask_arr is None and len(masks) > 0:
        nearest_f = min(masks.keys(), key=lambda f: abs(f - target_freq))
        mask_arr = masks[nearest_f]

    target_row = em_targets[em_targets["target_GHz"] == target_freq]
    if target_row.empty and not em_targets.empty:
        nearest_target = min(em_targets["target_GHz"], key=lambda f: abs(f - target_freq))
        target_row = em_targets[em_targets["target_GHz"] == nearest_target]

    # ── Mask + Spectrum ──────────────────────────────────────────────────
    col_mask, col_spec = st.columns([1, 1.6])

    with col_mask:
        st.markdown("#### 🧩 Generated Antenna Mask")
        if mask_arr is not None:
            fig_mask = go.Figure(data=go.Heatmap(
                z=np.flipud(mask_arr),
                colorscale=[[0, "#0d1117"], [1, "#00d2ff"]],
                showscale=False, xgap=2, ygap=2,
                hovertemplate="Row %{y}, Col %{x}<br>Pixel: %{z}<extra></extra>",
            ))
            fig_mask.update_layout(
                width=420, height=420,
                plot_bgcolor="#0d1117", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"),
            )
            st.plotly_chart(fig_mask, use_container_width=True)

            pixels_on = int(np.sum(mask_arr != 0))
            st.markdown(f"**Pixels ON:** {pixels_on} / {mask_arr.size} "
                         f"({100*pixels_on/mask_arr.size:.1f}%)")
        else:
            st.warning(f"No mask file found for {target_freq} GHz.")

    with col_spec:
        st.markdown("#### 📉 Predicted S₁₁ Spectrum")
        spec_col = f"{target_freq}GHz"
        if spec_col not in em_spectra.columns:
            spec_cols = [c for c in em_spectra.columns if c != "Frequency_GHz"]
            if spec_cols:
                spec_col = min(spec_cols, key=lambda c: abs(float(c.replace("GHz", "")) - target_freq))

        if spec_col in em_spectra.columns:
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(
                x=em_spectra["Frequency_GHz"], y=em_spectra[spec_col],
                mode="lines", line=dict(color="#00d2ff", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,210,255,0.08)",
                name="Predicted S₁₁",
            ))
            fig_spec.add_hline(y=-10, line_dash="dash", line_color="#ff6ec7",
                                annotation_text="−10 dB threshold",
                                annotation_position="top left",
                                annotation_font_color="#ff6ec7")
            fig_spec.add_vline(x=target_freq, line_dash="dot", line_color="#ffd700",
                                annotation_text=f"Target {target_freq} GHz",
                                annotation_position="top right",
                                annotation_font_color="#ffd700")
            fig_spec.add_vrect(x0=13.0, x1=17.0, fillcolor="rgba(0,210,255,0.05)", line_width=0,
                                annotation_text="In-Band (13–17 GHz)",
                                annotation_position="top left",
                                annotation_font_color="rgba(0,210,255,0.4)")
            plotly_dark(fig_spec, xaxis_title="Frequency (GHz)", yaxis_title="S₁₁ (dB)", showlegend=False)
            st.plotly_chart(fig_spec, use_container_width=True)
        else:
            st.warning(f"Spectrum data not found for {spec_col}.")

    # ── Antenna properties ───────────────────────────────────────────────
    if not target_row.empty:
        row = target_row.iloc[0]
        st.markdown('<div class="section-hdr">📋 Antenna Properties</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Target Frequency", f"{row['target_GHz']} GHz")
        with c2:
            metric_card("Predicted Resonance", f"{row['pred_resonance_GHz']} GHz",
                         f"Error: {abs(row['target_GHz'] - row['pred_resonance_GHz']):.3f} GHz")
        with c3:
            status = "Below −10 dB ✅" if row['pred_min_S11_dB'] <= -10 else "Above −10 dB ⚠️"
            metric_card("Min S₁₁", f"{row['pred_min_S11_dB']:.1f} dB", status)
        with c4:
            metric_card("Number of Dips", f"{int(row['n_dips'])}")
        with c5:
            metric_card("Pixels ON", f"{int(row['pixels_ON'])}", "of 144 total")

    # ── Physical design summary ──────────────────────────────────────────
    with st.expander("🔧 Physical Antenna Specifications", expanded=False):
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Grid size | 12 × 12 pixels (144 total) |
        | Substrate | Air (εᵣ = 1) |
        | Patch dimensions | 2 × 3.75 mm / √εᵣ per axis |
        | Pixel overlap | 25% (fabrication tolerance) |
        | Feed type | Microstrip, left edge, 50 Ω |
        | Ground plane | 2× patch size |
        | Conductor | Copper, 35 μm thickness |
        | Dielectric thickness | 0.61 mm |
        """)

    # ── Download mask ────────────────────────────────────────────────────
    if mask_arr is not None:
        st.markdown("---")
        csv_data = "\n".join(",".join(str(int(v)) for v in r) for r in mask_arr)
        st.download_button(
            "⬇️ Download Antenna Mask (CSV)",
            data=csv_data,
            file_name=f"antenna_mask_{target_freq}GHz.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Performance
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<div class="section-hdr">📊 Model Performance Overview</div>', unsafe_allow_html=True)

    # Metric cards from model_metrics
    if model_metrics is not None and not model_metrics.empty:
        cols = st.columns(len(model_metrics))
        for i, (_, r) in enumerate(model_metrics.iterrows()):
            with cols[i]:
                metric_card(
                    r["subset"],
                    f"{r['success_%']:.1f}%",
                    f"MAE {r['spec_MAE_dB']:.3f} dB · BW err {r['bw_err_GHz']:.3f} GHz",
                )
        st.markdown("")

    # Full test metrics
    if full_test is not None:
        st.markdown('<div class="section-hdr">📋 Test-Set Metrics</div>', unsafe_allow_html=True)
        ft_display = full_test.copy()
        ft_display.columns = [c.replace("_", " ").title() for c in ft_display.columns]
        st.dataframe(ft_display, use_container_width=True, hide_index=True)

    # Failure types
    if fail_in is not None:
        st.markdown('<div class="section-hdr">🔍 Failure Analysis</div>', unsafe_allow_html=True)
        tab_in, tab_out = st.tabs(["In-Band Failures", "Outside-Band Failures"])

        for tab, df, label in [(tab_in, fail_in, "In-Band"), (tab_out, fail_out, "Outside-Band")]:
            with tab:
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    cols_data = [c for c in df.columns if c != df.columns[0]]
                    fig = go.Figure(go.Bar(
                        x=[c.replace("_%", " %").replace("_", " ").title() for c in cols_data],
                        y=[row[c] for c in cols_data],
                        marker=dict(
                            color=["#22c55e", "#ef4444", "#f59e0b", "#8b5cf6"],
                            line=dict(width=0),
                        ),
                        text=[f"{row[c]:.1f}%" for c in cols_data],
                        textposition="auto",
                    ))
                    plotly_dark(fig, 350, xaxis_title="", yaxis_title="Percentage (%)",
                                 showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No {label} failure data available.")

    # Images from original analysis
    img_comp = IMAGES / "full_test_metrics.png"
    img_fail = IMAGES / "failure_types.png"
    if img_comp.exists() or img_fail.exists():
        st.markdown("#### Detailed Visualizations")
        ic = st.columns(2)
        if img_comp.exists():
            with ic[0]:
                st.image(str(img_comp), caption="Full Test Metrics", use_container_width=True)
        if img_fail.exists():
            with ic[1]:
                st.image(str(img_fail), caption="Failure Types", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: Training Curves
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Training Curves":
    st.markdown('<div class="section-hdr">📈 Training Curves</div>', unsafe_allow_html=True)

    train_df = training[training["epoch"] >= 1].copy()
    best_ep = summary["best_epoch"]

    # Main loss curves
    fig_loss = make_subplots(specs=[[{"secondary_y": True}]])
    fig_loss.add_trace(go.Scatter(
        x=train_df["epoch"], y=train_df["train_loss"],
        name="Train Loss", line=dict(color="#ff6ec7", width=2),
    ), secondary_y=False)
    fig_loss.add_trace(go.Scatter(
        x=train_df["epoch"], y=train_df["val_loss"],
        name="Val Loss", line=dict(color="#00d2ff", width=2),
    ), secondary_y=False)
    fig_loss.add_trace(go.Scatter(
        x=train_df["epoch"], y=train_df["lr"],
        name="Learning Rate", line=dict(color="#ffd700", width=1.5, dash="dot"),
    ), secondary_y=True)
    fig_loss.add_vline(x=best_ep, line_dash="dash", line_color="#22c55e",
                        annotation_text=f"Best epoch ({best_ep})",
                        annotation_font_color="#22c55e")
    plotly_dark(fig_loss, legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"))
    fig_loss.update_xaxes(title_text="Epoch")
    fig_loss.update_yaxes(title_text="Loss", secondary_y=False)
    fig_loss.update_yaxes(title_text="Learning Rate", secondary_y=True)
    st.plotly_chart(fig_loss, use_container_width=True)

    # Loss component improvement
    st.markdown('<div class="section-hdr">📦 Loss Improvement (Start → Best Epoch)</div>',
                unsafe_allow_html=True)
    comp = pd.DataFrame(summary["loss_components_start_vs_best"])
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=comp["index"], y=comp["start"],
                               name="Start", marker_color="#ff6ec7"))
    fig_comp.add_trace(go.Bar(x=comp["index"], y=comp[f"best epoch ({best_ep})"],
                               name=f"Best (epoch {best_ep})", marker_color="#00d2ff"))
    plotly_dark(fig_comp, 380, barmode="group", xaxis_title="", yaxis_title="Loss Value")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Component breakdowns
    st.markdown("#### Component Losses over Epochs")
    col_s, col_d = st.columns(2)
    with col_s:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=train_df["epoch"], y=train_df["train_spec"],
                                    name="Train", line=dict(color="#ff6ec7")))
        fig_s.add_trace(go.Scatter(x=train_df["epoch"], y=train_df["val_spec"],
                                    name="Val", line=dict(color="#00d2ff")))
        plotly_dark(fig_s, 300, xaxis_title="Epoch", yaxis_title="Spectrum Loss (MSE, dB²)")
        st.plotly_chart(fig_s, use_container_width=True)
    with col_d:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=train_df["epoch"], y=train_df["train_design"],
                                    name="Train", line=dict(color="#ff6ec7")))
        fig_d.add_trace(go.Scatter(x=train_df["epoch"], y=train_df["val_design"],
                                    name="Val", line=dict(color="#00d2ff")))
        plotly_dark(fig_d, 300, xaxis_title="Epoch", yaxis_title="Design Loss")
        st.plotly_chart(fig_d, use_container_width=True)

    # Training curves image
    tc_img = IMAGES / "training_curves.png"
    if tc_img.exists():
        with st.expander("📸 Full Training Curves (static plot)", expanded=False):
            st.image(str(tc_img), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: Per-Bin Analysis
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Per-Bin Analysis":
    st.markdown('<div class="section-hdr">🎯 Success Rate by Frequency Bin</div>', unsafe_allow_html=True)

    fig_bin = go.Figure()
    fig_bin.add_trace(go.Scatter(
        x=per_bin["bin_start_GHz"], y=per_bin["success_%"],
        mode="lines+markers",
        line=dict(color="#00d2ff", width=2.5),
        marker=dict(size=7, color="#00d2ff", line=dict(width=1, color="#fff")),
        name="Success %",
        fill="tozeroy", fillcolor="rgba(0,210,255,0.06)",
    ))
    fig_bin.add_vrect(x0=13.0, x1=17.0, fillcolor="rgba(0,210,255,0.06)", line_width=0,
                       annotation_text="In-Band (13–17 GHz)",
                       annotation_position="top left",
                       annotation_font_color="rgba(0,210,255,0.5)")
    plotly_dark(fig_bin, 420, xaxis_title="Bin Start (GHz)", yaxis_title="Success Rate (%)",
                 showlegend=False)
    st.plotly_chart(fig_bin, use_container_width=True)

    # Dataset distribution
    st.markdown('<div class="section-hdr">📊 Dataset Distribution</div>', unsafe_allow_html=True)
    col_rh, col_sb = st.columns(2)

    with col_rh:
        st.markdown("**Resonance Frequency Distribution**")
        fig_rh = go.Figure(go.Bar(
            x=hist_res["bin_start_GHz"], y=hist_res["resonant_designs"],
            marker_color=["#00d2ff" if ib else "rgba(0,210,255,0.3)" for ib in hist_res["in_band"]],
        ))
        fig_rh.add_vrect(x0=13.0, x1=17.0, fillcolor="rgba(0,210,255,0.04)", line_width=0)
        plotly_dark(fig_rh, 320, xaxis_title="Bin Start (GHz)", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig_rh, use_container_width=True)

    with col_sb:
        st.markdown("**Training Subset Sampling**")
        fig_sb = go.Figure()
        fig_sb.add_trace(go.Bar(x=subset_bins["bin_start_GHz"], y=subset_bins["available_in_train"],
                                 name="Available", marker_color="rgba(0,210,255,0.3)"))
        fig_sb.add_trace(go.Bar(x=subset_bins["bin_start_GHz"], y=subset_bins["used_in_train_subset"],
                                 name="Used (capped at 6000)", marker_color="#00d2ff"))
        plotly_dark(fig_sb, 320, barmode="overlay", xaxis_title="Bin Start (GHz)", yaxis_title="Count",
                     legend=dict(bgcolor="rgba(0,0,0,0.4)"))
        st.plotly_chart(fig_sb, use_container_width=True)

    # Bin statistics table
    st.markdown("#### Bin Statistics")
    st.dataframe(bin_stats, use_container_width=True, hide_index=True)

    # Supplemental images
    supp_imgs = [("success_per_bin.png", "Success per Bin"),
                 ("subset_analysis.png", "Subset Analysis")]
    cols_img = st.columns(2)
    for i, (fname, cap) in enumerate(supp_imgs):
        p = IMAGES / fname
        if p.exists():
            with cols_img[i]:
                st.image(str(p), caption=cap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: EM Validation
# ══════════════════════════════════════════════════════════════════════════
elif page == "🧪 EM Validation":
    st.markdown('<div class="section-hdr">🧪 Full-Wave EM Simulation Validation</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Antenna masks generated by the ML model were validated using **MATLAB Antenna Toolbox** "
        "full-wave EM simulations (`sparameters`, 10–20 GHz, 81 frequency points)."
    )

    if em_val is not None and not em_val.empty:
        # Summary cards
        hits = em_val["Hit"].sum()
        total = len(em_val)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Designs Simulated", str(total))
        with c2:
            metric_card("Hits (S₁₁ ≤ −10 dB)", f"{hits}/{total}",
                         f"{100*hits/total:.0f}% success")
        with c3:
            metric_card("Median Freq Error", f"{em_val['Frequency_Error_GHz'].median():.3f} GHz")
        with c4:
            metric_card("Median Min S₁₁", f"{em_val['Minimum_S11_dB'].median():.1f} dB")

        st.markdown("")

        # Bar chart: min S11 near target
        fig_em = go.Figure(go.Bar(
            x=em_val["Target_GHz"].astype(str),
            y=em_val["MinS11_near_target_dB"],
            marker_color=["#22c55e" if h else "#ef4444" for h in em_val["Hit"]],
            text=[f"{v:.1f} dB" for v in em_val["MinS11_near_target_dB"]],
            textposition="auto",
        ))
        fig_em.add_hline(y=-10, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                          annotation_text="−10 dB threshold", annotation_font_color="#ff6ec7")
        plotly_dark(fig_em, 400, xaxis_title="Target Frequency (GHz)",
                     yaxis_title="Min S₁₁ near target (dB)", showlegend=False)
        st.plotly_chart(fig_em, use_container_width=True)

        # Results table
        st.markdown("#### Detailed Results")
        st.dataframe(em_val, use_container_width=True, hide_index=True)

        # EM validation images
        em_imgs = [
            ("em_validation/S11_prev_vs_new.png", "Simulated S₁₁ Curves"),
            ("em_validation/masks_prev_vs_new.png", "Antenna Masks"),
            ("em_validation/summary_prev_vs_new.png", "Validation Summary"),
        ]
        existing_imgs = [(IMAGES / f, c) for f, c in em_imgs if (IMAGES / f).exists()]
        if existing_imgs:
            st.markdown("#### MATLAB Simulation Plots")
            for p, cap in existing_imgs:
                st.image(str(p), caption=cap, use_container_width=True)
    else:
        st.info("No EM validation results found. Run `batch_validate_antennas.m` in MATLAB to generate them.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 6: Project Report
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Project Report":
    st.markdown('<div class="section-hdr">📋 Project Report</div>', unsafe_allow_html=True)

    st.markdown("""
    ## 1. Introduction

    This project develops a **machine-learning-driven inverse design** system for **12x12 pixel antennas**.
    Given a target resonant frequency, the model generates a binary pixel mask that, when fabricated as a
    microstrip patch antenna, resonates at the desired frequency within the **13 - 17 GHz** band.

    The antenna is a planar structure consisting of a 12x12 grid of metallic pixels on a dielectric
    substrate. Each pixel is either **ON** (metal present) or **OFF** (removed). The pixel pattern
    determines the antenna's electromagnetic response, specifically its S11 (return loss) spectrum.

    ---

    ## 2. Methodology

    ### 2.1 Model Architecture
    The model performs **inverse design**: given a target S11 spectrum defined by a target resonant
    frequency, it outputs a 12x12 binary mask representing the antenna geometry.

    ### 2.2 Training Strategy
    - **In-band fine-tuning** on samples resonating within 13 - 17 GHz
    - **Balanced bin sampling**: capped at 6,000 samples per 0.5 GHz bin
    - **Cosine learning rate schedule**: from 1e-4 to 1e-6 over 30 epochs
    - **Multi-component loss**: Spectrum MSE (alpha = 5.8) + design regularization (beta = 1.0) + binarization penalty

    ### 2.3 Training Configuration
    """)

    config_df = pd.DataFrame([
        {"Parameter": "Band", "Value": f"{config['BAND'][0]} - {config['BAND'][1]} GHz"},
        {"Parameter": "S11 Threshold", "Value": f"{config['THRESH_DB']} dB"},
        {"Parameter": "Tolerance", "Value": f"{config['TOL_GHZ']} GHz"},
        {"Parameter": "Bin Width", "Value": f"{config['BIN_W']} GHz"},
        {"Parameter": "Max per Bin", "Value": f"{config['MAX_PER_BIN']}"},
        {"Parameter": "Learning Rate", "Value": f"{config['LR']}"},
        {"Parameter": "LR Min", "Value": f"{config['LR_MIN']}"},
        {"Parameter": "Scheduler", "Value": config["SCHEDULER"]},
        {"Parameter": "Alpha (spectrum weight)", "Value": config["ALPHA"]},
        {"Parameter": "Beta (design weight)", "Value": config["BETA"]},
        {"Parameter": "Epochs", "Value": config["EPOCHS"]},
        {"Parameter": "Patience", "Value": config["PATIENCE"]},
        {"Parameter": "Batch Size", "Value": config["BATCH"]},
        {"Parameter": "Seed", "Value": config["SEED"]},
    ])
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    ---

    ## 3. Results

    ### 3.1 Training Progress
    The model was trained for 30 epochs with cosine annealing. The best validation loss was achieved at
    **epoch {summary['best_epoch']}** with total validation loss **{summary['best_selection_metric']:.4f}**.

    ### 3.2 Key Visualizations
    """)

    image_grid = [
        ("training_curves.png", "Training Curves"),
        ("success_per_bin.png", "Success per Frequency Bin"),
        ("target_vs_predicted_frequency.png", "Target vs Predicted Frequency"),
        ("dataset_examples.png", "Dataset Examples"),
        ("mean_spectra_and_pixel_map.png", "Mean Spectra & Pixel Map"),
        ("resonance_error_hist_ecdf.png", "Resonance Error Distribution"),
        ("em_targets_predicted_spectra.png", "Inverse Design Predicted Spectra"),
        ("mean_masks.png", "Mean Antenna Masks"),
    ]
    for i in range(0, len(image_grid), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(image_grid):
                fname, caption = image_grid[i + j]
                p = IMAGES / fname
                if p.exists():
                    with col:
                        st.image(str(p), caption=caption, use_container_width=True)

    st.markdown(f"""
    ---

    ## 4. Definitions

    | Term | Definition |
    |------|-----------|
    | **In-band resonant** | Any S11 <= threshold inside 13 - 17 GHz |
    | **Success** | Predicted dip <= threshold within +/-0.25 GHz of target |
    | **Threshold** | -10 dB |
    | **Tolerance** | +/-0.25 GHz |

    ---

    ## 5. Dataset Integrity

    - **Total rows:** {summary['duplicates']['rows']:,}
    - **Distinct masks:** {summary['duplicates']['distinct_masks']:,}
    - **Test leakage:** {summary['duplicates']['orig_test_rows_with_mask_in_train']} (0 = clean, no leakage)
    - **Clean test rows:** {summary['duplicates']['clean_test_rows']:,}

    ---

    ## 6. Antenna Design

    The physical antenna is modeled using MATLAB Antenna Toolbox:

    | Parameter | Value |
    |-----------|-------|
    | Grid | 12 x 12 pixels |
    | Substrate | Air (epsilon_r = 1) |
    | Patch size | ~7.5 mm x 7.5 mm |
    | Pixel overlap | 25% |
    | Feed | Microstrip, 50 ohm |
    | Ground plane | 2x patch dimensions |
    | Conductor | Copper, 35 um |
    | Thickness | 0.61 mm |

    EM validation uses `sparameters()` over 10 - 20 GHz (81 frequency points).

    ---

    ## 7. Conclusion

    The in-band fine-tuned model successfully generates antenna masks that resonate at specified
    target frequencies within the 13 - 17 GHz band. The inverse design pipeline has been validated
    both by ML predictions and full-wave EM simulations in MATLAB.
    """)


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#5a6577; font-size:0.8rem;">'
    'AI Pixel Antenna Inverse Design &middot; In-Band Fine-Tuned Model &middot; '
    'Built with Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
