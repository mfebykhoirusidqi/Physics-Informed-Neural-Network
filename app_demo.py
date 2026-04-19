"""
Physics-Informed Neural Network — Interactive Portfolio Demo
================================================================
A premium Streamlit application for a Physics + Computational Physics graduate.

Features
--------
Overview       — Hero section, metrics, architecture, applications
Simulator      — Interactive 3D/2D visualisation with parameter controls
Live Training  — Train a PINN in real-time and watch physics emerge
Analysis       — Deep comparison vs analytical, error maps, profiles
Theory         — LaTeX equations, autodiff walkthrough, references

Run:
    streamlit run app_demo.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PINN Demo | Physics × AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark Academic / Physics Lab Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Base Industrial Theme ── */
.stApp {
    background-color: #0b0f19;
    background-image: 
        radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
        radial-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px);
    background-size: 20px 20px, 100px 100px;
    background-position: 0 0, 10px 10px;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}
html, body { scroll-behavior: smooth; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar (Control Panel) ── */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
    box-shadow: inset -5px 0 15px rgba(0,0,0,0.5);
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { 
    color: #f8fafc !important; 
    font-weight: 700; 
    text-transform: uppercase; 
    letter-spacing: 1.5px; 
    font-family: 'JetBrains Mono', monospace;
}

/* ── Hero ── */
.hero-wrapper {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px dashed #334155;
    margin-bottom: 2.5rem;
    background: linear-gradient(180deg, rgba(15,23,42,0.6) 0%, transparent 100%);
}
.hero-badge {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #10b981;
    border-left: 4px solid #10b981;
    border-radius: 2px;
    padding: 0.4rem 1.2rem;
    font-size: 0.8rem;
    color: #10b981;
    letter-spacing: 0.15em;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
}
.gradient-title {
    color: #f8fafc;
    font-size: clamp(2.2rem, 4.5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 1.2rem;
    letter-spacing: -0.02em;
    text-shadow: 0 4px 10px rgba(0,0,0,0.5);
}
.hero-sub {
    color: #cbd5e1;
    font-size: 1.1rem;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Metric cards (Telemetry Data) ── */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 4px solid #f97316;
    border-radius: 4px;
    padding: 1.2rem 1rem;
    text-align: left;
    box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    transition: transform 0.2s, border-left-color 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-left-color: #38bdf8; }
.mc-icon { font-size: 1.2rem; margin-bottom: 0.5rem; opacity: 0.7; }
.mc-val {
    font-size: 1.8rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #f8fafc;
}
.mc-lbl { 
    font-size: 0.7rem; color: #94a3b8; margin-top: 0.3rem;
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; 
}

/* ── Section headers ── */
.sec-hdr {
    display: flex; align-items: flex-end; gap: 0.8rem;
    margin: 2.5rem 0 0.5rem;
}
.sec-hdr-icon { font-size: 1.2rem; opacity: 0.8;}
.sec-hdr-text { 
    font-size: 1.1rem; font-weight: 700; color: #f8fafc; 
    text-transform: uppercase; letter-spacing: 0.1em; 
    font-family: 'JetBrains Mono', monospace;
}
.sec-div {
    height: 1px;
    background: linear-gradient(90deg, #64748b, transparent);
    margin-bottom: 1.5rem;
}

/* ── Info / physics cards ── */
.info-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
    backdrop-filter: blur(4px);
}
.info-card p, .info-card li { color: #cbd5e1; font-size: 0.95rem; margin: 0; line-height: 1.6; }

/* ── Pill tags ── */
.pill {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #475569;
    border-radius: 2px;
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem; color: #38bdf8; margin: 0.2rem;
    font-weight: 600; font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
}

/* ── Alert box ── */
.alert-box {
    background: rgba(56, 189, 248, 0.05);
    border: 1px solid #0369a1;
    border-left: 4px solid #38bdf8;
    border-radius: 2px;
    padding: 1rem 1.4rem;
    margin: 1.5rem 0;
    color: #e0f2fe; font-size: 0.95rem;
}
.alert-warn {
    background: rgba(245, 158, 11, 0.05);
    border: 1px solid #b45309;
    border-left: 4px solid #f59e0b;
    border-radius: 2px;
    padding: 1rem 1.4rem;
    margin: 1.5rem 0;
    color: #fef3c7; font-size: 0.95rem;
}

/* ── Code snippet ── */
.code-snip {
    background: #020617;
    border: 1px solid #1e293b;
    border-left: 3px solid #10b981;
    border-radius: 2px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; color: #34d399;
    margin: 0.8rem 0;
    white-space: pre;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
}

/* ── Feature rows ── */
.feat-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    transition: background 0.2s;
}
.feat-row:hover { background: #334155; }
.feat-icon { color: #f97316; font-size: 1.2rem; flex-shrink: 0; }
.feat-accent { color: #f8fafc; font-weight: 600; font-size: 0.95rem; }
.feat-text { color: #cbd5e1; font-size: 0.9rem;}

/* ── Tabs (Industrial Folders) ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 0;
    padding: 0; gap: 2px;
    border-bottom: 1px solid #475569;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0; color: #64748b; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
    padding: 0.8rem 1.5rem;
    background: #1e293b;
    border: 1px solid #334155;
    border-bottom: none;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    background: #0b0f19 !important;
    color: #10b981 !important;
    border-top: 3px solid #10b981 !important;
    border-left: 1px solid #475569 !important;
    border-right: 1px solid #475569 !important;
    border-bottom: none !important;
}

/* ── Sliders & inputs ── */
.stSlider>label { 
    color: #cbd5e1 !important; font-weight: 600; 
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem !important; 
    text-transform: uppercase; 
}
[data-testid="stMetricValue"] { color: #10b981; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricLabel"] { color: #64748b; font-weight: 700; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;}

/* ── Progress bar ── */
.stProgress>div>div { background: #10b981; }

/* ── Buttons ── */
.stButton>button {
    background: #1e293b !important;
    color: #e2e8f0 !important; border: 1px solid #475569 !important;
    font-weight: 700 !important; border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase; letter-spacing: 1px;
    transition: all 0.2s;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.stButton>button:hover { 
    background: #10b981 !important; 
    border-color: #10b981 !important; color: #000 !important; 
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.4); 
}

/* ── DataFrames ── */
.stDataFrame { border-radius: 2px; overflow: hidden; border: 1px solid #475569; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PINN IMPORT (cached so it only happens once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_pinn_classes():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pinn_complete_starter import PhysicsInformedNN, generate_synthetic_data
    return PhysicsInformedNN, generate_synthetic_data


PhysicsInformedNN, generate_synthetic_data = _load_pinn_classes()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def analytical(x, t, omega_0=1.0, zeta=0.05, A=0.1, k=np.pi):
    return A * np.exp(-zeta * omega_0 * t) * np.cos(omega_0 * t) * np.sin(k * x)


@st.cache_resource(show_spinner=False)
def _try_load_model(mtime):
    """Load saved model from disk. `mtime` busts cache when file changes."""
    path = "pinn_vibration_model.pth"
    if not os.path.exists(path):
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        m = PhysicsInformedNN(hidden_dim=256, num_layers=5, device=device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m.to(device)
    except Exception:
        return None


def load_model():
    path = "pinn_vibration_model.pth"
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    return _try_load_model(mtime)


def model_predict_grid(m, X, T, device="cpu"):
    x_ = torch.tensor(X.flatten(), dtype=torch.float32).to(device)
    t_ = torch.tensor(T.flatten(), dtype=torch.float32).to(device)
    xt = torch.stack([x_, t_], dim=1)
    with torch.no_grad():
        u = m(xt)
    return u.cpu().numpy().reshape(X.shape)


# ── Plotly theme helpers ───────────────────────────────────────────────────

_LAYOUT = dict(
    paper_bgcolor="rgba(6,11,24,.97)",
    plot_bgcolor="rgba(6,11,24,.97)",
    font=dict(family="Inter", color="#90caf9"),
    margin=dict(l=0, r=0, t=40, b=0),
)


def _apply_axes(fig, xtitle="", ytitle="", log_y=False):
    kw = dict(gridcolor="rgba(79,195,247,.12)", color="#90caf9",
              linecolor="rgba(79,195,247,.25)")
    fig.update_xaxes(title_text=xtitle, **kw)
    fig.update_yaxes(title_text=ytitle, type="log" if log_y else "linear", **kw)
    return fig


def contour_fig(Z, X, T, title, cmap="RdBu_r", h=380):
    fig = go.Figure(go.Contour(
        z=Z, x=X[0], y=T[:, 0], colorscale=cmap, ncontours=28,
        colorbar=dict(tickfont=dict(color="#90caf9"),
                      title=dict(text="u(x,t)", font=dict(color="#90caf9"))),
        line=dict(width=.4),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e8f0fe", size=13)),
        height=h, **_LAYOUT,
    )
    _apply_axes(fig, "Position x [m]", "Time t [s]")
    return fig


def surface_fig(Z, X, T, title, cmap="RdBu_r"):
    fig = go.Figure(go.Surface(
        z=Z, x=X[0], y=T[:, 0], colorscale=cmap, opacity=.92,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e8f0fe", size=13)),
        scene=dict(
            xaxis=dict(title="x [m]", gridcolor="rgba(79,195,247,.2)",
                       backgroundcolor="rgba(6,11,24,.8)", color="#90caf9"),
            yaxis=dict(title="t [s]",  gridcolor="rgba(79,195,247,.2)",
                       backgroundcolor="rgba(6,11,24,.8)", color="#90caf9"),
            zaxis=dict(title="u",      gridcolor="rgba(124,77,255,.2)",
                       backgroundcolor="rgba(10,6,24,.8)", color="#90caf9"),
            camera=dict(eye=dict(x=1.45, y=-1.45, z=1.15)),
        ),
        height=460, **_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem">
      <div style="font-size:3rem"></div>
      <div style="font-size:1.05rem;font-weight:800;color:#e8f0fe;margin-top:.4rem">PINN Framework</div>
      <div style="font-size:.75rem;color:#4fc3f7;font-weight:600;margin-top:.2rem">Physics × Deep Learning</div>
    </div>
    <hr style="border-color:rgba(79,195,247,.18);margin:.5rem 0 1rem">
    """, unsafe_allow_html=True)

    st.markdown("**Developer**")
    st.markdown("""
    <div class="info-card" style="margin:.4rem 0">
      <div style="color:#10b981;font-weight:700;margin-bottom:0.3rem">M Feby Khoiru Sidqi</div>
      <div>Physics Graduate</div>
      <div>Computational Physics</div>
      <div>AI / ML Engineer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🖥️ Device**")
    st.info("CUDA (GPU)" if torch.cuda.is_available() else "CPU")

    st.markdown("**Model Status**")
    mp = "pinn_vibration_model.pth"
    if os.path.exists(mp):
        kb = os.path.getsize(mp) / 1024
        st.success(f"Trained model ({kb:.0f} KB)")
    else:
        st.warning("No saved model\nRun training first")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.75rem;color:#546e7a;text-align:center">
      <a href="https://arxiv.org/abs/1711.10566" target="_blank"
         style="color:#4fc3f7;text-decoration:none">Raissi et al. 2019</a><br>
      Original PINN Paper
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-wrapper">
  <div class="hero-badge">Computational Physics × Deep Learning</div>
  <div class="gradient-title">Physics-Informed Neural Network</div>
  <div class="hero-sub">
    A neural network constrained by the laws of physics.<br>
    Solving PDEs, discovering hidden parameters, and generalising beyond data —
    where classical ML fails, PINN excels.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, icon, val, lbl in [
    (c1, "", "0.0001", "MAE vs Analytical [m]"),
    (c2, "", "&lt;1 ms", "Inference Time"),
    (c3, "", "500 KB", "Model Footprint"),
    (c4, "", "10×", "Data Efficiency vs DNN"),
]:
    col.markdown(
        f'<div class="metric-card"><div class="mc-icon">{icon}</div>'
        f'<div class="mc-val">{val}</div><div class="mc-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

t1, t2, t3, t4, t5 = st.tabs(
    ["Overview", "Simulator", "Live Training", "Analysis", "Theory"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with t1:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon"></span><span class="sec-hdr-text">What is a PINN?</span></div><div class="sec-div"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <p>A <strong style="color:#4fc3f7">Physics-Informed Neural Network</strong> encodes
          physical laws (PDEs) directly into the training objective via
          <strong style="color:#7c4dff">automatic differentiation</strong>.</p>
          <p style="margin-top:.6rem">Unlike black-box models, PINNs are
          <em>constrained</em> to satisfy governing equations, boundary conditions,
          and initial conditions — giving physically meaningful predictions even
          outside the training domain.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Physical Problem**")
        st.markdown("""
        <div class="info-card" style="border-color:rgba(124,77,255,.35)">
          <div style="font-family:'JetBrains Mono',monospace;color:#a5d6fb;font-size:.85rem">
            Damped Wave Equation (1-D)
          </div>
          <div style="font-family:'JetBrains Mono',monospace;color:#e8f0fe;font-size:.98rem;
                      margin:.5rem 0;padding:.5rem;background:rgba(0,0,0,.35);border-radius:6px">
            ∂²u/∂t² + 2ζω₀(∂u/∂t) + ω₀²u = 0
          </div>
          <div>
            <span class="pill">u — displacement</span>
            <span class="pill">ω₀ — natural freq</span>
            <span class="pill">ζ — damping ratio</span>
            <span class="pill">x ∈ [0,1] m</span>
            <span class="pill">t ∈ [0,5] s</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Network Architecture**")
        st.code(
            "Input  :  [x, t]  ──  ℝ²\n"
            "         ↓\n"
            "Dense(2→256) + Tanh\n"
            "Dense(256→256) + Tanh  × 5 layers\n"
            "         ↓\n"
            "Dense(256→1)\n"
            "         ↓\n"
            "Output :  u(x,t)  ──  ℝ¹   (displacement)\n\n"
            "Learnable physics params: ω₀, ζ  (inverse problem!)",
            language="text",
        )

    with right:
        st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon"></span><span class="sec-hdr-text">PINN vs Conventional Approaches</span></div><div class="sec-div"></div>', unsafe_allow_html=True)

        import pandas as pd
        df_cmp = pd.DataFrame({
            "Aspect": ["Data needed", "Physics constraint", "Extrapolation",
                       "Interpretability", "Parameters", "Training time"],
            "Standard DNN": ["Millions", "None (black-box)", "Poor",
                                "Low", "Black-box", "Hours"],
            "FEM / FDM": ["None (mesh)", "Full", "Good",
                             "High", "Prescribed", "Minutes–hours"],
            "PINN (this)": ["~5 000 pts", "PDE enforced", "Good",
                              "High", "Learned (ω₀, ζ)", "~10 min"],
        })
        st.dataframe(df_cmp.set_index("Aspect"), use_container_width=True, height=240)

        st.markdown('<div class="sec-hdr" style="margin-top:1.4rem"><span class="sec-hdr-icon"></span><span class="sec-hdr-text">Real-World Applications</span></div><div class="sec-div"></div>', unsafe_allow_html=True)

        for icon, title, desc in [
            ("", "Structural Health Monitoring", "Detect damage from vibration signatures"),
            ("", "Aerospace Engineering",         "Wing flutter & aeroelastic stability"),
            ("", "Seismic Analysis",              "Earthquake wave propagation"),
            ("", "Heat Transfer",                "Thermal management in electronics"),
            ("", "Fluid Dynamics",               "Navier-Stokes with scarce data"),
            ("", "Material Characterisation",    "Inverse identification of elastic moduli"),
        ]:
            st.markdown(
                f'<div class="feat-row"><span class="feat-icon">{icon}</span>'
                f'<div><span class="feat-accent">{title}</span><br>'
                f'<span class="feat-text">{desc}</span></div></div>',
                unsafe_allow_html=True,
            )

    # Quick-start steps
    st.markdown("---")
    st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon"></span><span class="sec-hdr-text">Quick Start</span></div><div class="sec-div"></div>', unsafe_allow_html=True)

    qa, qb, qc, qd = st.columns(4)
    for col, step, clr, cmd in [
        (qa, "Install", "#4fc3f7", "pip install torch numpy\nmatplotlib scipy\nstreamlit plotly"),
        (qb, "Train",   "#7c4dff", "python pinn_complete_starter.py\n# ~10 min on CPU"),
        (qc, "Demo",    "#00e5ff", "streamlit run app_demo.py\n# This app!"),
        (qd, "Extend",  "#ef9a9a", "python experiments/\nheat_equation_pinn.py"),
    ]:
        col.markdown(
            f'<div class="info-card" style="border-color:rgba(100,100,100,.3);text-align:center">'
            f'<div style="font-size:1.5rem;margin-bottom:.4rem">{step}</div>'
            f'<div class="code-snip">{cmd}</div></div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PHYSICS SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

with t2:
    st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon"></span><span class="sec-hdr-text">Interactive Physics Simulator</span></div><div class="sec-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="alert-box">Visualise u(x,t) — the displacement field of a damped vibrating beam. Adjust physical parameters and observe how energy dissipates in space and time.</div>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    omega_0   = sc1.slider("ω₀  Natural Frequency [rad/s]", 0.1, 6.0, 1.0, 0.1)
    zeta      = sc2.slider("ζ   Damping Ratio",              0.01, 0.95, 0.05, 0.01)
    amplitude = sc3.slider("A   Amplitude [m]",              0.01, 0.30, 0.10, 0.01)
    t_max     = sc4.slider("T   Simulation Time [s]",        2.0,  15.0, 5.0,  0.5)

    # Derived quantities
    wd  = omega_0 * np.sqrt(max(0, 1 - zeta**2))
    tau = 1 / (zeta * omega_0) if zeta * omega_0 > 0 else float("inf")

    if zeta < 0.999:
        regime, rc = f"Underdamped  (ζ={zeta:.2f}  →  oscillatory + decay)", "#4caf50"
    elif zeta < 1.001:
        regime, rc = "Critically Damped  (ζ=1.00  →  fastest return)", "#ffd54f"
    else:
        regime, rc = f"Overdamped  (ζ={zeta:.2f}  →  no oscillation)", "#ef5350"

    st.markdown(
        f'<div style="background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.1);'
        f'border-radius:8px;padding:.6rem 1.1rem;margin:.4rem 0;color:{rc};font-size:.9rem;">'
        f'{regime} &nbsp;|&nbsp; '
        f'<span style="color:#78909c">ωd = {wd:.3f} rad/s &nbsp;|&nbsp; τ = {tau:.2f} s</span></div>',
        unsafe_allow_html=True,
    )

    # Generate grid
    xg = np.linspace(0, 1, 130)
    tg = np.linspace(0, t_max, 190)
    XG, TG = np.meshgrid(xg, tg)
    UG = analytical(XG, TG, omega_0, zeta, amplitude)

    view = st.radio("View", ["3D Surface", "2D Contour", "Temporal Profiles", "All Views"],
                    horizontal=True)

    if "3D" in view:
        st.plotly_chart(
            surface_fig(UG, XG, TG, f"u(x,t) — ω₀={omega_0}, ζ={zeta}"),
            use_container_width=True,
        )

    elif "2D" in view:
        st.plotly_chart(
            contour_fig(UG, XG, TG, f"Displacement Field u(x,t)"),
            use_container_width=True,
        )

    elif "Temporal" in view:
        fig_t = go.Figure()
        for i, tv in enumerate(np.linspace(0, t_max, 8)):
            u_snap = amplitude * np.exp(-zeta * omega_0 * tv) * np.cos(omega_0 * tv) * np.sin(np.pi * xg)
            clr = px.colors.sequential.Plasma[int(i * 7 / 7)]
            fig_t.add_trace(go.Scatter(x=xg, y=u_snap, mode="lines",
                                       name=f"t={tv:.1f}s",
                                       line=dict(color=clr, width=2.5)))
        fig_t.update_layout(title="Spatial Profiles at Selected Times", height=430, **_LAYOUT)
        _apply_axes(fig_t, "Position x [m]", "Displacement u [m]")
        fig_t.update_layout(legend=dict(bgcolor="rgba(0,0,0,.3)", font=dict(color="#90caf9")))
        st.plotly_chart(fig_t, use_container_width=True)

    else:  # All views
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(surface_fig(UG, XG, TG, "3D Surface"), use_container_width=True)
        with cb:
            st.plotly_chart(contour_fig(UG, XG, TG, "2D Contour"), use_container_width=True)

        # Temporal decay at mid-span
        u_mid = amplitude * np.exp(-zeta * omega_0 * tg) * np.cos(omega_0 * tg) * np.sin(np.pi * 0.5)
        env_p = amplitude * np.exp(-zeta * omega_0 * tg) * np.sin(np.pi * 0.5)
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=tg, y=env_p,  mode="lines", name="Envelope +",
                                   line=dict(color="#ef9a9a", dash="dash", width=1.5)))
        fig_d.add_trace(go.Scatter(x=tg, y=-env_p, mode="lines", name="Envelope −",
                                   line=dict(color="#ef9a9a", dash="dash", width=1.5),
                                   fill="tonexty", fillcolor="rgba(239,154,154,.05)"))
        fig_d.add_trace(go.Scatter(x=tg, y=u_mid,  mode="lines", name="u(0.5, t)",
                                   line=dict(color="#4fc3f7", width=2.5)))
        fig_d.update_layout(title="Temporal Decay at mid-span (x=0.5 m)",
                            height=370, **_LAYOUT,
                            legend=dict(bgcolor="rgba(0,0,0,.3)", font=dict(color="#90caf9")))
        _apply_axes(fig_d, "Time t [s]", "Displacement u [m]")
        st.plotly_chart(fig_d, use_container_width=True)

    # PINN comparison (if model saved)
    model = load_model()
    if model is not None:
        st.markdown("---")
        st.markdown('<div class="alert-box">Saved PINN model detected — comparing prediction vs analytical below.</div>', unsafe_allow_html=True)

        ω0_m = model.omega_0.item()
        ζ_m  = model.zeta.item()
        x_p  = np.linspace(0, 1, 100)
        t_p  = np.linspace(0, 5, 140)
        Xp, Tp = np.meshgrid(x_p, t_p)

        U_pinn = model_predict_grid(model, Xp, Tp, DEVICE)
        U_anal = analytical(Xp, Tp, ω0_m, ζ_m)
        U_err  = np.abs(U_pinn - U_anal)

        col_a, col_b, col_c = st.columns(3)
        col_a.plotly_chart(contour_fig(U_pinn, Xp, Tp, f"PINN  (ω₀={ω0_m:.4f})"), use_container_width=True)
        col_b.plotly_chart(contour_fig(U_anal, Xp, Tp, "Analytical Solution"),     use_container_width=True)
        col_c.plotly_chart(contour_fig(U_err,  Xp, Tp, f"Error (MAE={U_err.mean():.2e})", cmap="YlOrRd"), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Learned ω₀",   f"{ω0_m:.4f}", f"{(ω0_m-1.0)*100:+.3f}% vs truth")
        m2.metric("Learned ζ",    f"{ζ_m:.4f}",  f"{(ζ_m-0.05)*100:+.3f}% vs truth")
        m3.metric("MAE", f"{U_err.mean():.2e} m")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE TRAINING
# ══════════════════════════════════════════════════════════════════════════════

with t3:
    st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon"></span><span class="sec-hdr-text">Live PINN Training</span></div><div class="sec-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="alert-box">Train a PINN in real-time. Watch physics drive the network to self-discover ω₀ and ζ from data — a live demonstration of the <em>inverse problem</em>.</div>', unsafe_allow_html=True)

    tr1, tr2, tr3 = st.columns(3)
    epochs_choice = tr1.select_slider("Epochs", [100, 200, 500, 1000, 2000], value=500)
    arch_choice   = tr2.selectbox("Architecture",
                                  ["Small (64×3) — Fast demo",
                                   "Medium (128×4) — Balanced",
                                   "Full (256×5) — High accuracy"])
    λ_phys        = tr3.slider("λ Physics weight", 0.1, 5.0, 1.0, .1)

    _arch_map = {
        "Small (64×3) — Fast demo":       (64,  3),
        "Medium (128×4) — Balanced":      (128, 4),
        "Full (256×5) — High accuracy":   (256, 5),
    }
    h_dim, n_lay = _arch_map[arch_choice]

    if st.button("Start Training", type="primary", use_container_width=True):
        device_tr = DEVICE

        with st.spinner("Generating synthetic data …"):
            data_tr = generate_synthetic_data(num_data=2_000, num_physics=5_000)
            for k in data_tr:
                if isinstance(data_tr[k], torch.Tensor):
                    data_tr[k] = data_tr[k].to(device_tr)

        model_tr  = PhysicsInformedNN(hidden_dim=h_dim, num_layers=n_lay, device=device_tr).to(device_tr)
        optimizer = torch.optim.Adam(model_tr.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.90)

        prog_bar = st.progress(0, text="Training …")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        ph_loss  = mcol1.empty()
        ph_phys  = mcol2.empty()
        ph_omega = mcol3.empty()
        ph_zeta  = mcol4.empty()
        chart_ph = st.empty()

        loss_h, phys_h, data_h_ = [], [], []
        omega_h, zeta_h = [], []
        t0 = time.time()
        UPDATE_EVERY = max(1, epochs_choice // 50)

        for ep in range(epochs_choice):
            optimizer.zero_grad()

            res, _ = model_tr.pde_residual(data_tr["x_physics"], data_tr["t_physics"])
            l_phys = torch.mean(res**2)

            xt_d = torch.stack([data_tr["x_data"], data_tr["t_data"]], dim=1)
            u_pd = model_tr(xt_d)
            l_data = torch.mean((u_pd - data_tr["u_data"].unsqueeze(1))**2)

            l_bc = model_tr.boundary_condition_loss(data_tr["x_bc"], data_tr["t_bc"])
            l_ic = model_tr.initial_condition_loss(data_tr["x_ic"], data_tr["u_ic"])

            l_tot = λ_phys * l_phys + 10.0 * l_data + l_bc + l_ic
            l_tot.backward()
            torch.nn.utils.clip_grad_norm_(model_tr.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if ep % UPDATE_EVERY == 0 or ep == epochs_choice - 1:
                frac = (ep + 1) / epochs_choice
                prog_bar.progress(frac, text=f"Epoch {ep+1}/{epochs_choice}")

                ov = model_tr.omega_0.item()
                zv = model_tr.zeta.item()
                lv = l_tot.item()
                pv = l_phys.item()

                loss_h.append(lv); phys_h.append(pv)
                data_h_.append(l_data.item()); omega_h.append(ov); zeta_h.append(zv)

                ph_loss.metric("Total Loss",   f"{lv:.3e}")
                ph_phys.metric("Physics Loss", f"{pv:.3e}")
                ph_omega.metric("Learned ω₀",  f"{ov:.4f}", "target 1.0")
                ph_zeta.metric("Learned ζ",    f"{zv:.4f}", "target 0.05")

                if len(loss_h) > 1:
                    fig_live = make_subplots(rows=1, cols=2,
                                            subplot_titles=["Loss (log)", "Learned Parameters"])
                    fig_live.add_trace(go.Scatter(y=loss_h,  mode="lines", name="Total",
                                                  line=dict(color="#4fc3f7", width=2)), 1, 1)
                    fig_live.add_trace(go.Scatter(y=phys_h,  mode="lines", name="Physics",
                                                  line=dict(color="#7c4dff", width=1.5, dash="dash")), 1, 1)
                    fig_live.add_trace(go.Scatter(y=omega_h, mode="lines", name="ω₀ (→1.0)",
                                                  line=dict(color="#4fc3f7", width=2.5)), 1, 2)
                    fig_live.add_trace(go.Scatter(y=zeta_h,  mode="lines", name="ζ  (→0.05)",
                                                  line=dict(color="#ef9a9a", width=2.5)), 1, 2)
                    fig_live.update_yaxes(type="log", row=1, col=1,
                                          gridcolor="rgba(79,195,247,.1)")
                    fig_live.update_yaxes(row=1, col=2,
                                          gridcolor="rgba(79,195,247,.1)")
                    fig_live.update_xaxes(gridcolor="rgba(79,195,247,.1)")
                    fig_live.update_layout(
                        height=320, **_LAYOUT,
                        legend=dict(bgcolor="rgba(0,0,0,.3)", font=dict(color="#90caf9")),
                    )
                    chart_ph.plotly_chart(fig_live, use_container_width=True)

        elapsed = time.time() - t0
        final_ω = model_tr.omega_0.item()
        final_ζ = model_tr.zeta.item()
        st.success(
            f"Done in {elapsed:.1f}s — "
            f"ω₀ = {final_ω:.4f} (err {abs(final_ω-1.0)*100:.2f}%) | "
            f"ζ = {final_ζ:.4f} (err {abs(final_ζ-0.05)*100:.2f}%)"
        )

        if st.button("Save Model to Disk", use_container_width=True):
            torch.save(model_tr.state_dict(), "pinn_vibration_model.pth")
            st.success("Model saved as `pinn_vibration_model.pth`")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with t4:
    st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon"></span><span class="sec-hdr-text">Results Analysis</span></div><div class="sec-div"></div>', unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.markdown('<div class="alert-warn">No trained model found. Train one using the <strong>Live Training</strong> tab, or run <code>python pinn_complete_starter.py</code> in your terminal.</div>', unsafe_allow_html=True)
    else:
        ω0_a = model.omega_0.item()
        ζ_a  = model.zeta.item()

        with st.spinner("Computing predictions …"):
            xa = np.linspace(0, 1, 120)
            ta = np.linspace(0, 5, 170)
            Xa, Ta = np.meshgrid(xa, ta)
            Up = model_predict_grid(model, Xa, Ta, DEVICE)
            Ua = analytical(Xa, Ta, ω0_a, ζ_a)
            Ue = np.abs(Up - Ua)
            mae = Ue.mean()

        # Summary metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE",      f"{mae:.2e} m")
        m2.metric("Max Error",f"{Ue.max():.2e} m")
        m3.metric("Rel Error",f"{(Ue/(np.abs(Ua)+1e-9)).mean()*100:.4f}%")
        m4.metric("ω₀ Error", f"{abs(ω0_a-1.0)*100:.4f}%")
        m5.metric("ζ Error",  f"{abs(ζ_a-0.05)*100:.4f}%")

        st.markdown("---")

        # Contour trio
        ca, cb, cc = st.columns(3)
        ca.plotly_chart(contour_fig(Up, Xa, Ta, "PINN Prediction"),                   use_container_width=True)
        cb.plotly_chart(contour_fig(Ua, Xa, Ta, "Analytical Solution"),               use_container_width=True)
        cc.plotly_chart(contour_fig(Ue, Xa, Ta, f"Error  (MAE={mae:.1e})", "YlOrRd"), use_container_width=True)

        # Spatial profiles comparison
        st.markdown("**Spatial Profiles: PINN vs Analytical**")
        times_p = [0.5, 1.0, 2.0, 3.0, 5.0]
        fig_pr = make_subplots(rows=1, cols=len(times_p),
                               subplot_titles=[f"t={v:.1f}s" for v in times_p])
        for i, tv in enumerate(times_p):
            x_tor = torch.tensor(xa, dtype=torch.float32).to(DEVICE)
            t_tor = torch.full_like(x_tor, tv)
            xt_pr = torch.stack([x_tor, t_tor], dim=1)
            with torch.no_grad():
                u_pr = model(xt_pr).cpu().numpy().flatten()
            u_an = analytical(xa, tv, ω0_a, ζ_a)
            fig_pr.add_trace(go.Scatter(x=xa, y=u_pr, mode="lines", name="PINN",
                                        line=dict(color="#4fc3f7", width=2.5),
                                        showlegend=(i == 0)), 1, i + 1)
            fig_pr.add_trace(go.Scatter(x=xa, y=u_an, mode="lines", name="Analytical",
                                        line=dict(color="#ef9a9a", dash="dash", width=1.5),
                                        showlegend=(i == 0)), 1, i + 1)
        fig_pr.update_xaxes(gridcolor="rgba(79,195,247,.1)", showticklabels=False)
        fig_pr.update_yaxes(gridcolor="rgba(79,195,247,.1)")
        fig_pr.update_layout(height=290, **_LAYOUT,
                             legend=dict(bgcolor="rgba(0,0,0,.3)", font=dict(color="#90caf9")))
        st.plotly_chart(fig_pr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — THEORY
# ══════════════════════════════════════════════════════════════════════════════

with t5:
    tc1, tc2 = st.columns(2, gap="large")

    with tc1:
        st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon">📐</span><span class="sec-hdr-text">Governing Equations</span></div><div class="sec-div"></div>', unsafe_allow_html=True)

        st.markdown("**Damped Oscillator (dimensional form)**")
        st.latex(r"m\ddot{u} + c\dot{u} + ku = 0")
        st.caption("m = mass · c = damping coeff · k = stiffness")

        st.markdown("**Normalised PDE (PINN target)**")
        st.latex(r"\frac{\partial^2 u}{\partial t^2} + 2\zeta\omega_0\,\frac{\partial u}{\partial t} + \omega_0^2\,u = 0")

        st.markdown("**Analytical Solution (1-D)**")
        st.latex(r"u(x,t) = A\,e^{-\zeta\omega_0 t}\cos(\omega_d t)\sin(\pi x)")
        st.latex(r"\omega_d = \omega_0\sqrt{1-\zeta^2} \quad (\text{damped frequency})")

        st.markdown("**Multi-Component Loss Function**")
        st.latex(
            r"\mathcal{L} = "
            r"\underbrace{\lambda_p\|\mathcal{N}[u]\|^2}_{\text{PDE residual}}"
            r"+ \underbrace{\lambda_d\|u - u^{\text{obs}}\|^2}_{\text{data fidelity}}"
            r"+ \underbrace{\|u_{\text{BC}}\|^2}_{\text{BC}}"
            r"+ \underbrace{\|u_{\text{IC}} - g(x)\|^2}_{\text{IC}}"
        )

        st.markdown("**5️⃣ Boundary and Initial Conditions**")
        st.latex(r"u(0,t) = u(1,t) = 0 \quad (\text{fixed ends})")
        st.latex(r"u(x,0) = A\sin(\pi x), \qquad \dot{u}(x,0)=0")

    with tc2:
        st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon"></span><span class="sec-hdr-text">Automatic Differentiation for PDEs</span></div><div class="sec-div"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <p><strong style="color:#4fc3f7">Key insight:</strong>
          Instead of finite differences, PINN computes
          <em>exact</em> partial derivatives of the network via PyTorch's
          <code>autograd</code> — enabling the PDE residual to flow directly
          into gradient descent.</p>
        </div>
        """, unsafe_allow_html=True)

        st.code(
            "# Compute ∂u/∂t and ∂²u/∂t² via autograd\n"
            "xt.requires_grad_(True)\n"
            "u = model(xt)\n\n"
            "# First derivative: ∂u/∂x, ∂u/∂t\n"
            "grads = torch.autograd.grad(\n"
            "    u, xt, grad_outputs=ones_like(u),\n"
            "    create_graph=True)[0]\n"
            "u_t = grads[:, 1:2]       # ∂u/∂t\n\n"
            "# Second derivative: ∂²u/∂t²\n"
            "u_tt = torch.autograd.grad(\n"
            "    u_t, xt, grad_outputs=ones_like(u_t),\n"
            "    create_graph=True)[0][:, 1:2]\n\n"
            "# PDE residual (should → 0)\n"
            "R = u_tt + 2*ζ*ω₀*u_t + ω₀²*u",
            language="python",
        )

        st.markdown('<div class="sec-hdr" style="margin-top:1.5rem"><span class="sec-hdr-icon">📚</span><span class="sec-hdr-text">Key References</span></div><div class="sec-div"></div>', unsafe_allow_html=True)

        refs = [
            ("Raissi, Perdikaris & Karniadakis (2019)",
             "Physics-informed neural networks. JCP 378:686–707.",
             "https://arxiv.org/abs/1711.10566"),
            ("Lagaris, Likas & Fotiadis (1998)",
             "Artificial neural networks for ODEs/PDEs. IEEE TNN 9(5):987–1000.",
             "https://doi.org/10.1109/72.712178"),
            ("Karniadakis et al. (2021)",
             "Physics-informed machine learning. Nature Reviews Physics 3:422–440.",
             "https://www.nature.com/articles/s42254-021-00314-5"),
            ("Cuomo et al. (2022)",
             "Scientific machine learning through PINNs: where we are and next. JML-R.",
             "https://arxiv.org/abs/2201.05624"),
        ]
        for i, (auth, title, url) in enumerate(refs):
            st.markdown(
                f'<div style="padding:.55rem .75rem;border-left:2px solid rgba(79,195,247,.4);'
                f'margin:.4rem 0;font-size:.82rem;color:#78909c">'
                f'[{i+1}] <a href="{url}" target="_blank" style="color:#4fc3f7;text-decoration:none">'
                f'{auth}</a>. {title}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("""
        <div class="info-card" style="margin-top:1rem;border-color:rgba(124,77,255,.3)">
          <strong style="color:#7c4dff">Computational Physics Advantage</strong>
          <p style="margin-top:.5rem;font-size:.88rem">
            PINN sits at the intersection of <strong>Numerical Methods</strong>
            (FEM, spectral methods, finite differences) taught in
            Computational Physics, and modern <strong>Deep Learning</strong>.
            A physics graduate's unique edge: understanding <em>why</em> the
            equations are structured as they are — not just how to code them.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1.2rem 0;color:#546e7a;font-size:.83rem">
  <strong style="color:#4fc3f7">PINN — Physics-Informed Neural Network</strong><br>
  Portfolio project by a Physics + Computational Physics graduate<br>
  Stack: PyTorch · Plotly · Streamlit · NumPy · SciPy<br><br>
  <a href="https://arxiv.org/abs/1711.10566" target="_blank"
     style="color:#4fc3f7;text-decoration:none">Raissi et al. 2019</a>
  &nbsp;|&nbsp;
  <a href="https://deepxde.readthedocs.io/" target="_blank"
     style="color:#4fc3f7;text-decoration:none">DeepXDE Library</a>
  &nbsp;|&nbsp;
  <a href="https://github.com" target="_blank"
     style="color:#4fc3f7;text-decoration:none">GitHub</a>
</div>
""", unsafe_allow_html=True)
