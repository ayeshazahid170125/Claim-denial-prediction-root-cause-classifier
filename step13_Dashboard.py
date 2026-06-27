"""
STEP 13 - Streamlit Dashboard (Client-Ready Design)
Claim Denial Prediction System
Run: streamlit run step13_Dashboard.py
Requires: uvicorn step12_Fastapi:app --reload --port 8000
"""

from pathlib import Path
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000"
BASE_DIR = Path(__file__).resolve().parent
NLP_DIR  = BASE_DIR / "nlp_outputs"
MODEL_DIR = BASE_DIR / "model_outputs"

st.set_page_config(
    page_title="Claim Denial Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── THEME STATE ──────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Design tokens for each theme. Only these values differ between
# Light Mode and Dark Mode — every CSS rule below references the
# resulting CSS variables, so the whole UI repaints consistently.
THEME_TOKENS = {
    "dark": {
        "bg-page":              "#0B0F19",
        "bg-navbar":            "rgba(6,10,19,0.85)",
        "bg-card":              "rgba(21,29,48,0.45)",
        "bg-form-panel":        "rgba(21,29,48,0.34)",
        "bg-input":             "#111827",
        "bg-input-hover":       "#162033",
        "bg-track":             "rgba(255,255,255,0.05)",
        "shadow-color":         "rgba(0,0,0,0.3)",
        "card-hover-shadow":    "rgba(59,130,246,0.15)",
        "border-color":         "rgba(255,255,255,0.06)",
        "border-color-soft":    "rgba(255,255,255,0.04)",
        "border-color-strong":  "rgba(255,255,255,0.08)",
        "input-border":         "#1E293B",
        "text-primary":         "#F8FAFC",
        "text-secondary":       "#94A3B8",
        "text-tertiary":        "#CBD5E1",
        "text-muted":           "#64748B",
        "text-faint":           "#475569",
        "input-text":           "#F1F5F9",
        "accent":               "#3B82F6",
        "accent-light":         "#60A5FA",
        "accent-lighter":       "#38BDF8",
        "success-text":         "#34D399",
        "warning-text":         "#FBBF24",
        "danger-text":          "#F87171",
        "nav-logo-bg-1":        "rgba(37,99,235,0.3)",
        "nav-logo-bg-2":        "rgba(13,148,136,0.3)",
        "nav-logo-border":      "rgba(59,130,246,0.35)",
        "nav-badge-bg":         "rgba(59,130,246,0.08)",
        "nav-badge-border":     "rgba(59,130,246,0.18)",
        "nav-badge-text":       "#60A5FA",
        "top-header-bg-1":      "rgba(37,99,235,0.12)",
        "top-header-bg-2":      "rgba(13,148,136,0.12)",
        "top-header-border":    "rgba(59,130,246,0.15)",
        "cause-bg-1":           "rgba(30,58,138,0.2)",
        "cause-bg-2":           "rgba(6,78,59,0.2)",
        "cause-border":         "rgba(59,130,246,0.2)",
        "cause-label":          "#60A5FA",
        "cause-conf":           "#93C5FD",
        "fix-label-text":       "#34D399",
        "fix-text-text":        "#A7F3D0",
        "pill-bg":              "rgba(255,255,255,0.04)",
        "pill-border":          "rgba(255,255,255,0.08)",
        "pill-text":            "#CBD5E1",
        "badge-high-bg":        "rgba(239,68,68,0.1)",
        "badge-high-text":      "#F87171",
        "badge-high-border":    "rgba(239,68,68,0.3)",
        "badge-med-bg":         "rgba(245,158,11,0.1)",
        "badge-med-text":       "#FBBF24",
        "badge-med-border":     "rgba(245,158,11,0.3)",
        "badge-low-bg":         "rgba(16,185,129,0.1)",
        "badge-low-text":       "#34D399",
        "badge-low-border":     "rgba(16,185,129,0.3)",
        "ready-icon-bg-1":      "rgba(37,99,235,0.15)",
        "ready-icon-bg-2":      "rgba(13,148,136,0.15)",
        "ready-icon-border":    "rgba(59,130,246,0.2)",
        "ready-subtitle-strong":"#94A3B8",
        "feature-bg":           "rgba(255,255,255,0.02)",
        "feature-border":       "rgba(255,255,255,0.04)",
        "fi-blue-bg":           "rgba(59,130,246,0.12)",
        "fi-blue-border":       "rgba(59,130,246,0.15)",
        "fi-blue-text":         "#38BDF8",
        "fi-amber-bg":          "rgba(251,191,36,0.10)",
        "fi-amber-border":      "rgba(251,191,36,0.15)",
        "fi-amber-text":        "#FBBF24",
        "fi-green-bg":          "rgba(52,211,153,0.10)",
        "fi-green-border":      "rgba(52,211,153,0.15)",
        "fi-green-text":        "#34D399",
        "fi-red-bg":            "rgba(248,113,113,0.10)",
        "fi-red-border":        "rgba(248,113,113,0.15)",
        "fi-red-text":          "#F87171",
        "toggle-bg":            "rgba(21,29,48,0.7)",
        "toggle-border":        "rgba(255,255,255,0.12)",
        "toggle-text":          "#F8FAFC",
    },
    "light": {
        "bg-page":              "#F4F6FB",
        "bg-navbar":            "rgba(255,255,255,0.88)",
        "bg-card":              "rgba(255,255,255,0.85)",
        "bg-form-panel":        "rgba(255,255,255,0.62)",
        "bg-input":             "#FFFFFF",
        "bg-input-hover":       "#F8FAFC",
        "bg-track":             "rgba(15,23,42,0.07)",
        "shadow-color":         "rgba(15,23,42,0.08)",
        "card-hover-shadow":    "rgba(37,99,235,0.12)",
        "border-color":         "rgba(15,23,42,0.08)",
        "border-color-soft":    "rgba(15,23,42,0.05)",
        "border-color-strong":  "rgba(15,23,42,0.12)",
        "input-border":         "#CBD5E1",
        "text-primary":         "#0F172A",
        "text-secondary":       "#475569",
        "text-tertiary":        "#334155",
        "text-muted":           "#64748B",
        "text-faint":           "#94A3B8",
        "input-text":           "#0F172A",
        "accent":               "#2563EB",
        "accent-light":         "#1D4ED8",
        "accent-lighter":       "#2563EB",
        "success-text":         "#047857",
        "warning-text":         "#92400E",
        "danger-text":          "#B91C1C",
        "nav-logo-bg-1":        "rgba(37,99,235,0.16)",
        "nav-logo-bg-2":        "rgba(13,148,136,0.16)",
        "nav-logo-border":      "rgba(37,99,235,0.3)",
        "nav-badge-bg":         "rgba(37,99,235,0.08)",
        "nav-badge-border":     "rgba(37,99,235,0.2)",
        "nav-badge-text":       "#1D4ED8",
        "top-header-bg-1":      "rgba(37,99,235,0.07)",
        "top-header-bg-2":      "rgba(13,148,136,0.07)",
        "top-header-border":    "rgba(37,99,235,0.16)",
        "cause-bg-1":           "rgba(37,99,235,0.06)",
        "cause-bg-2":           "rgba(13,148,136,0.06)",
        "cause-border":         "rgba(37,99,235,0.18)",
        "cause-label":          "#1D4ED8",
        "cause-conf":           "#2563EB",
        "fix-label-text":       "#047857",
        "fix-text-text":        "#065F46",
        "pill-bg":              "rgba(15,23,42,0.04)",
        "pill-border":          "rgba(15,23,42,0.09)",
        "pill-text":            "#334155",
        "badge-high-bg":        "rgba(239,68,68,0.1)",
        "badge-high-text":      "#B91C1C",
        "badge-high-border":    "rgba(239,68,68,0.32)",
        "badge-med-bg":         "rgba(217,119,6,0.12)",
        "badge-med-text":       "#92400E",
        "badge-med-border":     "rgba(217,119,6,0.32)",
        "badge-low-bg":         "rgba(5,150,105,0.1)",
        "badge-low-text":       "#047857",
        "badge-low-border":     "rgba(5,150,105,0.32)",
        "ready-icon-bg-1":      "rgba(37,99,235,0.1)",
        "ready-icon-bg-2":      "rgba(13,148,136,0.1)",
        "ready-icon-border":    "rgba(37,99,235,0.2)",
        "ready-subtitle-strong":"#334155",
        "feature-bg":           "rgba(15,23,42,0.02)",
        "feature-border":       "rgba(15,23,42,0.05)",
        "fi-blue-bg":           "rgba(37,99,235,0.1)",
        "fi-blue-border":       "rgba(37,99,235,0.18)",
        "fi-blue-text":         "#1D4ED8",
        "fi-amber-bg":          "rgba(217,119,6,0.1)",
        "fi-amber-border":      "rgba(217,119,6,0.18)",
        "fi-amber-text":        "#92400E",
        "fi-green-bg":          "rgba(5,150,105,0.1)",
        "fi-green-border":      "rgba(5,150,105,0.18)",
        "fi-green-text":        "#047857",
        "fi-red-bg":            "rgba(220,38,38,0.1)",
        "fi-red-border":        "rgba(220,38,38,0.18)",
        "fi-red-text":          "#B91C1C",
        "toggle-bg":            "rgba(255,255,255,0.9)",
        "toggle-border":        "rgba(15,23,42,0.12)",
        "toggle-text":          "#0F172A",
    },
}

theme = st.session_state.theme
tokens = THEME_TOKENS[theme]
_root_vars = "\n".join(f"    --{k}: {v};" for k, v in tokens.items())
st.markdown(f"<style>\n:root {{\n{_root_vars}\n}}\n</style>", unsafe_allow_html=True)

# Icon-only circular toggle — clean, attractive, and renders reliably
# (st.button gives us a single well-documented DOM node to style,
# unlike st.toggle's internal BaseWeb switch markup which varies
# across Streamlit versions).
toggle_icon = "☀️" if theme == "dark" else "🌙"
if st.button(toggle_icon, key="theme_toggle_btn", help="Switch to " + ("Light" if theme == "dark" else "Dark") + " Mode"):
    st.session_state.theme = "light" if theme == "dark" else "dark"
    theme = st.session_state.theme
tokens = THEME_TOKENS[theme]
_root_vars = "\n".join(f"    --{k}: {v};" for k, v in tokens.items())
st.markdown(f"<style>\n:root {{\n{_root_vars}\n}}\n</style>", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) none;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}
.st-emotion-cache-11xx4re {
    -webkit-box-align: center;
    align-items: center;
    background-color: rgb(255, 75, 75) !important;
    border-radius: 100%;
    border-style: none;
    display: flex;
    -webkit-box-pack: center;
    justify-content: center;
    height: 0.75rem;
    width: 0.75rem;
    box-shadow: none;
}
            
html, body {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
a, button, select, input, textarea, .card, .stat-card, .pill {
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Hide sidebar completely */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Main background */
.main { background: var(--bg-page) !important; }
[data-testid="stAppViewContainer"] { background: var(--bg-page) !important; }

/* Hide defaults */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { display: none !important; }

/* ── THEME TOGGLE BUTTON — circular icon FAB, premium glassy feel ── */
div[data-testid="stButton"] {
    position: fixed;
    top: 14px;
    right: 32px;
    z-index: 1001;
    width: auto !important;
}
div[data-testid="stButton"] > button {
    position: relative;
    background: linear-gradient(150deg, var(--toggle-bg), var(--bg-card)) !important;
    border: 1.5px solid var(--toggle-border) !important;
    color: var(--toggle-text) !important;
    border-radius: 50% !important;
    width: 46px !important;
    height: 46px !important;
    min-width: 46px !important;
    min-height: 46px !important;
    padding: 0 !important;
    line-height: 1 !important;
    font-size: 19px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow:
        0 4px 18px var(--shadow-color),
        0 0 0 1px var(--border-color-soft) inset !important;
    transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.3s ease,
                border-color 0.3s ease !important;
}
div[data-testid="stButton"] > button p {
    font-size: 19px !important;
    line-height: 1 !important;
    margin: 0 !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    box-shadow:
        0 6px 22px var(--card-hover-shadow),
        0 0 16px var(--card-hover-shadow),
        0 0 0 1px var(--accent) inset !important;
    transform: translateY(-2px) rotate(18deg) scale(1.08) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) rotate(18deg) scale(0.92) !important;
}
div[data-testid="stButton"] > button:focus {
    outline: none !important;
    box-shadow:
        0 4px 18px var(--shadow-color),
        0 0 0 3px var(--card-hover-shadow) !important;
}

/* ── NAVBAR ── */
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: var(--bg-navbar);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-color);
    padding: 0 190px 0 36px;
    height: 62px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, var(--nav-logo-bg-1), var(--nav-logo-bg-2));
    border: 1.5px solid var(--nav-logo-border);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: var(--accent-lighter);
}
.nav-tagline {
    font-size: 11px; font-weight: 600; color: var(--text-faint);
    text-transform: uppercase; letter-spacing: 0.08em;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 20px;
}
.nav-status {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 13px; font-weight: 500;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
}
.status-connected {
    background-color: #10B981;
    box-shadow: 0 0 8px #10B981;
    animation: pulse-c 2s infinite;
}
.status-disconnected {
    background-color: #EF4444;
    box-shadow: 0 0 8px #EF4444;
    animation: pulse-d 2s infinite;
}
@keyframes pulse-c {
    0%   { box-shadow: 0 0 0 0   rgba(16,185,129,0.7); }
    70%  { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0   rgba(16,185,129,0); }
}
@keyframes pulse-d {
    0%   { box-shadow: 0 0 0 0   rgba(239,68,68,0.7); }
    70%  { box-shadow: 0 0 0 6px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0   rgba(239,68,68,0); }
}
.nav-badge {
    background: var(--nav-badge-bg);
    border: 1px solid var(--nav-badge-border);
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    color: var(--nav-badge-text);
    letter-spacing: 0.04em;
}

/* Top header banner */
.top-header {
    background: linear-gradient(135deg, var(--top-header-bg-1) 0%, var(--top-header-bg-2) 100%) !important;
    border: 1px solid var(--top-header-border) !important;
    padding: 32px 36px !important;
    border-radius: 16px !important;
    margin-bottom: 30px !important;
    box-shadow: 0 8px 32px 0 var(--shadow-color) !important;
    backdrop-filter: blur(8px);
}
.top-header h1 {
    color: var(--text-primary) !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
}
.top-header p {
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    margin: 6px 0 0 0 !important;
}

/* Cards */
.card, .stat-card {
    background: var(--bg-card) !important;
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px 0 var(--shadow-color) !important;
    margin-bottom: 20px !important;
}
.card:hover, .stat-card:hover {
    border-color: rgba(59,130,246,0.35) !important;
    box-shadow: 0 8px 32px 0 var(--card-hover-shadow) !important;
    transform: translateY(-2px);
}
.card-title {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 20px !important;
}

/* Risk badges */
.badge-HIGH {
    display: inline-block;
    background: var(--badge-high-bg) !important; color: var(--badge-high-text) !important;
    border: 1.5px solid var(--badge-high-border) !important;
    padding: 6px 22px !important; border-radius: 100px !important;
    font-weight: 700 !important; font-size: 14px !important;
    letter-spacing: 0.08em !important; box-shadow: 0 0 15px rgba(239,68,68,0.15) !important;
}
.badge-MEDIUM {
    display: inline-block;
    background: var(--badge-med-bg) !important; color: var(--badge-med-text) !important;
    border: 1.5px solid var(--badge-med-border) !important;
    padding: 6px 22px !important; border-radius: 100px !important;
    font-weight: 700 !important; font-size: 14px !important;
    letter-spacing: 0.08em !important; box-shadow: 0 0 15px rgba(245,158,11,0.15) !important;
}
.badge-LOW {
    display: inline-block;
    background: var(--badge-low-bg) !important; color: var(--badge-low-text) !important;
    border: 1.5px solid var(--badge-low-border) !important;
    padding: 6px 22px !important; border-radius: 100px !important;
    font-weight: 700 !important; font-size: 14px !important;
    letter-spacing: 0.08em !important; box-shadow: 0 0 15px rgba(16,185,129,0.15) !important;
}

/* Driver bars */
.driver-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.driver-name { font-family: 'Fira Code', monospace !important; font-size: 12px !important; color: var(--text-tertiary) !important; min-width: 190px !important; }
.driver-bar-wrap { flex: 1; height: 8px; background: var(--bg-track); border-radius: 6px; overflow: hidden; }
.driver-bar-high { height: 8px; background: linear-gradient(90deg, #EF4444, #F87171) !important; border-radius: 6px; }
.driver-bar-low  { height: 8px; background: linear-gradient(90deg, #10B981, #34D399) !important; border-radius: 6px; }
.driver-val { font-family: 'Fira Code', monospace !important; font-size: 12px !important; color: var(--text-secondary) !important; min-width: 60px !important; text-align: right !important; }

/* Cause card */
.cause-card {
    background: linear-gradient(135deg, var(--cause-bg-1) 0%, var(--cause-bg-2) 100%) !important;
    border: 1px solid var(--cause-border) !important;
    border-radius: 16px !important; padding: 24px !important;
    margin-bottom: 16px !important; box-shadow: 0 8px 32px 0 var(--shadow-color) !important;
}
.cause-label { font-size: 18px !important; font-weight: 700 !important; color: var(--cause-label) !important; margin-bottom: 4px !important; }
.cause-conf { font-size: 13px !important; color: var(--cause-conf) !important; font-weight: 500 !important; }
.fix-label { font-size: 11px !important; font-weight: 700 !important; color: var(--fix-label-text) !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
.fix-text { font-size: 14px !important; color: var(--fix-text-text) !important; margin-top: 6px !important; line-height: 1.6 !important; }
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.pill {
    background: var(--pill-bg) !important; border: 1px solid var(--pill-border) !important;
    border-radius: 100px !important; padding: 6px 14px !important;
    font-size: 12px !important; color: var(--pill-text) !important;
}

/* Form section labels */
[data-testid="stForm"] {
    background: var(--bg-form-panel) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 18px !important;
    padding: 22px 22px 24px 22px !important;
    box-shadow: 0 10px 34px var(--shadow-color) !important;
}

.form-section {
    font-size: 12px !important; font-weight: 700 !important; color: var(--accent-light) !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    margin: 28px 0 12px 0 !important;
    border-bottom: 1px solid var(--top-header-border) !important; padding-bottom: 6px !important;
}

/* Submit button */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563EB, #0D9488) !important;
    color: #FFFFFF !important; border: none !important;
    border-radius: 12px !important; padding: 16px 0 !important;
    font-size: 16px !important; font-weight: 600 !important; width: 100% !important;
    letter-spacing: 0.05em !important; box-shadow: 0 4px 20px rgba(37,99,235,0.25) !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #0F766E) !important;
    box-shadow: 0 6px 24px rgba(37,99,235,0.4) !important;
    transform: translateY(-2px) !important;
}

/* Inputs */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: var(--bg-input) !important;
    border: 1.5px solid var(--input-border) !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] div {
    background-color: transparent !important;
    color: var(--input-text) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: var(--input-text) !important;
    fill: var(--input-text) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:hover {
    background: var(--bg-input-hover) !important;
    border-color: var(--border-color-strong) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: var(--bg-input) !important;
    color: var(--input-text) !important;
    border: 1.5px solid var(--input-border) !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04) !important;
}
[data-testid="stTextInput"] input:hover,
[data-testid="stTextArea"] textarea:hover {
    background-color: var(--bg-input-hover) !important;
    border-color: var(--border-color-strong) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    background-color: var(--bg-input) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-faint) !important;
    opacity: 0.75 !important;
}
label[data-testid="stWidgetLabel"] p { color: var(--text-secondary) !important; font-weight: 600 !important; font-size: 13px !important; }
[data-testid="stCheckbox"] label {
    align-items: center !important;
}
[data-testid="stCheckbox"] label p { color: var(--text-tertiary) !important; font-weight: 500 !important; }
[data-testid="stCheckbox"] [data-testid="stCheckboxRoot"] > div {
    border-color: var(--input-border) !important;
    background: var(--bg-input) !important;
}
[data-testid="stCheckbox"] [data-testid="stCheckboxRoot"] > div[aria-checked="true"] {
    border-color: var(--accent) !important;
    background: var(--accent) !important;
}
[data-testid="stSlider"] [data-testid="stWidgetLabel"] { min-height: 40px !important; display: flex !important; align-items: flex-end !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child { background: var(--bg-track) !important; }
[data-testid="stSlider"] [role="slider"] {
    # background: var(--accent) !important;
    # border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* Dropdown / select menus (BaseWeb portals) */
[data-baseweb="popover"] div[role="listbox"],
[data-baseweb="menu"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color-strong) !important;
    box-shadow: 0 16px 36px var(--shadow-color) !important;
}
[data-baseweb="menu"] li,
ul[role="listbox"] li {
    color: var(--input-text) !important;
    background: var(--bg-input) !important;
}
[data-baseweb="menu"] li:hover,
ul[role="listbox"] li:hover {
    background: var(--bg-input-hover) !important;
}

/* Disclaimer */
.disclaimer-bar {
    background: var(--bg-track) !important; border: 1px solid var(--border-color) !important;
    border-radius: 10px !important; padding: 12px 18px !important;
    font-size: 11px !important; color: var(--text-muted) !important;
    margin-top: 16px !important; line-height: 1.5 !important;
}

/* Ready card */
.ready-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 36px 24px;
    box-shadow: 0 8px 32px 0 var(--shadow-color);
    margin-top: 24px;
    text-align: center;
}
.ready-icon {
    width: 52px; height: 52px; border-radius: 14px;
    background: linear-gradient(135deg, var(--ready-icon-bg-1), var(--ready-icon-bg-2));
    border: 1.5px solid var(--ready-icon-border);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 18px auto; font-size: 26px; color: var(--accent-lighter);
}
.ready-title { font-size: 19px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; letter-spacing: -0.02em; }
.ready-subtitle { font-size: 13px; color: var(--text-muted); line-height: 1.6; margin-bottom: 28px; }
.ready-subtitle strong { color: var(--ready-subtitle-strong); }
.feature-list { display: flex; flex-direction: column; gap: 10px; text-align: left; }
.feature-item {
    display: flex; align-items: center; gap: 14px; padding: 14px 16px;
    background: var(--feature-bg); border: 1px solid var(--feature-border); border-radius: 12px;
}
.feature-icon {
    width: 38px; height: 38px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 20px;
}
.fi-blue  { background: var(--fi-blue-bg); border: 1px solid var(--fi-blue-border); color: var(--fi-blue-text); }
.fi-amber { background: var(--fi-amber-bg); border: 1px solid var(--fi-amber-border); color: var(--fi-amber-text); }
.fi-green { background: var(--fi-green-bg); border: 1px solid var(--fi-green-border); color: var(--fi-green-text); }
.fi-red   { background: var(--fi-red-bg); border: 1px solid var(--fi-red-border); color: var(--fi-red-text); }
.feature-title { font-size: 14px; font-weight: 600; color: var(--text-primary); }
.feature-desc  { font-size: 12px; color: var(--text-muted); margin-top: 3px; }
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────
def api_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=4).status_code == 200
    except Exception:
        return False

def call_api(payload):
    try:
        r = requests.post(f"{API_URL}/predict/full", json=payload, timeout=60)
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        st.error("FastAPI offline — run: uvicorn step12_Fastapi:app --reload --port 8000")
        return None

def risk_color(tier):
    palette = {
        "dark":  {"HIGH": "#F87171", "MEDIUM": "#FBBF24", "LOW": "#34D399"},
        "light": {"HIGH": "#B91C1C", "MEDIUM": "#92400E", "LOW": "#047857"},
    }
    return palette[st.session_state.get("theme", "dark")].get(tier, "#94A3B8")

def gauge_theme():
    if st.session_state.get("theme", "dark") == "light":
        return {
            "tickcolor": "#94A3B8",
            "tickfont": "#475569",
            "track_bg": "rgba(15,23,42,0.07)",
            "title_font": "#475569",
            "threshold": "#0F172A",
            "steps": ["rgba(5,150,105,0.08)", "rgba(217,119,6,0.08)", "rgba(220,38,38,0.08)"],
        }
    return {
        "tickcolor": "#475569",
        "tickfont": "#94A3B8",
        "track_bg": "rgba(255,255,255,0.05)",
        "title_font": "#94A3B8",
        "threshold": "#FFFFFF",
        "steps": ["rgba(16,185,129,0.06)", "rgba(245,158,11,0.06)", "rgba(239,68,68,0.06)"],
    }


# ── NAVBAR ──────────────────────────────────────────────────
ok = api_health()
status_dot_cls = "status-connected" if ok else "status-disconnected"
status_text    = "<span style='color:var(--success-text);'>API Connected</span>" if ok else "<span style='color:var(--danger-text);'>API Offline</span>"

st.markdown(f"""
<div class="navbar">
    <div class="nav-brand">
        <div class="nav-logo">&#9889;</div>
        <div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:18px; font-weight:800;
                color:var(--text-primary); letter-spacing:-0.03em; line-height:1.1;">
                Claim Denial Intelligence
            </div>
            <div class="nav-tagline">CMS Medicare PUF 2023 &middot; X12 RARC</div>
        </div>
    </div>
    <div class="nav-right">
        <span class="nav-badge">&#128736; Live Demo</span>
        <div class="nav-status">
            <span class="status-dot {status_dot_cls}"></span>
            {status_text}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.warning("⚠️ FastAPI backend offline — run: `uvicorn step12_Fastapi:app --reload --port 8000`")


# ══════════════════════════════════════════════════════════════
# PAGE — LIVE DEMO ONLY
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-header">
    <div>
        <h1>Claim Denial Predictor</h1>
        <p>Enter claim details &mdash; get denial probability + root cause before you submit</p>
    </div>
</div>
""", unsafe_allow_html=True)

col_form, col_gap, col_result = st.columns([5, 0.3, 5])

with col_form:
    with st.form("claim_form"):

        st.markdown('<div class="form-section">Provider Info</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            provider_type = st.selectbox("Provider Type", [
                "Internal Medicine", "Family Practice", "Cardiology",
                "Orthopedic Surgery", "Psychiatry", "Neurology",
                "Emergency Medicine", "Radiology", "Anesthesiology",
                "Certified Registered Nurse Anesthetist (CRNA)",
                "Clinical Laboratory", "Medical Oncology",
                "Pathology", "Hematology-Oncology", "Nuclear Medicine",
            ])
            state      = st.selectbox("State", ["CA","TX","FL","NY","IL","PA","OH","GA","NC","MI","WA","AZ","CO","NV","OR"])
            credential = st.selectbox("Credential Group", ["Physician","NP","PA","CRNA","Therapist","Other_Doctor","Nursing_Other"])
        with c2:
            hcpcs = st.selectbox("HCPCS / CPT Code", [
                "99213 — Office Visit (Low Risk)",
                "99214 — Office Visit Detailed (Low Risk)",
                "J1200 — Drug Injection (High Risk)",
                "88300 — Pathology Exam (High Risk)",
                "A9575 — Radiopharmaceutical (High Risk)",
                "96127 — Behavioral Screening (High Risk)",
                "51798 — Bladder Scan (High Risk)",
            ])
            entity_type   = st.selectbox("Entity Type", ["Individual (1)", "Organization (0)"])
            participating = st.selectbox("Medicare Participating", ["Yes (1)", "No (0)"])

        st.markdown('<div class="form-section">Utilization</div>', unsafe_allow_html=True)
        u1, u2, u3 = st.columns(3)
        with u1:
            tot_srvcs  = st.slider("Services (log)",  0.0, 10.0, 3.5, 0.1)
        with u2:
            avg_charge = st.slider("Charge (log)",    0.0, 10.0, 5.0, 0.1)
        with u3:
            avg_payment = st.slider("Payment (log)",  0.0, 10.0, 4.5, 0.1)

        st.markdown('<div class="form-section">Denial Remark — for Root Cause (optional)</div>', unsafe_allow_html=True)
        remark_text = st.text_area("Denial Remark Text",
            placeholder="e.g. Prior authorization was required but not obtained before service was rendered...",
            height=80, label_visibility="collapsed")

        st.markdown('<div class="form-section">Risk Flags</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1: flag_charge = st.checkbox("Charge < Payment")
        with f2: flag_zero   = st.checkbox("Zero Payment")
        with f3: flag_srvcs  = st.checkbox("Services < Benes")

        drug = st.selectbox("Drug Claim",        ["No (0)", "Yes (1)"])
        pos  = st.selectbox("Place of Service",  ["Office (0)", "Facility (1)"])

        submitted = st.form_submit_button("⚡ Analyze Denial Risk", use_container_width=True)

with col_result:
    if submitted:
        hcpcs_code = hcpcs.split(" — ")[0].strip()
        payload = {
            "provider_type":            provider_type,
            "state":                    state,
            "credential_group":         credential,
            "entity_type":              1 if "1" in entity_type else 0,
            "participating":            1 if "1" in participating else 0,
            "hcpcs_code":               hcpcs_code,
            "drug_indicator":           1 if "1" in drug else 0,
            "place_of_service":         1 if "1" in pos else 0,
            "total_services_log":       tot_srvcs,
            "avg_submitted_charge_log": avg_charge,
            "avg_medicare_payment_log": avg_payment,
            "avg_allowed_amount_log":   avg_payment + 0.1,
            "total_beneficiaries_log":  tot_srvcs - 0.5,
            "flag_charge_lt_payment":   int(flag_charge),
            "flag_zero_payment":        int(flag_zero),
            "flag_services_lt_benes":   int(flag_srvcs),
            "flag_zero_allowed":        0,
            "flag_invalid_state":       0,
            "flag_non_us_country":      0,
            "ruca_code":                1.0,
            "remark_text":              remark_text,
        }

        with st.spinner("Analyzing claim..."):
            result = call_api(payload)

        if result:
            risk  = result.get("denial_risk")
            cause = result.get("root_cause")

            if risk:
                tier = risk["risk_tier"]
                prob = risk["denial_probability"]
                gt = gauge_theme()

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%", "font": {"size": 44, "family": "Plus Jakarta Sans, sans-serif", "color": risk_color(tier)}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": gt["tickcolor"], "tickwidth": 1, "tickfont": {"color": gt["tickfont"]}},
                        "bar":  {"color": risk_color(tier), "thickness": 0.28},
                        "bgcolor": gt["track_bg"],
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,  40],  "color": gt["steps"][0]},
                            {"range": [40, 70],  "color": gt["steps"][1]},
                            {"range": [70, 100], "color": gt["steps"][2]},
                        ],
                        "threshold": {
                            "line": {"color": gt["threshold"], "width": 2},
                            "thickness": 0.8,
                            "value": round(risk["threshold_used"] * 100),
                        },
                    },
                    title={"text": "Denial Probability", "font": {"size": 13, "color": gt["title_font"], "family": "Plus Jakarta Sans, sans-serif"}},
                ))
                fig.update_layout(
                    height=240,
                    margin=dict(t=50, b=0, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    f"<div style='text-align:center;margin-top:-8px;margin-bottom:20px'>"
                    f"<span class='badge-{tier}'>{tier} RISK</span></div>",
                    unsafe_allow_html=True,
                )

                if risk.get("top3_drivers"):
                    st.markdown('<div class="card-title">Top Denial Drivers</div>', unsafe_allow_html=True)
                    max_val = max(abs(d["shap_value"]) for d in risk["top3_drivers"]) or 1
                    for d in risk["top3_drivers"]:
                        pct     = min(int(abs(d["shap_value"]) / max_val * 100), 100)
                        bar_cls = "driver-bar-high" if d["direction"] == "increases_risk" else "driver-bar-low"
                        arrow   = "&#8679;" if d["direction"] == "increases_risk" else "&#8681;"
                        st.markdown(f"""
                        <div class="driver-row">
                            <div class="driver-name">{arrow} {d['feature']}</div>
                            <div class="driver-bar-wrap"><div class="{bar_cls}" style="width:{pct}%"></div></div>
                            <div class="driver-val">{d['shap_value']:+.2f}</div>
                        </div>""", unsafe_allow_html=True)

            if cause:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="cause-card">
                    <div style="display:flex; align-items:center; gap:14px; margin-bottom:14px;">
                        <div style="width:38px; height:38px; border-radius:10px; flex-shrink:0;
                            background:rgba(96,165,250,0.12); border:1px solid rgba(96,165,250,0.2);
                            display:flex; align-items:center; justify-content:center;
                            font-size:20px; color:var(--cause-label);">&#9670;</div>
                        <div>
                            <div class="cause-label">{cause['display_name']}</div>
                            <div class="cause-conf">Confidence: {cause['confidence']:.1%}</div>
                        </div>
                    </div>
                    <div style="padding-top:14px; border-top:1px solid var(--border-color);">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                            <span style="color:var(--fix-label-text); font-size:14px;">&#10022;</span>
                            <span class="fix-label">First-Pass Fix</span>
                        </div>
                        <div class="fix-text">{cause['first_pass_fix']}</div>
                    </div>
                    <div class="pill-row" style="border-top:1px solid var(--border-color-soft); padding-top:12px; margin-top:14px;">
                        {''.join(f"<span class='pill'>{item['label']} &middot; {item['confidence']:.0%}</span>" for item in cause['top3'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif remark_text.strip():
                st.info("Root cause model not loaded — check nlp_outputs/models/tfidf_root_cause_classifier.pkl")

            st.markdown(
                f"<div class='disclaimer-bar'>&#9200; {result['latency_ms']}ms &nbsp;&middot;&nbsp; "
                f"Request {result['request_id']} &nbsp;&middot;&nbsp; CMS PUF 2023 &middot; Portfolio demo only</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("""
<div class="ready-card">
    <div class="ready-icon">&#9889;</div>
    <div class="ready-title">Ready to Analyze</div>
    <div class="ready-subtitle">
        Fill in claim details on the left and click<br>
        <strong>Analyze Denial Risk</strong> to get instant results.
    </div>
    <div class="feature-list">
        <div class="feature-item">
            <div class="feature-icon fi-blue">&#9711;</div>
            <div>
                <div class="feature-title">Denial Probability Score</div>
                <div class="feature-desc">Real-time ML-based denial risk from 0&ndash;100%</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon fi-amber">&#9650;</div>
            <div>
                <div class="feature-title">Risk Tier &nbsp;&middot;&nbsp; HIGH / MEDIUM / LOW</div>
                <div class="feature-desc">Categorized using optimized probability thresholds</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon fi-green">&#8801;</div>
            <div>
                <div class="feature-title">Top Feature Drivers</div>
                <div class="feature-desc">SHAP-based explainability for each prediction</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon fi-red">&#9670;</div>
            <div>
                <div class="feature-title">Root Cause + Remediation Fix</div>
                <div class="feature-desc">NLP classifier maps remark text to X12 RARC taxonomy</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)