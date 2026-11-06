"""Visual shell for ML Pipeline Studio.

Dark theme: black app canvas, #121212 panels, #1ed760 accent, pill buttons,
8px panel radius. Split-screen layout: library on the left, the working page
in the centre, a "current model" panel on the right, and a player bar along
the bottom that tracks pipeline progress.
"""
from __future__ import annotations

import html
from urllib.parse import quote

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Tokens ───────────────────────────────────────────────────────────────────
BLACK = "#000000"
BASE = "#121212"
ELEVATED = "#1f1f1f"
HIGHLIGHT = "#2a2a2a"
PRESS = "#3b3b3b"
TEXT = "#ffffff"
SUBDUED = "#b3b3b3"
MUTED = "#7c7c7c"
GREEN = "#1ed760"
GREEN_HI = "#3be477"
NEGATIVE = "#f3727f"
WARNING = "#ffa42b"
BLUE = "#539df5"

# Categorical series for charts — accent first, then the semantic hues.
SERIES = [GREEN, BLUE, WARNING, NEGATIVE, "#b49bc8", SUBDUED]

FONT_STACK = "Figtree, 'Helvetica Neue', Helvetica, Arial, sans-serif"

# ── Icons (24px stroke icons, currentColor) ──────────────────────────────────
_ICON_PATHS = {
    # Line versions of the page artwork below (chip, magnifier, report).
    "train": '<rect x="7" y="7" width="10" height="10" rx="2"/><path d="M9.5 3v4M12 3v4M14.5 3v4M9.5 17v4M12 17v4M14.5 17v4M3 9.5h4M3 12h4M3 14.5h4M17 9.5h4M17 12h4M17 14.5h4"/>',
    "predict": '<circle cx="10.5" cy="10.5" r="7"/><path d="M15.5 15.5L21 21M8 13v-1.5M10.5 13V9M13 13V7.5"/>',
    "results": '<path d="M12 21H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8l5 5v3M14 3v5h5M8 15v-2M11 15v-4"/><circle cx="17.5" cy="17.5" r="4"/><path d="M15.8 17.6l1.2 1.2 2.3-2.4"/>',
    "admin": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
    "library": '<path d="M4 3v18M9 3v18M14 3.6l6 16.8"/>',
    "data": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M20 12c0 1.66-3.58 3-8 3s-8-1.34-8-3"/><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/>',
    "panel": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M15 3v18"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "spark": '<path d="M3 17l5-6 4 4 8-10"/><path d="M15 5h5v5"/>',
}


def icon(name: str, size: int = 24, stroke: float = 2.0) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{_ICON_PATHS[name]}</svg>'
    )


def _icon_uri(name: str, color: str = "#ffffff") -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{_ICON_PATHS[name]}</svg>'
    ).replace("currentColor", color)
    return "url(\"data:image/svg+xml," + quote(svg) + "\")"


# ── Page artwork (full-colour, 64×64) ────────────────────────────────────────
# Train = ML chip, Predict = magnifier over a bar chart, Results = report with a
# check badge. Each sits on its own pastel tile (the "tint" in PAGES).
_ART = {
    "train": (
        '<g stroke="#0a56b4" stroke-width="4.6" stroke-linecap="round">'
        '<path d="M24 6v8M32 6v8M40 6v8M24 50v8M32 50v8M40 50v8M6 24h8M6 32h8M6 40h8M50 24h8M50 32h8M50 40h8"/></g>'
        '<rect x="13" y="13" width="38" height="38" rx="9" fill="#0a56b4"/>'
        '<rect x="18.5" y="18.5" width="27" height="27" rx="4.5" fill="#d5e7fb"/>'
        '<path d="M22.6 38V26.5L27.1 33.2 31.6 26.5V38M35.6 26.5V38H41.6" fill="none" stroke="#0a56b4" '
        'stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "predict": (
        '<path d="M44.5 44.5L55 55" stroke="#622bad" stroke-width="8" stroke-linecap="round"/>'
        '<circle cx="29" cy="29" r="20" fill="#f1e9fb" stroke="#622bad" stroke-width="6.2"/>'
        '<g fill="#622bad"><rect x="20.4" y="29.5" width="5.2" height="9" rx="1.4"/>'
        '<rect x="26.4" y="24.5" width="5.2" height="14" rx="1.4"/>'
        '<rect x="32.4" y="19" width="5.2" height="19.5" rx="1.4"/></g>'
    ),
    "results": (
        '<defs><mask id="mlps-art-res-cut"><rect width="64" height="64" fill="#fff"/>'
        '<circle cx="48" cy="47" r="15.5" fill="#000"/></mask></defs>'
        '<g mask="url(#mlps-art-res-cut)">'
        '<path d="M12 6.5h23.5l12 12V54a4 4 0 0 1-4 4H12a4 4 0 0 1-4-4V10.5a4 4 0 0 1 4-4z" fill="#fff" '
        'stroke="#0b8a1b" stroke-width="4.6" stroke-linejoin="round"/>'
        '<path d="M35.5 6.5V15a3.5 3.5 0 0 0 3.5 3.5h8.5" fill="none" stroke="#0b8a1b" stroke-width="4" '
        'stroke-linejoin="round"/>'
        '<g fill="#0b8a1b"><rect x="13" y="30" width="5.6" height="9" rx="1.4"/>'
        '<rect x="21.2" y="24.5" width="5.6" height="14.5" rx="1.4"/>'
        '<rect x="29.4" y="20.5" width="5.6" height="18.5" rx="1.4"/></g>'
        '<path d="M14.5 46.5h16" stroke="#5cc56a" stroke-width="3.6" stroke-linecap="round"/></g>'
        '<circle cx="48" cy="47" r="12.5" fill="#0b8a1b"/>'
        '<path d="M42.6 47.4l3.8 3.8 7.2-7.6" fill="none" stroke="#d6f1d1" stroke-width="3.6" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "admin": (
        '<g stroke="#3d3d3d" stroke-width="4.4" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="43" cy="21" r="7.5" fill="#e9e9e9"/>'
        '<path d="M40 36.5h4.5A11.5 11.5 0 0 1 56 48v4" fill="none"/>'
        '<circle cx="25" cy="23" r="9" fill="#f7f7f7"/>'
        '<path d="M8 54v-3a12 12 0 0 1 12-12h10a12 12 0 0 1 12 12v3z" fill="#f7f7f7"/></g>'
    ),
}


def art(name: str, size: int) -> str:
    return (f'<svg class="ps-art" width="{size}" height="{size}" viewBox="0 0 64 64" '
            f'aria-hidden="true">{_ART[name]}</svg>')


def _art_uri(name: str) -> str:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">{_ART[name]}</svg>'
    return "url(\"data:image/svg+xml," + quote(svg) + "\")"


CHECK_BADGE = (
    '<svg class="ps-check" width="24" height="24" viewBox="0 0 24 24" aria-label="Done">'
    f'<circle cx="12" cy="12" r="11" fill="{GREEN}"/>'
    '<path d="M7 12.5l3.2 3.2L17 9" fill="none" stroke="#000" stroke-width="2.4" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

LOGO = (
    '<svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">'
    f'<circle cx="16" cy="16" r="16" fill="{GREEN}"/>'
    '<rect x="8" y="16" width="3.6" height="8" rx="1.8" fill="#000"/>'
    '<rect x="14.2" y="11" width="3.6" height="13" rx="1.8" fill="#000"/>'
    '<rect x="20.4" y="7" width="3.6" height="17" rx="1.8" fill="#000"/></svg>'
)

# ── Pages ────────────────────────────────────────────────────────────────────
# `color` drives the hero gradient (the icon's ink, toned for white text);
# `tint` is the pastel tile behind the icon artwork.
PAGES = {
    "Train":   {"color": "#1f5fae", "tint": "#b1d6fc", "icon": "train",   "kind": "Pipeline", "sub": "Upload & fit"},
    "Predict": {"color": "#5e33a6", "tint": "#d6bbfb", "icon": "predict", "kind": "Pipeline", "sub": "Score new data"},
    "Results": {"color": "#17803a", "tint": "#c8f2bf", "icon": "results", "kind": "Pipeline", "sub": "Metrics & charts"},
    "Admin":   {"color": "#535353", "tint": "#dedede", "icon": "admin",   "kind": "Settings", "sub": "Users & access"},
}

# ── Algorithms → "album covers" ──────────────────────────────────────────────
_ALGO = {
    "random_forest_classifier":     ("Random Forest", "RF", "#27856a", "#1e3264"),
    "random_forest_regressor":      ("Random Forest", "RF", "#27856a", "#1e3264"),
    "logistic_regression":          ("Logistic Regression", "LR", "#1e3264", "#509bf5"),
    "linear_regression":            ("Linear Regression", "LIN", "#1e3264", "#509bf5"),
    "ridge_regression":             ("Ridge Regression", "RDG", "#1e3264", "#477d95"),
    "lasso_regression":             ("Lasso Regression", "LSO", "#477d95", "#1e3264"),
    "gradient_boosting_classifier": ("Gradient Boosting", "GB", "#bc5900", "#8c1932"),
    "gradient_boosting_regressor":  ("Gradient Boosting", "GB", "#bc5900", "#8c1932"),
    "xgboost_classifier":           ("XGBoost", "XGB", "#e8115b", "#bc5900"),
    "xgboost_regressor":            ("XGBoost", "XGB", "#e8115b", "#bc5900"),
    "lgbm_classifier":              ("LightGBM", "LGBM", "#148a08", "#27856a"),
    "lgbm_regressor":               ("LightGBM", "LGBM", "#148a08", "#27856a"),
    "svm_classifier":               ("Support Vector Machine", "SVM", "#8d67ab", "#af2896"),
    "svr":                          ("Support Vector Regressor", "SVR", "#8d67ab", "#af2896"),
}


def algo_info(key: str | None, problem_type: str | None = None) -> dict:
    """Display name, cover initials and cover colours for an algorithm key."""
    is_auto = key in (None, "", "auto")
    resolved = key
    if is_auto and problem_type:
        try:
            from src.core.algorithm_registry import AUTO_DEFAULTS
            resolved = AUTO_DEFAULTS.get(problem_type, key)
        except Exception:
            resolved = key
    name, initials, c1, c2 = _ALGO.get(
        resolved or "",
        ((resolved or "Model").replace("_", " ").title(), "ML", "#535353", "#1f1f1f"),
    )
    return {"name": name, "initials": initials, "c1": c1, "c2": c2, "auto": is_auto}


PROBLEM_LABEL = {
    "binary_classification": "Binary classification",
    "multiclass_classification": "Multiclass classification",
    "regression": "Regression",
}

_UNIT_METRICS = ("roc_auc", "f1", "accuracy", "r2", "precision", "recall")


def primary_metric(metrics: dict | None) -> tuple[str, float] | None:
    """Headline metric on a 0–1 scale, for the player-bar meter."""
    if not metrics:
        return None
    for m in _UNIT_METRICS:
        if m in metrics and metrics[m] is not None:
            return m, float(metrics[m])
    return None


def metric_label(name: str) -> str:
    return {"roc_auc": "ROC AUC", "r2": "R²", "f1": "F1", "rmse": "RMSE",
            "mae": "MAE", "mape": "MAPE"}.get(name.lower(), name.replace("_", " ").title())


def esc(v) -> str:
    return html.escape(str(v), quote=True)


# ── Plotly ───────────────────────────────────────────────────────────────────
def setup_plotly() -> None:
    axis = dict(
        gridcolor="rgba(255,255,255,0.07)",
        linecolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.14)",
        title_font=dict(color=SUBDUED, size=12),
        tickfont=dict(color=SUBDUED, size=11),
    )
    pio.templates["studio_dark"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, family=FONT_STACK, size=13),
            xaxis=axis,
            yaxis=axis,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SUBDUED)),
            colorway=SERIES,
            title=dict(font=dict(color=TEXT, size=16, family=FONT_STACK)),
            hoverlabel=dict(bgcolor=HIGHLIGHT, bordercolor=HIGHLIGHT,
                            font=dict(color=TEXT, family=FONT_STACK)),
            margin=dict(l=8, r=8, t=40, b=8),
        )
    )
    pio.templates.default = "plotly_dark+studio_dark"


# ── CSS ──────────────────────────────────────────────────────────────────────
_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800;900&display=swap');

:root {
  --ps-black:#000; --ps-base:#121212; --ps-elev:#1f1f1f; --ps-high:#2a2a2a; --ps-press:#3b3b3b;
  --ps-text:#fff; --ps-sub:#b3b3b3; --ps-muted:#7c7c7c;
  --ps-green:#1ed760; --ps-green-hi:#3be477; --ps-neg:#f3727f; --ps-warn:#ffa42b; --ps-blue:#539df5;
  --ps-top:64px; --ps-bottom:72px; --ps-gap:8px; --ps-radius:8px;
  --ps-font: Figtree, 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* ── Streamlit chrome ─────────────────────────────────────────── */
.stApp { background: var(--ps-black); font-family: var(--ps-font); }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebarNav"], footer, [data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }

/* ── Type ─────────────────────────────────────────────────────── */
h1, h2, h3, h4 { font-family: var(--ps-font) !important; color: var(--ps-text) !important; letter-spacing: -0.01em; }
h2 { font-size: 1.5rem !important; font-weight: 800 !important; }
h3 { font-size: 1.25rem !important; font-weight: 700 !important; }
[data-testid="stCaptionContainer"], .stCaption, small { color: var(--ps-sub) !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--ps-text); }
hr { border-color: rgba(255,255,255,0.1) !important; margin: 1.25rem 0 !important; }
a { color: var(--ps-text) !important; }

/* ── Scrollbars ───────────────────────────────────────────────── */
* { scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.28) transparent; }
::-webkit-scrollbar { width: 12px; height: 12px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.28); border: 3px solid transparent; background-clip: padding-box; border-radius: 0; }
::-webkit-scrollbar-track { background: transparent; }

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
  border-radius: 9999px !important; font-weight: 700 !important; font-family: var(--ps-font) !important;
  min-height: 40px; padding: 6px 24px !important; transition: transform .1s ease, background-color .1s ease, border-color .1s ease;
}
.stButton button p, .stDownloadButton button p, .stFormSubmitButton button p { font-weight: 700 !important; font-size: 0.9rem; }
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover { transform: scale(1.04); }
.stButton button:active, .stDownloadButton button:active { transform: scale(1); }
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--ps-green) !important; border: none !important; color: #000 !important; min-height: 48px; padding: 8px 32px !important;
}
button[data-testid="stBaseButton-primary"] p, button[data-testid="stBaseButton-primaryFormSubmit"] p { color: #000 !important; font-size: 1rem; }
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover { background: var(--ps-green-hi) !important; }
button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-secondaryFormSubmit"] {
  background: transparent !important; border: 1px solid var(--ps-muted) !important; color: var(--ps-text) !important;
}
button[data-testid="stBaseButton-secondary"]:hover, button[data-testid="stBaseButton-secondaryFormSubmit"]:hover { border-color: var(--ps-text) !important; }
button:disabled { opacity: .45 !important; transform: none !important; }

/* ── Inputs ───────────────────────────────────────────────────── */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  background: var(--ps-elev) !important; border-radius: 6px !important;
}
[data-baseweb="input"], [data-baseweb="select"] > div { border: 1px solid transparent !important; box-shadow: inset 0 0 0 1px #4d4d4d !important; }
[data-baseweb="input"]:hover, [data-baseweb="select"] > div:hover { box-shadow: inset 0 0 0 1px var(--ps-sub) !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within { box-shadow: inset 0 0 0 2px var(--ps-text) !important; }
[data-baseweb="input"] input, [data-baseweb="select"] input { color: var(--ps-text) !important; font-family: var(--ps-font) !important; }
[data-baseweb="input"] input::placeholder { color: var(--ps-muted) !important; }
[data-testid="stNumberInput"] button { background: transparent !important; border: none !important; }
[data-testid="stWidgetLabel"] p { color: var(--ps-text) !important; font-weight: 600 !important; font-size: 0.875rem !important; }
[data-baseweb="tag"] { background: var(--ps-high) !important; border-radius: 9999px !important; }
[data-baseweb="tag"] span { color: var(--ps-text) !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: #282828 !important; border-radius: 6px !important; }
[data-baseweb="popover"] li { font-family: var(--ps-font) !important; }

/* Radio groups → filter chips */
[data-testid="stRadio"] [role="radiogroup"] { gap: 8px !important; flex-wrap: wrap; }
[data-testid="stRadio"] label[data-baseweb="radio"] {
  background: var(--ps-high); border-radius: 9999px; padding: 6px 14px !important; margin: 0 !important;
  transition: background-color .1s ease; cursor: pointer;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:hover { background: var(--ps-press); }
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none !important; }
[data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child { padding-left: 0 !important; margin-left: 0 !important; }
[data-testid="stRadio"] label[data-baseweb="radio"] p { color: var(--ps-text) !important; font-size: 0.875rem !important; font-weight: 500 !important; }
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) { background: var(--ps-text); }
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p { color: #000 !important; }

/* Tabs → chips */
[data-testid="stTabs"] [role="tablist"] { gap: 8px; border-bottom: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabs"] button[role="tab"] {
  background: var(--ps-high) !important; border-radius: 9999px !important; padding: 6px 16px !important; height: auto !important; border: none !important;
}
[data-testid="stTabs"] button[role="tab"]:hover { background: var(--ps-press) !important; }
[data-testid="stTabs"] button[role="tab"] p { color: var(--ps-text) !important; font-weight: 500 !important; font-size: 0.875rem !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: var(--ps-text) !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: #000 !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
  background: var(--ps-elev) !important; border: 1px dashed #4d4d4d !important; border-radius: var(--ps-radius) !important; padding: 24px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--ps-text) !important; background: var(--ps-high) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--ps-text) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--ps-sub) !important; }
[data-testid="stFileUploaderFile"] { background: var(--ps-high) !important; border-radius: 6px !important; padding: 8px 12px !important; }

/* Containers */
[data-testid="stExpander"] details { background: var(--ps-elev) !important; border: none !important; border-radius: var(--ps-radius) !important; }
[data-testid="stExpander"] summary:hover { background: var(--ps-high) !important; border-radius: var(--ps-radius); }
[data-testid="stExpander"] summary p { font-weight: 600 !important; }
[data-testid="stMetric"] {
  background: var(--ps-elev); border-radius: var(--ps-radius); padding: 16px !important;
  transition: background-color .3s ease;
}
[data-testid="stMetric"]:hover { background: var(--ps-high); }
[data-testid="stMetricLabel"] p { color: var(--ps-sub) !important; font-size: 0.75rem !important; font-weight: 700 !important; letter-spacing: .08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--ps-text) !important; font-weight: 800 !important; font-size: 1.75rem !important; letter-spacing: -0.02em; }
[data-testid="stAlert"] > div { border-radius: var(--ps-radius) !important; border: none !important; }
[data-testid="stAlertContentSuccess"] p { color: var(--ps-text) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: var(--ps-radius); overflow: hidden; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; }
[data-testid="stCode"] pre, code { background: var(--ps-elev) !important; border-radius: 6px !important; }
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; }
[data-testid="stMarkdownContainer"] th { color: var(--ps-sub); font-weight: 600; font-size: .8rem; text-align: left; border: none !important; border-bottom: 1px solid rgba(255,255,255,.1) !important; }
[data-testid="stMarkdownContainer"] td { border: none !important; font-size: .9rem; }
[data-testid="stMarkdownContainer"] tr:hover td { background: rgba(255,255,255,.06); }

/* ── Shared bits ──────────────────────────────────────────────── */
.ps-check { flex: none; display: block; }
.cover {
  position: relative; overflow: hidden; flex: none; border-radius: 4px; color: #fff;
  background:
    radial-gradient(120% 90% at 85% 0%, rgba(255,255,255,.28), transparent 55%),
    repeating-linear-gradient(135deg, rgba(255,255,255,.05) 0 2px, transparent 2px 9px),
    linear-gradient(135deg, var(--c1), var(--c2));
  display: flex; align-items: flex-end; box-shadow: 0 4px 16px rgba(0,0,0,.35);
}
.cover b { font-weight: 900; letter-spacing: -0.05em; line-height: .9; padding: 9%; }
</style>
"""

_SHELL_CSS = """
<style>
/* ── Frame: top bar / panels / player bar ─────────────────────── */
[data-testid="stAppViewContainer"] { top: var(--ps-top) !important; bottom: var(--ps-bottom) !important; height: auto !important; background: var(--ps-black); overflow: clip !important; }
[data-testid="stAppViewContainer"] > div:has(> [data-testid="stMain"]) { height: 100% !important; min-height: 0 !important; }
[data-testid="stMain"] { background: var(--ps-black); padding: 0 var(--ps-gap) var(--ps-gap) 0; overflow: clip; height: 100%; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: none !important; height: 100%; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0 !important; height: 100%; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { position: absolute; width: 0; height: 0; overflow: hidden; }

/* Sidebar = "Your Library" panel */
section[data-testid="stSidebar"] {
  background: var(--ps-black) !important; border: none !important; height: 100% !important;
  padding: 0 var(--ps-gap) var(--ps-gap) var(--ps-gap);
}
@media (min-width: 901px) {
  section[data-testid="stSidebar"] { width: 320px !important; min-width: 320px !important; max-width: 320px !important; }
  /* The library is always open on desktop, even if it was collapsed on a phone earlier */
  section[data-testid="stSidebar"][aria-expanded="false"] { transform: none !important; margin-left: 0 !important; }
}
[data-testid="stSidebarContent"] { background: var(--ps-base) !important; border-radius: var(--ps-radius); height: 100%; overflow-y: auto; }
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none !important; }
[data-testid="stSidebarUserContent"] { padding: 12px 8px 16px 8px !important; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: 2px; }

/* Shell columns */
.st-key-ps_shell { height: 100%; }
.st-key-ps_shell [data-testid="stHorizontalBlock"] { gap: var(--ps-gap) !important; height: 100%; flex-wrap: nowrap; }
.st-key-ps_shell [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { height: 100%; min-width: 0; }
.st-key-ps_center, .st-key-ps_right {
  flex: none !important; height: calc(100vh - var(--ps-top) - var(--ps-bottom) - var(--ps-gap)) !important;
  overflow-y: auto; overflow-x: hidden; background: var(--ps-base); border-radius: var(--ps-radius);
  isolation: isolate; flex-wrap: nowrap !important;
}
.st-key-ps_center { padding: 0 24px 40px 24px; }
.st-key-ps_right { padding: 16px; gap: 16px; }
.st-key-ps_center [data-testid="stHorizontalBlock"] { height: auto; gap: 16px !important; flex-wrap: wrap; }
.st-key-ps_center [data-testid="stColumn"] { height: auto !important; }

/* ── Top bar ──────────────────────────────────────────────────── */
.ps-top {
  position: fixed; inset: 0 0 auto 0; height: var(--ps-top); z-index: 999990; background: var(--ps-black);
  display: grid; grid-template-columns: 1fr minmax(0, 560px) 1fr; align-items: center; gap: 16px; padding: 0 16px 0 24px;
  font-family: var(--ps-font);
}
.ps-top-left { display: flex; align-items: center; gap: 12px; min-width: 0; }
.ps-top-left span { font-weight: 800; font-size: 1rem; color: var(--ps-text); white-space: nowrap; letter-spacing: -0.01em; }
.ps-pill {
  height: 48px; border-radius: 9999px; background: var(--ps-elev); display: flex; align-items: center; gap: 10px;
  padding: 0 20px 0 14px; color: var(--ps-sub); font-size: .95rem; min-width: 0; box-shadow: inset 0 0 0 1px rgba(255,255,255,.04);
}
.ps-pill-icon { width: 34px; height: 34px; border-radius: 50%; background: var(--ps-high); display: grid; place-items: center; color: var(--ps-text); flex: none; }
.ps-pill-text { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ps-pill-text b { color: var(--ps-text); font-weight: 700; }
.ps-pill-text i { font-style: normal; margin: 0 8px; color: var(--ps-muted); }
.ps-top-right { display: flex; justify-content: flex-end; align-items: center; gap: 12px; }
.ps-top-right .who { color: var(--ps-sub); font-size: .875rem; white-space: nowrap; }
.ps-avatar {
  width: 32px; height: 32px; border-radius: 50%; background: #e8115b; color: #fff; font-weight: 800; font-size: .85rem;
  display: grid; place-items: center; box-shadow: 0 0 0 6px var(--ps-elev); margin-right: 6px; flex: none;
}

/* ── Player bar ───────────────────────────────────────────────── */
.ps-bar {
  position: fixed; inset: auto 0 0 0; height: var(--ps-bottom); z-index: 999990; background: var(--ps-black);
  display: grid; grid-template-columns: 1fr minmax(0, 620px) 1fr; align-items: center; gap: 16px; padding: 0 16px 0 16px;
  font-family: var(--ps-font);
}
.ps-bar-left { display: flex; align-items: center; gap: 12px; min-width: 0; }
.ps-bar-tile { width: 56px; height: 56px; border-radius: 4px; display: grid; place-items: center; color: #fff; flex: none; background: var(--ps-high); }
.ps-bar-t { color: var(--ps-text); font-size: .875rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ps-bar-s { color: var(--ps-sub); font-size: .75rem; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ps-bar-mid { display: flex; flex-direction: column; align-items: center; gap: 8px; min-width: 0; }
.ps-stages { display: flex; align-items: center; gap: 6px; font-size: .75rem; font-weight: 700; color: var(--ps-muted); }
.ps-stage { position: relative; padding: 2px 8px; border-radius: 9999px; letter-spacing: .02em; }
.ps-stage.done { color: var(--ps-green); }
.ps-stage.done::after { content: ""; position: absolute; left: 50%; bottom: -5px; width: 4px; height: 4px; margin-left: -2px; border-radius: 50%; background: var(--ps-green); }
.ps-stage.next { color: #000; background: var(--ps-text); }
.ps-stage-sep { color: #4d4d4d; }
.ps-scrub { display: flex; align-items: center; gap: 8px; width: 100%; font-size: .6875rem; color: var(--ps-sub); font-variant-numeric: tabular-nums; }
.ps-track { flex: 1; height: 4px; border-radius: 9999px; background: #4d4d4d; overflow: hidden; }
.ps-track i { display: block; height: 100%; border-radius: 9999px; background: var(--ps-text); }
.ps-bar:hover .ps-track i, .ps-meter:hover .ps-track i { background: var(--ps-green); }
.ps-bar-right { display: flex; justify-content: flex-end; align-items: center; gap: 10px; color: var(--ps-sub); font-size: .75rem; min-width: 0; white-space: nowrap; }
.ps-meter { display: flex; align-items: center; gap: 8px; width: 220px; }
.ps-meter b { color: var(--ps-text); font-weight: 700; font-variant-numeric: tabular-nums; }
.ps-meter .ps-track { height: 4px; }

/* ── Library (sidebar) ────────────────────────────────────────── */
.ps-lib-head { display: flex; align-items: center; justify-content: space-between; padding: 8px 8px 12px 8px; color: var(--ps-sub); }
.ps-lib-head span { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1rem; color: var(--ps-text); }
.ps-lib-section { display: flex; justify-content: space-between; align-items: center; padding: 20px 8px 6px 8px; color: var(--ps-sub); font-size: .8125rem; }
.ps-lib-section b { color: var(--ps-text); font-size: .875rem; font-weight: 700; }
[class*="st-key-nav_"] button {
  width: 100%; min-height: 64px !important; padding: 8px !important; border: none !important; border-radius: 6px !important;
  background: transparent !important; justify-content: flex-start !important; transform: none !important;
}
[class*="st-key-nav_"] button:hover { background: var(--ps-high) !important; transform: none !important; }
[class*="st-key-nav_"] button > div { display: flex; align-items: center; justify-content: flex-start !important; gap: 12px; width: 100%; }
[class*="st-key-nav_"] button > div::before {
  content: ""; width: 48px; height: 48px; flex: none; border-radius: 4px;
  background-repeat: no-repeat; background-position: center; background-size: 22px 22px;
}
[class*="st-key-nav_"] button [data-testid="stMarkdownContainer"] { text-align: left; min-width: 0; flex: 1 1 auto; }
[class*="st-key-nav_"] button p { font-size: 1rem !important; font-weight: 500 !important; color: var(--ps-text) !important; line-height: 1.3; }
[class*="st-key-nav_"] button p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
[class*="st-key-nav_"] button p::after { display: block; font-size: .875rem; font-weight: 400; color: var(--ps-sub); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ps-run { display: flex; align-items: center; gap: 12px; padding: 8px; border-radius: 6px; }
.ps-run:hover { background: var(--ps-high); }
.ps-run-t { color: var(--ps-text); font-size: 1rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ps-run-t.live { color: var(--ps-green); }
.ps-run-s { color: var(--ps-sub); font-size: .875rem; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ps-run-s svg { vertical-align: -2px; margin-right: 4px; color: var(--ps-green); }
.ps-lib-empty { margin: 4px 0; padding: 16px; background: var(--ps-elev); border-radius: var(--ps-radius); }
.ps-lib-empty b { display: block; color: var(--ps-text); font-size: 1rem; margin-bottom: 6px; }
.ps-lib-empty span { color: var(--ps-sub); font-size: .875rem; line-height: 1.4; }
.ps-lib-foot { color: var(--ps-muted); font-size: .6875rem; padding: 16px 8px 0 8px; line-height: 1.5; }
.st-key-ps_lib_actions { padding: 12px 8px 0 8px; }
.st-key-ps_lib_actions [data-testid="stVerticalBlock"] { gap: 8px !important; }
.st-key-ps_lib_actions button { min-height: 32px !important; padding: 4px 16px !important; }
.st-key-ps_lib_actions button p { font-size: .8125rem !important; }

/* ── Hero (playlist header) ───────────────────────────────────── */
.ps-hero {
  position: relative; margin: 0 -24px; padding: 72px 24px 24px 24px; min-height: 300px;
  display: flex; align-items: flex-end; gap: 24px;
  background: linear-gradient(180deg, transparent 0%, rgba(0,0,0,.5) 100%), var(--page);
}
.ps-hero::after {
  content: ""; position: absolute; left: 0; right: 0; top: 100%; height: 240px; z-index: -1; pointer-events: none;
  background: linear-gradient(180deg, color-mix(in srgb, var(--page) 40%, transparent) 0%, transparent 100%);
}
.ps-hero-art {
  width: 192px; height: 192px; flex: none; border-radius: 4px; display: grid; place-items: center;
  background: radial-gradient(120% 90% at 85% 0%, rgba(255,255,255,.45), transparent 60%), var(--tint);
  box-shadow: 0 4px 60px rgba(0,0,0,.5);
}
.ps-hero-art .ps-art { width: 67%; height: 67%; }
.ps-hero-text { min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.ps-kicker { font-size: .875rem; font-weight: 500; color: var(--ps-text); }
.ps-title { font-size: clamp(3rem, 6.5vw, 6rem); font-weight: 900; line-height: 1; letter-spacing: -0.04em; color: var(--ps-text); margin: 4px 0 8px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-bottom: .06em; }
.ps-desc { color: rgba(255,255,255,.72); font-size: .875rem; line-height: 1.5; }
.ps-meta { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; font-size: .875rem; color: rgba(255,255,255,.85); }
.ps-meta b { color: var(--ps-text); font-weight: 700; }
.ps-meta .dot { color: rgba(255,255,255,.6); margin: 0 2px; }
.ps-meta .ps-avatar { width: 24px; height: 24px; font-size: .7rem; box-shadow: none; margin-right: 2px; }

/* ── Tracklist (Train steps) ──────────────────────────────────── */
.ps-tl-head {
  display: grid; grid-template-columns: 40px minmax(0,1fr) minmax(0, 40%) 32px; gap: 16px; align-items: center;
  padding: 0 16px 8px 16px; margin-top: 16px; border-bottom: 1px solid rgba(255,255,255,.1);
  color: var(--ps-sub); font-size: .875rem;
}
.ps-tl-head .n { text-align: right; }
.ps-trk {
  display: grid; grid-template-columns: 40px minmax(0,1fr) minmax(0, 40%) 32px; gap: 16px; align-items: center;
  padding: 10px 16px; border-radius: 4px; margin-top: 12px; transition: background-color .1s ease;
}
.ps-trk:hover { background: rgba(255,255,255,.07); }
.ps-trk .n { text-align: right; color: var(--ps-sub); font-size: 1rem; font-variant-numeric: tabular-nums; }
.ps-trk.done .n { color: var(--ps-green); }
.ps-trk .t { color: var(--ps-text); font-size: 1rem; font-weight: 600; }
.ps-trk.done .t { color: var(--ps-green); }
.ps-trk .s { color: var(--ps-sub); font-size: .875rem; margin-top: 2px; }
.ps-trk .v { color: var(--ps-sub); font-size: .875rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ps-trk .v b { color: var(--ps-text); font-weight: 500; }
.ps-trk .c { display: flex; justify-content: flex-end; }
.ps-trk .c .ring { width: 20px; height: 20px; border-radius: 50%; border: 2px solid #4d4d4d; }

/* Section titles */
.st-key-ps_center h3 { font-size: 1.5rem !important; font-weight: 800 !important; }
.ps-chip { display: inline-flex; align-items: center; gap: 8px; background: var(--ps-high); color: var(--ps-sub); border-radius: 9999px; padding: 6px 14px; font-size: .8125rem; }
.ps-chip b { color: var(--ps-text); font-weight: 700; }
.ps-empty { background: var(--ps-elev); border-radius: var(--ps-radius); padding: 24px; margin-top: 24px; }
.ps-empty b { display: block; color: var(--ps-text); font-size: 1.25rem; font-weight: 800; margin-bottom: 6px; }
.ps-empty span { color: var(--ps-sub); font-size: .9375rem; }
.ps-section { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-top: 24px; }
.ps-section h4 { font-size: 1.5rem !important; font-weight: 800 !important; margin: 0 !important; padding: 0 !important; letter-spacing: -0.01em; }
.ps-section span { color: var(--ps-sub); font-size: .875rem; font-weight: 700; }
.ps-sub { color: var(--ps-sub); font-size: .875rem; margin-top: -4px; }

/* ── Now playing (right panel) ────────────────────────────────── */
.ps-np-head { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1rem; color: var(--ps-text); padding: 4px 0 4px 0; }
.ps-np-head svg { color: var(--ps-sub); }
.ps-np-cover { width: 100%; aspect-ratio: 1 / 1; border-radius: var(--ps-radius); container-type: inline-size; margin-top: 12px; }
.ps-np-cover b { font-size: 34cqw; }
.ps-np-cover .tag {
  position: absolute; top: 12px; left: 12px; font-size: .6875rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase;
  background: rgba(0,0,0,.35); padding: 4px 8px; border-radius: 9999px; color: #fff;
}
.ps-np-cover.empty { background: var(--ps-elev); box-shadow: none; display: grid; place-items: center; color: var(--ps-muted); }
.ps-np-title-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 16px; }
.ps-np-title { font-size: 1.5rem; font-weight: 800; color: var(--ps-text); letter-spacing: -0.02em; line-height: 1.2; }
.ps-np-sub { color: var(--ps-sub); font-size: 1rem; margin-top: 2px; }
.ps-card { border-radius: var(--ps-radius); padding: 16px; background: var(--ps-elev); margin-top: 16px; overflow: hidden; }
.ps-card-title { color: var(--ps-text); font-weight: 700; font-size: 1rem; margin-bottom: 12px; }
.ps-score { background: color-mix(in srgb, var(--c1) 62%, #000); }
.ps-lyric { color: rgba(255,255,255,.72); font-weight: 800; font-size: 1.375rem; line-height: 1.35; letter-spacing: -0.01em; }
.ps-lyric span { color: var(--ps-text); font-variant-numeric: tabular-nums; }
.ps-lyric.fade { opacity: .45; }
.ps-kv { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; font-size: .875rem; border-bottom: 1px solid rgba(255,255,255,.06); }
.ps-kv:last-child { border-bottom: none; }
.ps-kv span { color: var(--ps-sub); }
.ps-kv b { color: var(--ps-text); font-weight: 600; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ps-hist { display: flex; align-items: center; gap: 12px; padding: 6px 0; }
.ps-hist .t { color: var(--ps-text); font-size: .875rem; font-weight: 600; }
.ps-hist .s { color: var(--ps-sub); font-size: .75rem; margin-top: 1px; }
.ps-hist .m { margin-left: auto; color: var(--ps-text); font-size: .875rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.ps-steps li { color: var(--ps-sub); font-size: .875rem; margin: 6px 0; }
.ps-steps li b { color: var(--ps-text); }
.st-key-np_open button { background: var(--ps-text) !important; border: none !important; min-height: 32px !important; padding: 4px 16px !important; }
.st-key-np_open button p { color: #000 !important; font-size: .8125rem !important; }
.st-key-np_open { margin-top: -4px; }

/* ── Medium screens: "now playing" drops below the page, page scrolls ── */
@media (max-width: 1180px) {
  [data-testid="stMain"] { overflow: auto; }
  .st-key-ps_shell [data-testid="stHorizontalBlock"] { flex-wrap: wrap; height: auto; }
  .st-key-ps_shell [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { height: auto; flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }
  .st-key-ps_center, .st-key-ps_right { height: auto !important; overflow: visible; }
  .st-key-ps_shell { height: auto; }
  .ps-np-cover { max-width: 280px; }
  .ps-bar-right .ps-meter { width: 120px; }
  .ps-bar { grid-template-columns: minmax(0, 1fr) minmax(0, 440px) auto; }
}

/* ── Phones ───────────────────────────────────────────────────── */
@media (max-width: 900px) {
  .ps-top { grid-template-columns: auto minmax(0, 1fr) auto; gap: 10px; padding: 0 12px; }
  .ps-pill { height: 40px; padding: 0 14px 0 6px; }
  .ps-pill-icon { width: 28px; height: 28px; }
  .ps-top-right .who, .ps-top-left span { display: none; }
  .ps-bar { grid-template-columns: 1fr; }
  .ps-bar-mid, .ps-bar-right { display: none; }
  [data-testid="stMain"] { padding: 0 var(--ps-gap) var(--ps-gap) var(--ps-gap); }
  .ps-hero { flex-direction: column; align-items: flex-start; padding-top: 32px; }
  .ps-hero-art { width: 128px; height: 128px; }
  .ps-tl-head, .ps-trk { grid-template-columns: 24px minmax(0,1fr) 28px; }
  .ps-tl-head .v, .ps-trk .v { display: none; }
  /* Phones: bring back Streamlit's own open/close controls for the library drawer */
  header[data-testid="stHeader"] { display: block !important; background: transparent !important; height: 48px; }
  header[data-testid="stHeader"] [data-testid="stToolbar"] { display: flex !important; }
  [data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"] { display: none !important; }
  [data-testid="stSidebarHeader"] { display: flex !important; }
  [data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] { display: flex !important; }
  [data-testid="stMain"] { padding-top: 48px; }
}
</style>
"""

_LOGIN_CSS = """
<style>
.stApp { background: linear-gradient(180deg, #2a2a2a 0%, #000 100%) fixed; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; }
[data-testid="stMainBlockContainer"] { max-width: 520px !important; padding: 48px 16px 64px 16px !important; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
  background: var(--ps-base); border-radius: var(--ps-radius); padding: 40px 48px 32px 48px; gap: 12px;
}
.ps-login-head { text-align: center; }
.ps-login-head svg { width: 40px; height: 40px; }
.ps-login-head h1 { font-size: 1.75rem !important; font-weight: 800 !important; margin: 16px 0 4px 0 !important; padding: 0 !important; letter-spacing: -0.02em; }
.ps-login-head p { color: var(--ps-sub) !important; font-size: .9375rem; margin: 0 0 8px 0; }
[data-testid="stForm"] h2, [data-testid="stForm"] h3 { display: none !important; }
[data-testid="stForm"] [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]),
[data-testid="stForm"] :has(> [data-testid="stFormSubmitButton"]),
[data-testid="stForm"] [data-testid="stFormSubmitButton"], [data-testid="stForm"] .stFormSubmitButton { width: 100% !important; }
[data-testid="stForm"] [data-testid="stFormSubmitButton"] > div, [data-testid="stForm"] .stFormSubmitButton > div { width: 100% !important; }
[data-testid="stForm"] button[data-testid^="stBaseButton"] {
  width: 100% !important; margin-top: 12px; min-height: 48px; background: var(--ps-green) !important; border: none !important;
}
[data-testid="stForm"] button[data-testid^="stBaseButton"] p { color: #000 !important; font-size: 1rem !important; }
[data-testid="stForm"] button[data-testid^="stBaseButton"]:hover { background: var(--ps-green-hi) !important; }
[data-testid="stForm"] [data-baseweb="input"] { min-height: 48px; }
.ps-login-foot { text-align: center; color: var(--ps-sub); font-size: .8125rem; margin-top: 8px; padding-top: 24px; border-top: 1px solid rgba(255,255,255,.1); }
.ps-login-foot b { color: var(--ps-text); }
@media (max-width: 560px) { [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { padding: 32px 20px; } }
</style>
"""


def inject_base_css() -> None:
    st.markdown(_BASE_CSS, unsafe_allow_html=True)


def inject_login_css() -> None:
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)


def inject_shell_css(page: str, visible_pages: list[str]) -> None:
    """Layout CSS plus the per-page bits (active nav item, nav tile art)."""
    rules = []
    for name in visible_pages:
        meta = PAGES[name]
        rules.append(
            f'.st-key-nav_{name} button > div::before {{ background: {_art_uri(meta["icon"])} '
            f'center / 34px 34px no-repeat, {meta["tint"]}; }}'
            f'.st-key-nav_{name} button p::after {{ content: "{meta["kind"]} • {meta["sub"]}"; }}'
        )
    rules.append(
        f".st-key-nav_{page} button {{ background: {HIGHLIGHT} !important; }}"
        f".st-key-nav_{page} button p {{ color: {GREEN} !important; }}"
    )
    st.markdown(_SHELL_CSS + "<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)


# ── Building blocks ──────────────────────────────────────────────────────────
def cover(algo: dict, size: int) -> str:
    fs = max(10, int(size * (0.30 if len(algo["initials"]) <= 3 else 0.24)))
    return (
        f'<div class="cover" style="--c1:{algo["c1"]};--c2:{algo["c2"]};width:{size}px;height:{size}px">'
        f'<b style="font-size:{fs}px;padding:{max(4, size // 9)}px">{esc(algo["initials"])}</b></div>'
    )


def avatar(name: str) -> str:
    initial = (name or "?").strip()[:1].upper() or "?"
    return f'<span class="ps-avatar">{esc(initial)}</span>'


def hero_html(page: str, title: str, description: str = "", meta: list[str] | None = None,
              user: str | None = None) -> str:
    """Playlist-style page header. `meta` items are already-escaped HTML fragments."""
    p = PAGES[page]
    parts = []
    if user:
        parts.append(f"{avatar(user)}<b>{esc(user)}</b>")
    parts.extend(m for m in (meta or []) if m)
    meta_html = '<span class="dot">•</span>'.join(parts)
    return (
        f'<div class="ps-hero" style="--page:{p["color"]};--tint:{p["tint"]}">'
        f'<div class="ps-hero-art">{art(p["icon"], 128)}</div>'
        f'<div class="ps-hero-text"><div class="ps-kicker">{esc(p["kind"])}</div>'
        f'<div class="ps-title">{esc(title)}</div>'
        + (f'<div class="ps-desc">{esc(description)}</div>' if description else "")
        + (f'<div class="ps-meta">{meta_html}</div>' if meta_html else "")
        + "</div></div>"
    )


def hero(page: str, title: str, description: str = "", meta: list[str] | None = None,
         user: str | None = None, slot=None) -> None:
    (slot or st).markdown(hero_html(page, title, description, meta, user), unsafe_allow_html=True)


def tracklist_header(left: str = "Step", right: str = "Selection") -> None:
    st.markdown(
        f'<div class="ps-tl-head"><span class="n">#</span><span>{esc(left)}</span>'
        f'<span class="v">{esc(right)}</span><span class="c">{icon("clock", 16)}</span></div>',
        unsafe_allow_html=True,
    )


def track_html(num: int, title: str, subtitle: str = "", value: str | None = None,
               done: bool = False) -> str:
    """One numbered row in the Train tracklist. `value` may contain <b> markup (pre-escaped)."""
    state = CHECK_BADGE if done else '<span class="ring"></span>'
    return (
        f'<div class="ps-trk{" done" if done else ""}"><span class="n">{num}</span>'
        f'<div><div class="t">{esc(title)}</div>'
        + (f'<div class="s">{esc(subtitle)}</div>' if subtitle else "")
        + f'</div><div class="v">{value or "—"}</div><div class="c">{state}</div></div>'
    )


def track(num: int, title: str, subtitle: str = "", value: str | None = None,
          done: bool = False, slot=None) -> None:
    (slot or st).markdown(track_html(num, title, subtitle, value, done), unsafe_allow_html=True)


def chip(text: str, label: str = "") -> None:
    st.markdown(
        '<div class="ps-chip">' + (f"<b>{esc(label)}</b>" if label else "") + f"{esc(text)}</div>",
        unsafe_allow_html=True,
    )


def section(title: str, aside: str = "") -> None:
    st.markdown(
        f'<div class="ps-section"><h4>{esc(title)}</h4>'
        + (f"<span>{esc(aside)}</span>" if aside else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# ── Chrome ───────────────────────────────────────────────────────────────────
def top_bar(page: str, crumbs: list[str], user: str) -> None:
    p = PAGES[page]
    crumb_html = '<i>›</i>'.join(
        [f"<b>{esc(page)}</b>"] + [esc(c) for c in crumbs if c]
    )
    st.markdown(
        '<div class="ps-top">'
        f'<div class="ps-top-left">{LOGO}<span>ML Pipeline Studio</span></div>'
        f'<div class="ps-pill"><span class="ps-pill-icon" style="background:{p["tint"]}">{art(p["icon"], 22)}</span>'
        f'<span class="ps-pill-text">{crumb_html}</span></div>'
        f'<div class="ps-top-right"><span class="who">{esc(user)}</span>{avatar(user)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


STAGES = ["Data", "Target", "Train", "Evaluate", "Predict"]


def player_bar(dataset: dict | None, done: list[bool], headline: tuple[str, float] | None,
               last_target: str | None = None) -> None:
    if dataset:
        left = (
            f'<div class="ps-bar-tile" style="background:linear-gradient(135deg,#27856a,#1e3264)">{icon("data", 24)}</div>'
            f'<div style="min-width:0"><div class="ps-bar-t">{esc(dataset["name"])}</div>'
            f'<div class="ps-bar-s">{esc(dataset["detail"])}</div></div>'
        )
    else:
        hint = f"Last model trained on {esc(last_target)}" if last_target else "Upload a CSV on Train"
        left = (
            f'<div class="ps-bar-tile">{icon("data", 24)}</div>'
            '<div style="min-width:0"><div class="ps-bar-t">No dataset loaded</div>'
            f'<div class="ps-bar-s">{hint}</div></div>'
        )

    n_done = sum(done)
    nxt = next((i for i, d in enumerate(done) if not d), None)
    chips = []
    for i, (label, d) in enumerate(zip(STAGES, done)):
        cls = "done" if d else ("next" if i == nxt else "")
        chips.append(f'<span class="ps-stage {cls}">{label}</span>')
    stages = '<span class="ps-stage-sep">·</span>'.join(chips)
    pct = int(n_done / len(STAGES) * 100)
    status = "Pipeline complete" if nxt is None else f"Next: {STAGES[nxt]}"

    if headline:
        name, val = headline
        width = max(0.0, min(1.0, val)) * 100
        right = (
            f'<span>{esc(metric_label(name))}</span>'
            f'<div class="ps-meter"><div class="ps-track"><i style="width:{width:.1f}%"></i></div>'
            f'<b>{val:.3f}</b></div>'
        )
    else:
        right = '<span>No score yet</span><div class="ps-meter"><div class="ps-track"><i style="width:0"></i></div></div>'

    st.markdown(
        f'<div class="ps-bar"><div class="ps-bar-left">{left}</div>'
        f'<div class="ps-bar-mid"><div class="ps-stages">{stages}</div>'
        f'<div class="ps-scrub"><span>{n_done}/{len(STAGES)}</span>'
        f'<div class="ps-track"><i style="width:{pct}%"></i></div><span>{status}</span></div></div>'
        f'<div class="ps-bar-right">{right}</div></div>',
        unsafe_allow_html=True,
    )


def library_header() -> None:
    st.markdown(
        f'<div class="ps-lib-head"><span>{icon("library", 24)} Your Library</span></div>',
        unsafe_allow_html=True,
    )


def library_runs(history: list[dict], live_index: int | None) -> str:
    if not history:
        return (
            '<div class="ps-lib-section"><b>Recent runs</b></div>'
            '<div class="ps-lib-empty"><b>Train your first model</b>'
            "<span>Every run you train lands here, newest first.</span></div>"
        )
    rows = ['<div class="ps-lib-section"><b>Recent runs</b><span>Newest</span></div>']
    total = len(history)
    for idx in range(total - 1, max(-1, total - 7), -1):
        run = history[idx]
        algo = algo_info(run.get("algorithm"), run.get("problem_type"))
        head = primary_metric({k: v for k, v in run.items() if isinstance(v, (int, float))})
        score = f" • {metric_label(head[0])} {head[1]:.3f}" if head else ""
        live = idx == live_index
        rows.append(
            f'<div class="ps-run">{cover(algo, 48)}<div style="min-width:0">'
            f'<div class="ps-run-t{" live" if live else ""}">{esc(algo["name"])}</div>'
            f'<div class="ps-run-s">{icon("spark", 14) if live else ""}Run {idx + 1} • '
            f'{esc(run.get("target_column", ""))}{esc(score)}</div></div></div>'
        )
    return "".join(rows)


def now_playing_top(model: dict | None) -> str:
    head = f'<div class="ps-np-head">{icon("panel", 20)}<span>Current model</span></div>'
    if not model:
        return (
            head
            + f'<div class="ps-np-cover cover empty">{icon("results", 56, 1.4)}</div>'
            '<div class="ps-np-title-row"><div><div class="ps-np-title">Nothing trained yet</div>'
            '<div class="ps-np-sub">Your model shows up here.</div></div></div>'
        )
    algo = model["algo"]
    tag = '<span class="tag">Auto pick</span>' if algo["auto"] else ""
    return (
        head
        + f'<div class="ps-np-cover cover" style="--c1:{algo["c1"]};--c2:{algo["c2"]}">{tag}<b>{esc(algo["initials"])}</b></div>'
        f'<div class="ps-np-title-row"><div style="min-width:0"><div class="ps-np-title">{esc(algo["name"])}</div>'
        f'<div class="ps-np-sub">{esc(model["target"])} • {esc(PROBLEM_LABEL.get(model["problem_type"], model["problem_type"]))}</div></div>'
        f"{CHECK_BADGE}</div>"
        + _score_card(model)
    )


def _score_card(model: dict) -> str:
    metrics = model["metrics"] or {}
    items = list(metrics.items())[:4]
    lines = []
    for i, (k, v) in enumerate(items):
        fade = " fade" if i == len(items) - 1 and len(items) > 2 else ""
        lines.append(f'<div class="ps-lyric{fade}">{esc(metric_label(k))} <span>{v:.3f}</span></div>')
    cmp = model.get("compare")
    if cmp:
        c_algo = algo_info(cmp["algorithm"], model["problem_type"])
        lines.append(f'<div class="ps-lyric fade" style="font-size:1rem;margin-top:8px">vs {esc(c_algo["name"])}</div>')
    return (
        f'<div class="ps-card ps-score" style="--c1:{model["algo"]["c1"]}">'
        f'<div class="ps-card-title">Score preview</div>{"".join(lines)}</div>'
    )


def now_playing_rest(model: dict | None, dataset: dict | None, history: list[dict]) -> str:
    out = []
    if dataset or model:
        kv = []
        if dataset:
            kv += [("File", dataset["name"]), ("Rows", f'{dataset["rows"]:,}'), ("Columns", f'{dataset["cols"]:,}')]
        if model:
            kv += [("Target", model["target"]), ("Problem", PROBLEM_LABEL.get(model["problem_type"], model["problem_type"]))]
        out.append(
            '<div class="ps-card"><div class="ps-card-title">About the data</div>'
            + "".join(f'<div class="ps-kv"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in kv)
            + "</div>"
        )
    if history:
        rows = []
        for idx in range(len(history) - 1, max(-1, len(history) - 5), -1):
            run = history[idx]
            algo = algo_info(run.get("algorithm"), run.get("problem_type"))
            head = primary_metric({k: v for k, v in run.items() if isinstance(v, (int, float))})
            when = str(run.get("timestamp", ""))[11:16]
            rows.append(
                f'<div class="ps-hist">{cover(algo, 40)}<div style="min-width:0"><div class="t">{esc(algo["name"])}</div>'
                f'<div class="s">Run {idx + 1}{" • " + esc(when) + " UTC" if when else ""}</div></div>'
                + (f'<div class="m">{head[1]:.3f}</div>' if head else "")
                + "</div>"
            )
        out.append('<div class="ps-card"><div class="ps-card-title">Run history</div>' + "".join(rows) + "</div>")
    if not model:
        out.append(
            '<div class="ps-card"><div class="ps-card-title">How it works</div><ol class="ps-steps" style="padding-left:18px;margin:0">'
            "<li><b>Upload</b> a CSV on Train</li><li><b>Pick</b> the column to predict</li>"
            "<li><b>Train</b> — auto or hand-tuned</li><li><b>Explore</b> results, then predict</li></ol></div>"
        )
    return "".join(out)


def login_header() -> None:
    st.markdown(
        f'<div class="ps-login-head">{LOGO}<h1>Log in to ML Pipeline Studio</h1>'
        "<p>Train smarter. Predict faster.</p></div>",
        unsafe_allow_html=True,
    )


def login_footer() -> None:
    st.markdown(
        '<div class="ps-login-foot">Need an account? Ask your <b>admin</b> to add you.</div>',
        unsafe_allow_html=True,
    )


def scroll_center_to_top(token: str) -> None:
    import streamlit.components.v1 as components
    components.html(
        "<script>const d = window.parent.document;"
        "for (const s of ['.st-key-ps_center', '[data-testid=\"stMain\"]']) {"
        "  const el = d.querySelector(s); if (el) el.scrollTo({top: 0});"
        f"}}</script><!-- {esc(token)} -->",
        height=0,
    )
