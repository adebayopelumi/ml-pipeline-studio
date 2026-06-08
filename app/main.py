import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# Register a dark Plotly template — applies to ALL charts in the app automatically
pio.templates["ml_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="rgba(255,255,255,0.82)", family="system-ui, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            linecolor="rgba(255,255,255,0.12)",
            zerolinecolor="rgba(255,255,255,0.12)",
            title_font=dict(color="rgba(255,255,255,0.6)"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            linecolor="rgba(255,255,255,0.12)",
            zerolinecolor="rgba(255,255,255,0.12)",
            title_font=dict(color="rgba(255,255,255,0.6)"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="rgba(255,255,255,0.8)"),
        ),
        colorway=["#667eea", "#a78bfa", "#34d399", "#f59e0b", "#f87171", "#38bdf8"],
        title=dict(font=dict(color="#ffffff", size=15)),
    )
)
pio.templates.default = "plotly+ml_dark"

st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global theme ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
[data-testid="stSidebarNav"]  { display: none; }
[data-testid="stToolbar"]     { display: none; }
footer                        { display: none; }
header[data-testid="stHeader"]{ background: transparent !important; }

/* ── Page background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0f0c29 0%, #1e1b4b 55%, #1a0533 100%);
    min-height: 100vh;
}
[data-testid="stMain"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13103a 0%, #1a1040 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.85) !important;
    border-radius: 10px !important;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(102,126,234,0.25) !important;
    border-color: #667eea !important;
}
/* Permanently lock sidebar open — hide every possible collapse/expand control */
[data-testid="stSidebarCollapsedControl"]          { display: none !important; }
[data-testid="stSidebar"] [data-testid="baseButton-header"]          { display: none !important; }
[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] { display: none !important; }
[data-testid="stSidebar"] button[kind="header"]    { display: none !important; }
[data-testid="stSidebar"] > div > div > div > button:first-of-type  { display: none !important; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6,
[data-testid="stHeading"] { color: #ffffff !important; }
p, li, span, div { color: rgba(255,255,255,0.85); }
[data-testid="stCaption"], small, .stCaption { color: rgba(255,255,255,0.45) !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    backdrop-filter: blur(10px);
}
[data-testid="metric-container"] label { color: rgba(255,255,255,0.5) !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important; font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton button[kind="primary"],
button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 16px rgba(102,126,234,0.35) !important;
    transition: all 0.2s !important;
}
.stButton button[kind="primary"]:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton button[kind="secondary"],
.stButton button:not([kind]) {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: rgba(255,255,255,0.85) !important;
    border-radius: 10px !important;
}
.stButton button:not([kind]):hover {
    background: rgba(102,126,234,0.2) !important;
    border-color: #667eea !important;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important;
}
.stTextInput input::placeholder { color: rgba(255,255,255,0.3) !important; }

/* ── Selectbox / Multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg { fill: rgba(255,255,255,0.5) !important; }

/* ── Radio ── */
[data-testid="stRadio"] label { color: rgba(255,255,255,0.8) !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: rgba(255,255,255,0.8) !important; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] label span { color: rgba(255,255,255,0.8) !important; }

/* ── Slider ── */
[data-testid="stSlider"] { color: rgba(255,255,255,0.8) !important; }
[data-testid="stSlider"] .st-bq { background: #667eea !important; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, rgba(102,126,234,0.06), rgba(118,75,162,0.06)) !important;
    border: 2px dashed rgba(102,126,234,0.45) !important;
    border-radius: 16px !important;
    padding: 32px 24px !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #667eea !important;
    background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(118,75,162,0.12)) !important;
    box-shadow: 0 0 0 4px rgba(102,126,234,0.12), inset 0 0 30px rgba(102,126,234,0.05) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: rgba(255,255,255,0.75) !important; }
[data-testid="stFileUploaderDropzone"] svg { opacity: 0.7; }
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: rgba(255,255,255,0.4) !important; font-size: 0.82rem !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: rgba(102,126,234,0.12) !important;
    border: 1px solid rgba(102,126,234,0.3) !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    margin-top: 8px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * { color: rgba(255,255,255,0.85) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: rgba(255,255,255,0.85) !important; }
[data-testid="stExpander"] summary:hover { color: #ffffff !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border-bottom: none !important;
}
[data-testid="stTabs"] button[role="tab"] {
    color: rgba(255,255,255,0.55) !important;
    border-radius: 9px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    border: none !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ── Dataframe / Table ── */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    background: rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stAlert"] p { color: rgba(255,255,255,0.9) !important; }

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}

/* ── Code blocks ── */
[data-testid="stCode"], code {
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #a5f3fc !important;
}

/* ── Plotly chart background ── */
.js-plotly-plot, .plotly { background: transparent !important; }
.stPlotlyChart > div { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# Hide the sidebar completely until the user is logged in
if not st.session_state.get("authentication_status"):
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
else:
    # JS runs after every Streamlit re-render and removes the collapse button from the DOM
    st.markdown("""
    <script>
    (function removeSidebarToggle() {
        const sel = [
            '[data-testid="stSidebar"] [data-testid="baseButton-header"]',
            '[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"]',
            '[data-testid="stSidebarCollapsedControl"]'
        ];
        function purge() {
            sel.forEach(s => document.querySelectorAll(s).forEach(el => el.remove()));
        }
        purge();
        new MutationObserver(purge).observe(document.body, {childList: true, subtree: true});
    })();
    </script>
    """, unsafe_allow_html=True)


# ── Auth helpers ──────────────────────────────────────────────────────────────
import yaml
from pathlib import Path

AUTH_CONFIG_PATH = Path(__file__).parent.parent / "config" / "auth_config.yaml"


def _ensure_auth_config():
    if AUTH_CONFIG_PATH.exists():
        return
    AUTH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "credentials": {
            "usernames": {
                "admin": {
                    "email": "adebayopelumiayomiposi@gmail.com",
                    "name": "Admin",
                    "password": "admin123",
                }
            }
        },
        "cookie": {
            "expiry_days": 30,
            "key": "ml_pipeline_secret_key_xK9p2024_secure",
            "name": "ml_pipeline_auth",
        },
    }
    with open(AUTH_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def _load_auth_config():
    _ensure_auth_config()
    with open(AUTH_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _reset_user_session(username: str):
    import shutil
    preserve = {"authentication_status", "name", "username",
                "logout", "FormSubmitter:Login-Login", "_active_user"}
    for k in [k for k in st.session_state if k not in preserve]:
        del st.session_state[k]
    user_dir = Path("models") / username
    if user_dir.exists():
        shutil.rmtree(user_dir)


def _render_login_page():
    """Render a professional branded login page before the form."""
    st.markdown("""
    <style>
    /* Hide Streamlit chrome on login */
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { display: none; }
    footer { display: none; }

    /* Full-page gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }
    [data-testid="stMain"] {
        background: transparent;
    }

    /* Centre the login card */
    .login-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding-top: 6vh;
    }

    /* Logo mark */
    .logo-ring {
        width: 72px;
        height: 72px;
        border-radius: 18px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 20px auto;
        box-shadow: 0 8px 32px rgba(102,126,234,0.45);
    }
    .logo-ring svg {
        width: 38px;
        height: 38px;
    }

    /* App name */
    .app-name {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .app-tagline {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.55);
        text-align: center;
        margin-bottom: 36px;
        letter-spacing: 0.3px;
    }

    /* Login card */
    .login-card {
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 40px 44px 36px 44px;
        width: 100%;
        max-width: 420px;
        margin: 0 auto;
        box-shadow: 0 24px 64px rgba(0,0,0,0.4);
    }

    /* Style the Streamlit form inside the card */
    [data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stForm"] label {
        color: rgba(255,255,255,0.75) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stForm"] input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 10px 14px !important;
    }
    [data-testid="stForm"] input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.25) !important;
    }
    [data-testid="stForm"] input::placeholder {
        color: rgba(255,255,255,0.3) !important;
    }
    /* Login button */
    [data-testid="stForm"] button[kind="primaryFormSubmit"],
    [data-testid="stForm"] button[type="submit"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
        width: 100% !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 16px rgba(102,126,234,0.4) !important;
        margin-top: 8px !important;
    }
    [data-testid="stForm"] button:hover {
        opacity: 0.92 !important;
        transform: translateY(-1px) !important;
    }

    /* Hide the "Login" subheader that streamlit-authenticator adds */
    [data-testid="stForm"] h2,
    [data-testid="stForm"] h3 {
        display: none !important;
    }

    /* Error message */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        margin-top: 12px !important;
    }

    /* Footer note */
    .login-footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.78rem;
        margin-top: 28px;
    }
    </style>

    <div class="login-wrapper">
      <div class="logo-ring">
        <svg viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
          <!-- Bar chart icon -->
          <rect x="4"  y="22" width="6" height="12" rx="2" fill="white" opacity="0.9"/>
          <rect x="14" y="14" width="6" height="20" rx="2" fill="white"/>
          <rect x="24" y="8"  width="6" height="26" rx="2" fill="white" opacity="0.75"/>
          <!-- Trend line -->
          <polyline points="7,21 17,13 27,7" stroke="white" stroke-width="2"
                    stroke-linecap="round" stroke-linejoin="round" opacity="0.5"/>
        </svg>
      </div>
      <div class="app-name">ML Pipeline Studio</div>
      <div class="app-tagline">Train smarter. Predict faster.</div>
      <div class="login-card">
    """, unsafe_allow_html=True)


# ── Authentication ────────────────────────────────────────────────────────────
try:
    import streamlit_authenticator as stauth

    auth_cfg   = _load_auth_config()
    cookie_cfg = auth_cfg.get("cookie", {})

    authenticator = stauth.Authenticate(
        auth_cfg["credentials"],
        cookie_cfg.get("name", "ml_pipeline_auth"),
        cookie_cfg.get("key", "ml_pipeline_secret_key_xK9p2024_secure"),
        cookie_cfg.get("expiry_days", 30),
        auto_hash=True,
    )

    authentication_status = st.session_state.get("authentication_status")

    if not authentication_status:
        _render_login_page()

    authenticator.login(location="main")

    authentication_status = st.session_state.get("authentication_status")
    auth_name = st.session_state.get("name", "")

    if authentication_status is False:
        st.error("Incorrect username or password. Please try again.")
        st.markdown("</div></div>", unsafe_allow_html=True)
        st.stop()

    elif authentication_status is None:
        st.markdown("</div></div>", unsafe_allow_html=True)
        st.stop()

    # ── Clear session when user switches account ──────────────────────────────
    auth_username = st.session_state.get("username", "")
    if st.session_state.get("_active_user") != auth_username:
        preserve = {"authentication_status", "name", "username",
                    "logout", "FormSubmitter:Login-Login"}
        for k in [k for k in st.session_state
                  if k not in preserve and not k.startswith("_")]:
            del st.session_state[k]
        st.session_state["_active_user"] = auth_username

    is_admin = auth_username == "admin"

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("ML Pipeline Studio")
        st.caption(f"Signed in as **{auth_name}**")
        authenticator.logout(button_name="Sign out", location="sidebar")
        st.divider()
        nav_options = ["Train", "Predict", "Results"] + (["Admin"] if is_admin else [])
        page = st.radio("Navigate", nav_options, label_visibility="collapsed")
        if st.session_state.get("_current_page") != page:
            st.session_state["_current_page"] = page
            st.session_state["_page_scrolled"] = False
        st.divider()
        if st.button("Reset everything", use_container_width=True):
            _reset_user_session(auth_username)
            st.rerun()
        st.divider()
        st.caption("scikit-lean · XGBoost · LightGBM · MLflow · Optuna")

except ImportError:
    with st.sidebar:
        st.title("ML Pipeline Studio")
        st.caption("Train any model. On any dataset.")
        st.divider()
        page = st.radio("Navigate", ["Train", "Predict", "Results", "Admin"],
                        label_visibility="collapsed")
        st.divider()
        st.caption("scikit-learn · XGBoost · LightGBM · MLflow · Optuna")


# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Train":
    from app.pages.train import render
    render()
elif page == "Predict":
    from app.pages.predict import render
    render()
elif page == "Results":
    from app.pages.results import render
    render()
elif page == "Admin":
    from app.pages.admin import render
    render()
