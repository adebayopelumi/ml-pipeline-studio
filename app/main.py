import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path

import streamlit as st
import yaml

st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from app import theme  # noqa: E402

theme.setup_plotly()
theme.inject_base_css()


# ── Auth helpers ──────────────────────────────────────────────────────────────
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
    # On Streamlit Cloud, load credentials from st.secrets instead of the YAML file
    if "credentials" in st.secrets:
        return {
            "credentials": {
                "usernames": {
                    uname: dict(udata)
                    for uname, udata in st.secrets["credentials"]["usernames"].items()
                }
            },
            "cookie": dict(st.secrets["cookie"]),
        }
    # Local: fall back to YAML file
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


def _show_login(authenticator):
    theme.inject_login_css()
    theme.login_header()
    authenticator.login(location="main")
    theme.login_footer()


# ── Authentication ────────────────────────────────────────────────────────────
try:
    import streamlit_authenticator as stauth
except ImportError:
    stauth = None

authenticator = None
if stauth is not None:
    auth_cfg   = _load_auth_config()
    cookie_cfg = auth_cfg.get("cookie", {})

    authenticator = stauth.Authenticate(
        auth_cfg["credentials"],
        cookie_cfg.get("name", "ml_pipeline_auth"),
        cookie_cfg.get("key", "ml_pipeline_secret_key_xK9p2024_secure"),
        cookie_cfg.get("expiry_days", 30),
        auto_hash=True,
    )

    # The header slot sits above the form; it's only filled (and the login CSS
    # only injected) once we know the cookie didn't already sign the user in.
    login_head = st.empty()
    authenticator.login(location="main")
    authentication_status = st.session_state.get("authentication_status")

    if not authentication_status:
        theme.inject_login_css()
        with login_head.container():
            theme.login_header()

    if authentication_status is False:
        st.error("Incorrect username or password. Please try again.")
        theme.login_footer()
        st.stop()
    elif authentication_status is None:
        theme.login_footer()
        st.stop()

    auth_username = st.session_state.get("username", "")
    auth_name     = st.session_state.get("name", "") or auth_username

    # ── Clear session when user switches account ──────────────────────────────
    if st.session_state.get("_active_user") != auth_username:
        preserve = {"authentication_status", "name", "username",
                    "logout", "FormSubmitter:Login-Login"}
        for k in [k for k in st.session_state
                  if k not in preserve and not k.startswith("_")]:
            del st.session_state[k]
        st.session_state["_active_user"] = auth_username
else:
    auth_username, auth_name = "default", "Guest"

is_admin = auth_username == "admin"


# ── Navigation state ──────────────────────────────────────────────────────────
VISIBLE_PAGES = ["Train", "Predict", "Results"] + (["Admin"] if is_admin else [])

page = st.session_state.get("page", "Train")
if page not in VISIBLE_PAGES:
    page = "Train"
    st.session_state["page"] = page


def _go(target: str):
    st.session_state["page"] = target
    st.session_state["_scroll_reset"] = True


def _dataset_info():
    df = st.session_state.get("train_df_raw")
    if df is None:
        return None
    name = st.session_state.get("train_dataset_name") or \
        Path(st.session_state.get("train_dataset_path") or "dataset.csv").name
    rows, cols = len(df), len(df.columns)
    return {"name": name, "rows": rows, "cols": cols, "detail": f"{rows:,} rows · {cols} columns"}


def _model_info():
    metrics = st.session_state.get("last_metrics")
    if not metrics:
        return None
    problem_type = st.session_state.get("last_problem_type", "")
    compare = None
    if st.session_state.get("compare_metrics"):
        compare = {"algorithm": st.session_state.get("compare_algorithm"),
                   "metrics": st.session_state["compare_metrics"]}
    return {
        "algo": theme.algo_info(st.session_state.get("last_algorithm"), problem_type),
        "target": st.session_state.get("last_target", ""),
        "problem_type": problem_type,
        "metrics": metrics,
        "compare": compare,
    }


# Pull the last saved run back in so the side panels have something to show
# after a server restart (Results does the same on its own).
from app.pages.results import _restore_from_disk  # noqa: E402
_restore_from_disk()

theme.inject_shell_css(page, VISIBLE_PAGES)


# ── Library (left panel) ──────────────────────────────────────────────────────
with st.sidebar:
    theme.library_header()
    for name in VISIBLE_PAGES:
        st.button(name, key=f"nav_{name}", on_click=_go, args=(name,), use_container_width=True)
    # Filled now with what we know, then refreshed at the end of the run so a
    # model trained this run appears without the list blanking out meanwhile.
    runs_slot = st.empty()
    _history_now = st.session_state.get("training_history", [])
    runs_slot.markdown(
        theme.library_runs(_history_now, len(_history_now) - 1
                           if st.session_state.get("last_metrics") and _history_now else None),
        unsafe_allow_html=True,
    )
    with st.container(key="ps_lib_actions"):
        if st.button("Reset everything", key="ps_reset", use_container_width=True):
            _reset_user_session(auth_username)
            st.rerun()
        if authenticator is not None:
            authenticator.logout(button_name="Sign out", location="main",
                                 key="ps_logout", use_container_width=True)
    st.markdown(
        '<div class="ps-lib-foot">scikit-learn · XGBoost · LightGBM · MLflow · Optuna</div>',
        unsafe_allow_html=True,
    )

# Signed out just now → show the login form straight away.
if authenticator is not None and not st.session_state.get("authentication_status"):
    _show_login(authenticator)
    st.stop()


# ── Centre + right panels ─────────────────────────────────────────────────────
def _render_page(name: str):
    if name == "Train":
        from app.pages.train import render
    elif name == "Predict":
        from app.pages.predict import render
    elif name == "Results":
        from app.pages.results import render
    else:
        from app.pages.admin import render
    render()


with st.container(key="ps_shell"):
    center, right = st.columns([2.6, 1], gap="small")
    with center:
        with st.container(key="ps_center"):
            _render_page(page)

    # Read state after the page ran, so a model trained this run shows up immediately.
    model   = _model_info()
    dataset = _dataset_info()
    history = st.session_state.get("training_history", [])

    with right:
        with st.container(key="ps_right"):
            st.markdown(theme.now_playing_top(model), unsafe_allow_html=True)
            if model and page != "Results":
                st.button("Open results", key="np_open", on_click=_go, args=("Results",))
            st.markdown(theme.now_playing_rest(model, dataset, history), unsafe_allow_html=True)


# ── Chrome that reflects the end-of-run state ─────────────────────────────────
runs_slot.markdown(
    theme.library_runs(history, len(history) - 1 if model and history else None),
    unsafe_allow_html=True,
)

target = (model or {}).get("target") or st.session_state.get("train_target_column")
crumbs = [dataset["name"] if dataset else None, target, model["algo"]["name"] if model else None]
theme.top_bar(page, crumbs, auth_name)

stages_done = [
    dataset is not None or model is not None,
    bool(target),
    model is not None,
    model is not None and bool(st.session_state.get("pipe_evaluated")),
    model is not None and bool(st.session_state.get("pipe_predicted")),
]
theme.player_bar(dataset, stages_done, theme.primary_metric(model["metrics"]) if model else None,
                 last_target=(model or {}).get("target"))

if st.session_state.pop("_scroll_reset", False):
    theme.scroll_center_to_top(f"{page}-{time.time()}")
