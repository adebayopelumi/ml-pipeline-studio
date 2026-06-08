import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import streamlit.components.v1 as components
import yaml
from pathlib import Path

AUTH_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "auth_config.yaml"


def _load_config():
    with open(AUTH_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _save_config(config: dict):
    with open(AUTH_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def render():
    components.html(
        "<script>window.parent.document.querySelector('section[data-testid=\"stMain\"]').scrollTo(0,0);</script>",
        height=0,
    )
    st.title("User Management")
    st.caption("Add or remove users who can log into ML Pipeline Studio.")

    if st.session_state.get("username") != "admin":
        st.error("Access denied. This page is only available to the admin user.")
        return

    if not AUTH_CONFIG_PATH.exists():
        st.error("Auth config file not found. Restart the app to regenerate it.")
        return

    config = _load_config()
    users  = config.get("credentials", {}).get("usernames", {})

    # ── Current users ─────────────────────────────────────────────────────────
    st.subheader("Current Users")
    if not users:
        st.info("No users configured.")
    else:
        for username, info in users.items():
            col1, col2, col3, col4 = st.columns([2, 2, 3, 1])
            col1.markdown(f"**{username}**")
            col2.markdown(info.get("name", "—"))
            col3.markdown(info.get("email", "—"))
            if col4.button("Remove", key=f"remove_{username}"):
                del config["credentials"]["usernames"][username]
                _save_config(config)
                st.success(f"User **{username}** removed.")
                st.rerun()

    st.divider()

    # ── Add new user ──────────────────────────────────────────────────────────
    st.subheader("Add New User")
    with st.form("add_user_form", clear_on_submit=True):
        new_username = st.text_input("Username", placeholder="jsmith")
        new_name     = st.text_input("Full name", placeholder="Jane Smith")
        new_email    = st.text_input("Email", placeholder="jsmith@example.com")
        new_password = st.text_input("Password", type="password", placeholder="At least 6 characters")
        submitted    = st.form_submit_button("Add User", type="primary")

    if submitted:
        if not new_username or not new_name or not new_email or not new_password:
            st.error("All fields are required.")
        elif len(new_password) < 6:
            st.error("Password must be at least 6 characters.")
        elif new_username in users:
            st.error(f"Username **{new_username}** already exists.")
        else:
            config["credentials"]["usernames"][new_username] = {
                "name":     new_name,
                "email":    new_email,
                "password": new_password,  # auto-hashed by streamlit-authenticator on next login
            }
            _save_config(config)
            st.success(f"User **{new_username}** added. They can log in immediately.")
            st.rerun()

    st.divider()
    st.caption(
        "Passwords are stored as bcrypt hashes. "
        "A plain-text password entered above will be hashed automatically on the user's first login."
    )
