import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import joblib
import io
from pathlib import Path

from src.utils.config import load_config
from src.data.preprocess import load_preprocessor
from src.data.load_data import clean_raw_data


def _user_dir() -> Path:
    username = st.session_state.get("username", "default")
    return Path("models") / username


def _load_artifacts():
    user_dir   = _user_dir()
    model_path = user_dir / "model.joblib"
    pre_path   = user_dir / "preprocessor.joblib"
    le_path    = user_dir / "label_encoder.joblib"

    if not model_path.exists():
        return None, None, None

    model         = joblib.load(model_path)
    preprocessor  = load_preprocessor(str(pre_path))
    label_encoder = joblib.load(le_path) if le_path.exists() else None
    return model, preprocessor, label_encoder


def render():
    if not st.session_state.get("_page_scrolled"):
        st.markdown("<script>window.parent.document.querySelector('section[data-testid=\"stMain\"]').scrollTo(0,0);</script>", unsafe_allow_html=True)
        st.session_state["_page_scrolled"] = True
    st.title("Predict")
    st.caption("Run a single prediction or score an entire CSV file.")

    model, preprocessor, label_encoder = _load_artifacts()

    if model is None:
        st.warning("No model found. Go to **Train** and train a model first.")
        return

    st.success("Model loaded and ready.")

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

    # ── Single prediction ─────────────────────────────────────────────────────
    with tab1:
        st.subheader("Enter customer details")

        try:
            feature_names = preprocessor.get_feature_names_out()
            # Get original feature names from transformers
            num_features = preprocessor.transformers_[0][2]
            cat_features = preprocessor.transformers_[1][2]
        except Exception:
            st.error("Could not read feature names from preprocessor.")
            return

        input_data = {}
        cols = st.columns(2)

        for i, feat in enumerate(num_features):
            with cols[i % 2]:
                input_data[feat] = st.number_input(feat, value=0.0, format="%.2f")

        for i, feat in enumerate(cat_features):
            with cols[i % 2]:
                # Get unique values from the encoder
                try:
                    encoder = preprocessor.named_transformers_["cat"]["encoder"]
                    feat_idx = cat_features.index(feat)
                    choices = encoder.categories_[feat_idx].tolist()
                except Exception:
                    choices = []
                if choices:
                    input_data[feat] = st.selectbox(feat, choices)
                else:
                    input_data[feat] = st.text_input(feat)

        st.divider()
        predict_btn = st.button("Run Prediction", type="primary")

        if predict_btn:
            try:
                input_df = pd.DataFrame([input_data])
                processed = preprocessor.transform(input_df)
                prediction = int(model.predict(processed)[0])
                predicted_label = label_encoder.inverse_transform([prediction])[0] \
                    if label_encoder else prediction

                is_classifier = hasattr(model, "predict_proba")
                if is_classifier:
                    probability = float(model.predict_proba(processed)[0][prediction])
                    label = str(predicted_label)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prediction", label)
                    col2.metric("Confidence", f"{probability:.1%}")
                    col3.metric("Risk Level", "High" if probability > 0.6 else "Medium" if probability > 0.35 else "Low")

                    # Probability gauge
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={"text": "Predicted Probability", "font": {"color": "rgba(255,255,255,0.85)"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.5)"},
                            "bar": {"color": "#f87171" if probability > 0.5 else "#667eea"},
                            "bgcolor": "rgba(255,255,255,0.04)",
                            "bordercolor": "rgba(255,255,255,0.15)",
                            "steps": [
                                {"range": [0, 35],  "color": "rgba(52,211,153,0.15)"},
                                {"range": [35, 60], "color": "rgba(245,158,11,0.15)"},
                                {"range": [60, 100],"color": "rgba(248,113,113,0.15)"},
                            ],
                        },
                        number={"suffix": "%", "font": {"color": "#ffffff"}},
                    ))
                    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})

                else:
                    predicted_value = float(model.predict(processed)[0])
                    st.metric("Predicted Value", f"{predicted_value:,.2f}")

                # ── SHAP waterfall for this prediction ───────────────────────
                try:
                    import plotly.graph_objects as go
                    import numpy as np
                    from src.explainability.shap_explainer import compute_shap_single
                    shap_vals, shap_feats, expected = compute_shap_single(model, processed, preprocessor)

                    # Keep top 15 by absolute contribution
                    order = np.argsort(np.abs(shap_vals))[::-1][:15]
                    vals  = shap_vals[order]
                    names = [shap_feats[i] for i in order]
                    # Pair each feature with its input value for the label
                    feat_vals = [list(input_data.values())[i] if i < len(input_data) else "" for i in order]
                    labels = [f"{n} = {v}" for n, v in zip(names, feat_vals)]

                    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

                    st.divider()
                    st.subheader("Why this prediction?")
                    st.caption(
                        "Each bar shows how much that feature pushed the prediction above (red) "
                        "or below (blue) the model's baseline. "
                        f"Baseline: {expected:.4f}."
                    )
                    fig_w = go.Figure(go.Bar(
                        x=vals[::-1],
                        y=labels[::-1],
                        orientation="h",
                        marker_color=colors[::-1],
                    ))
                    fig_w.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig_w.update_layout(
                        xaxis_title="SHAP contribution",
                        height=max(350, len(vals) * 28),
                    )
                    st.plotly_chart(fig_w, use_container_width=True, config={"scrollZoom": False})
                except Exception:
                    pass  # SHAP is optional — never block the prediction result

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ── Batch prediction ──────────────────────────────────────────────────────
    with tab2:
        st.subheader("Upload a CSV for batch predictions")
        st.caption("The file should have the same feature columns as the training data (target column is optional).")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

        if uploaded:
            df = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(df):,} rows**")
            st.dataframe(df.head(5), use_container_width=True)

            if st.button("Run Batch Predictions", type="primary"):
                try:
                    # Get the exact feature columns the preprocessor was trained on
                    num_features = list(preprocessor.transformers_[0][2])
                    cat_features = list(preprocessor.transformers_[1][2])
                    expected_features = num_features + cat_features

                    df_clean = clean_raw_data(df.copy())

                    # Fill any features the model needs but the file doesn't have
                    missing_cols = set(expected_features) - set(df_clean.columns)
                    if missing_cols:
                        for col in missing_cols:
                            df_clean[col] = 0.0 if col in num_features else "unknown"
                        st.warning(
                            f"These columns were missing from your file and filled with defaults: "
                            f"{sorted(missing_cols)}. Results may be less accurate."
                        )

                    X = df_clean[expected_features]
                    processed = preprocessor.transform(X)
                    predictions = model.predict(processed)

                    results = df.copy()
                    results["predicted"] = predictions

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(processed)
                        if proba.shape[1] == 2:
                            results["probability"] = proba[:, 1].round(4)
                        if label_encoder is not None:
                            results["label"] = label_encoder.inverse_transform(predictions)
                        else:
                            results["label"] = pd.Series(predictions).map({1: "Yes", 0: "No"}).values
                        pos_count = int((predictions == 1).sum())
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Records", f"{len(results):,}")
                        c2.metric("Predicted Positive", f"{pos_count:,}")
                        c3.metric("Positive Rate", f"{pos_count / len(results) * 100:.1f}%")
                    else:
                        st.metric("Total Records", f"{len(results):,}")

                    st.dataframe(results.head(20), use_container_width=True)

                    # Download button
                    csv_buffer = io.StringIO()
                    results.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "Download Predictions CSV",
                        data=csv_buffer.getvalue(),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
