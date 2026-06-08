import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    classification_report,
)


def _metric_explanation(metric: str, value: float, target: str, problem_type: str) -> str:
    t = f"**{target}**"
    pct = f"{value * 100:.1f}%"
    val = f"{value:.4f}"

    explanations = {
        "rmse": (
            f"On average, the model's predictions are off by **{value:,.2f}** "
            f"from the actual {t} value. "
            f"This is in the same units as {t} — smaller is better."
        ),
        "mae": (
            f"The typical prediction error is **{value:,.2f}** away from the real {t}. "
            f"Unlike RMSE, this is not sensitive to large outliers."
        ),
        "r2": (
            f"The model explains **{pct}** of the variation in {t}. "
            f"A score of 100% means perfect predictions; 0% means the model is no better than guessing the average."
        ),
        "mape": (
            f"On average, predictions are off by **{value:.1f}%** relative to the actual {t} value. "
            f"This is a percentage error so it is easy to compare across datasets."
        ),
        "accuracy": (
            f"**{pct}** of all predictions for {t} were correct (both positives and negatives). "
            f"Can be misleading if one class is much more common than the other."
        ),
        "precision": (
            f"When the model predicts a positive {t}, it is correct **{pct}** of the time. "
            f"High precision means fewer false alarms."
        ),
        "recall": (
            f"The model correctly identifies **{pct}** of all actual positives in {t}. "
            f"Low recall means the model is missing real cases."
        ),
        "f1": (
            f"The balance between precision and recall is **{pct}**. "
            f"This is the most reliable single metric when both false alarms and missed cases matter."
        ),
        "roc_auc": (
            f"If you randomly pick one positive and one negative {t} case, "
            f"the model correctly ranks the positive as higher risk **{pct}** of the time. "
            f"Above 80% is generally considered good."
        ),
    }
    return explanations.get(metric.lower(), f"Value: {val}")


def _user_dir() -> Path:
    username = st.session_state.get("username", "default")
    return Path("models") / username


def _restore_from_disk():
    """Load the last training run from disk if session state was wiped (e.g. server restart)."""
    if "last_metrics" in st.session_state:
        return
    state_path = _user_dir() / "last_run_state.json"
    if not state_path.exists():
        return
    try:
        with open(state_path) as f:
            state = json.load(f)
        st.session_state["last_metrics"]             = state.get("metrics", {})
        st.session_state["last_problem_type"]        = state.get("problem_type", "")
        st.session_state["last_algorithm"]           = state.get("algorithm", "")
        st.session_state["last_target"]              = state.get("target_column", "")
        st.session_state["last_feature_importances"] = state.get("feature_importances")
        st.session_state["last_cm"]                  = state.get("cm_data", {})
        st.session_state["last_eval_data"]           = state.get("eval_data")
        st.session_state["training_history"]         = state.get("training_history", [])
        if state.get("compare_metrics"):
            st.session_state["compare_metrics"]   = state["compare_metrics"]
            st.session_state["compare_algorithm"] = state["compare_algorithm"]
        user_dir = _user_dir()
        sv_path  = user_dir / "last_shap_values.npy"
        sx_path  = user_dir / "last_shap_X.npy"
        sf_path  = user_dir / "last_shap_features.json"
        if sv_path.exists() and sx_path.exists() and sf_path.exists():
            st.session_state["last_shap_values"]   = np.load(str(sv_path))
            st.session_state["last_shap_X"]        = np.load(str(sx_path))
            with open(sf_path) as f:
                st.session_state["last_shap_features"] = json.load(f)
    except Exception as e:
        print(f"Could not restore state from disk: {e}")


def render():
    if not st.session_state.get("_page_scrolled"):
        st.markdown("<script>window.parent.document.querySelector('section[data-testid=\"stMain\"]').scrollTo(0,0);</script>", unsafe_allow_html=True)
        st.session_state["_page_scrolled"] = True
    st.title("Results")

    _restore_from_disk()

    if "last_metrics" not in st.session_state:
        st.warning("No results yet. Go to Train to run a model first.")
        return

    metrics      = st.session_state["last_metrics"]
    problem_type = st.session_state.get("last_problem_type", "")
    algorithm    = st.session_state.get("last_algorithm", "unknown")
    fi           = st.session_state.get("last_feature_importances")
    is_classification = "classification" in problem_type

    st.caption(f"Problem type: {problem_type}  |  Algorithm: {algorithm}")
    st.divider()

    # ── Model comparison (shown when a second algorithm was trained) ───────────
    compare_metrics   = st.session_state.get("compare_metrics")
    compare_algorithm = st.session_state.get("compare_algorithm")

    if compare_metrics:
        st.subheader("Model Comparison")
        all_metric_names = sorted(set(list(metrics.keys()) + list(compare_metrics.keys())))
        a_vals = [metrics.get(m, 0) for m in all_metric_names]
        b_vals = [compare_metrics.get(m, 0) for m in all_metric_names]

        fig = go.Figure()
        fig.add_trace(go.Bar(name=str(algorithm), x=all_metric_names, y=a_vals, marker_color="#4C78A8"))
        fig.add_trace(go.Bar(name=str(compare_algorithm), x=all_metric_names, y=b_vals, marker_color="#F58518"))
        fig.update_layout(
            barmode="group",
            yaxis_title="Score",
            height=380,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})

        # Winner per metric
        rows = []
        for m in all_metric_names:
            a, b = metrics.get(m), compare_metrics.get(m)
            if a is None or b is None:
                winner = "—"
            else:
                # For error metrics lower is better; for everything else higher is better
                lower_is_better = m in ("rmse", "mae", "mape")
                winner = str(algorithm) if (a < b if lower_is_better else a > b) else str(compare_algorithm)
            rows.append({"Metric": m.upper(), str(algorithm): f"{a:.4f}" if a else "—",
                         str(compare_algorithm): f"{b:.4f}" if b else "—", "Winner": winner})
        st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)
        st.divider()

    # ── Metrics cards ─────────────────────────────────────────────────────────
    target = st.session_state.get("last_target", "target")

    st.subheader(f"Performance Metrics — {algorithm}")
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name.upper(), f"{value:.4f}")

    st.divider()
    st.subheader("What do these numbers mean?")
    for name, value in metrics.items():
        explanation = _metric_explanation(name, value, target, problem_type)
        st.markdown(f"**{name.upper()}** — {explanation}")
        st.write("")

    # ── Confusion matrix — binary classification only ─────────────────────────
    if is_classification:
        # Always read from session state so the matrix matches the current run's metrics
        saved_file = st.session_state.get("last_cm", {})

        if all(k in saved_file for k in ("tn", "fp", "fn", "tp")):
            st.divider()
            st.subheader("Confusion Matrix")

            tn = saved_file["tn"]
            fp = saved_file["fp"]
            fn = saved_file["fn"]
            tp = saved_file["tp"]

            cm_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual: No", "Actual: Yes"],
                columns=["Predicted: No", "Predicted: Yes"],
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                fig = px.imshow(
                    cm_df, text_auto=True,
                    color_continuous_scale=["rgba(30,27,75,0.6)", "#667eea", "#a78bfa"],
                    title="Confusion Matrix", aspect="auto",
                )
                fig.update_layout(height=350)
                fig.update_traces(textfont=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})

            with col2:
                st.markdown("**What each cell means:**")
                st.markdown(f"""
| Cell | Count | Meaning |
|---|---|---|
| True Negatives | {tn} | Predicted No, actually No |
| False Positives | {fp} | Predicted Yes, actually No |
| False Negatives | {fn} | Predicted No, actually Yes — missed cases |
| True Positives | {tp} | Predicted Yes, actually Yes |
""")
                t_val = saved_file.get("threshold")
                if t_val is not None:
                    st.metric("Optimal Threshold", f"{t_val:.2f}",
                              help="Lower = catches more positives at cost of more false alarms")
                    st.metric("F1 at Optimal Threshold", f"{saved_file.get('f1_at_threshold', 0):.4f}")

    # ── Feature importances ───────────────────────────────────────────────────
    if fi and isinstance(fi, dict):
        st.divider()
        st.subheader("Feature Importances")

        top_n     = st.slider("Show top N features", 5, min(30, len(fi)), 15)
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n])
        labels    = [k.replace("num__", "").replace("cat__", "") for k in fi_sorted.keys()]
        values    = list(fi_sorted.values())

        fig = go.Figure(go.Bar(
            x=values, y=labels,
            orientation="h",
            marker_color="steelblue",
        ))
        fig.update_layout(
            title=f"Top {top_n} Features by Importance",
            xaxis_title="Importance Score",
            yaxis=dict(autorange="reversed"),
            height=max(400, top_n * 28),
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})

    # ── Evaluation charts ─────────────────────────────────────────────────────
    eval_data = st.session_state.get("last_eval_data")
    if not eval_data:
        st.divider()
        st.info("Evaluation charts (Actual vs Predicted, ROC curve, per-class breakdown, SHAP) will appear here after you retrain the model. The server was restarted since the last run, which cleared the session.")
    if eval_data:
        y_test = np.array(eval_data["y_test"])
        y_pred = np.array(eval_data["y_pred"])
        y_prob = eval_data.get("y_prob")
        classes = eval_data.get("classes")        # original label names if encoded

        st.divider()

        if is_classification:
            # ── Per-class breakdown ──────────────────────────────────────────
            st.subheader("Per-Class Performance")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_rows = {
                k: v for k, v in report.items()
                if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg")
            }
            if report_rows:
                report_df = pd.DataFrame(report_rows).T.reset_index()
                report_df.columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
                # Decode numeric class labels back to original names if available
                if classes is not None:
                    report_df["Class"] = report_df["Class"].apply(
                        lambda x: classes[int(x)] if str(x).lstrip("-").isdigit() and int(x) < len(classes) else x
                    )
                report_df = report_df.sort_values("F1-Score", ascending=False)
                fig = go.Figure()
                for metric_name, color in [("Precision", "#4C78A8"), ("Recall", "#F58518"), ("F1-Score", "#54A24B")]:
                    fig.add_trace(go.Bar(
                        name=metric_name,
                        x=report_df["Class"].astype(str),
                        y=report_df[metric_name],
                        marker_color=color,
                    ))
                fig.update_layout(
                    barmode="group",
                    title="Precision / Recall / F1 per Class",
                    xaxis_title="Class",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    height=400,
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})

            # ── ROC & PR curves — binary only ────────────────────────────────
            if problem_type == "binary_classification" and y_prob is not None:
                prob_arr = np.array(y_prob)
                scores = prob_arr[:, 1] if prob_arr.ndim == 2 else prob_arr

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, scores)
                    roc_auc_val = auc(fpr, tpr)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"Model (AUC = {roc_auc_val:.3f})",
                        line=dict(color="#4C78A8", width=2),
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Random classifier",
                        line=dict(color="gray", dash="dash"),
                    ))
                    fig.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=380,
                        legend=dict(x=0.4, y=0.1),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})
                    st.caption(
                        "The ROC curve shows the trade-off between catching real positives "
                        "(True Positive Rate) and raising false alarms (False Positive Rate). "
                        "A curve hugging the top-left corner means a strong model. "
                        f"AUC = {roc_auc_val:.3f} — closer to 1.0 is better."
                    )

                with col2:
                    st.subheader("Precision-Recall Curve")
                    prec, rec, _ = precision_recall_curve(y_test, scores)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rec, y=prec, mode="lines",
                        name="PR Curve",
                        line=dict(color="#F58518", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(245,133,24,0.1)",
                    ))
                    fig.update_layout(
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1]),
                        height=380,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})
                    st.caption(
                        "The Precision-Recall curve shows the trade-off between how many "
                        "predicted positives are correct (Precision) and how many real "
                        "positives were found (Recall). Useful when one class is rare."
                    )

        else:
            # ── Regression charts ────────────────────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Actual vs Predicted")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test, y=y_pred, mode="markers",
                    marker=dict(color="#4C78A8", opacity=0.5, size=5),
                    name="Predictions",
                ))
                lim = [float(min(y_test.min(), y_pred.min())),
                       float(max(y_test.max(), y_pred.max()))]
                fig.add_trace(go.Scatter(
                    x=lim, y=lim, mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Perfect fit",
                ))
                fig.update_layout(
                    xaxis_title=f"Actual {target}",
                    yaxis_title=f"Predicted {target}",
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})
                st.caption(
                    "Each dot is one row from the test set. Points along the red dashed line "
                    "are perfect predictions. Dots scattered far from the line are errors."
                )

            with col2:
                st.subheader("Residuals")
                residuals = y_test - y_pred
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_pred, y=residuals, mode="markers",
                    marker=dict(color="#F58518", opacity=0.5, size=5),
                    name="Residual",
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    xaxis_title=f"Predicted {target}",
                    yaxis_title="Residual (Actual − Predicted)",
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})
                st.caption(
                    "Residuals should be randomly scattered around 0. "
                    "A pattern (curve or funnel shape) means the model is systematically "
                    "wrong for certain prediction ranges."
                )

    # ── SHAP Explainability ───────────────────────────────────────────────────
    shap_vals    = st.session_state.get("last_shap_values")
    shap_features = st.session_state.get("last_shap_features")
    shap_X       = st.session_state.get("last_shap_X")

    if shap_vals is not None and shap_features is not None:
        st.divider()
        st.subheader("SHAP — Why did the model make these predictions?")
        st.caption(
            "SHAP (SHapley Additive Explanations) breaks each prediction into the "
            "contribution of every feature. Positive SHAP = pushes prediction higher. "
            "Negative SHAP = pushes prediction lower."
        )

        top_n_shap = st.slider("Number of features to show", 5, min(30, len(shap_features)), 15, key="shap_top_n")

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n_shap]
        top_names  = [shap_features[i] for i in order]
        top_mean   = [float(mean_abs[i]) for i in order]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Global importance — mean |SHAP value|**")
            st.caption("How much does each feature move the prediction on average across all test rows?")
            fig = go.Figure(go.Bar(
                x=top_mean[::-1],
                y=top_names[::-1],
                orientation="h",
                marker_color="#4C78A8",
            ))
            fig.update_layout(
                xaxis_title="Mean |SHAP value|",
                height=max(380, top_n_shap * 26),
                margin=dict(l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})

        with col2:
            st.markdown("**Beeswarm — value & direction for every test row**")
            st.caption(
                "Each dot is one row from the test set. "
                "Dots to the right pushed the prediction UP; to the left pushed it DOWN. "
                "Color shows the feature's value: blue = low, red = high."
            )
            if shap_X is not None:
                fig2 = go.Figure()
                for rank, i in enumerate(reversed(order)):
                    feat_col = shap_X[:, i]
                    col_min, col_max = feat_col.min(), feat_col.max()
                    norm = (feat_col - col_min) / (col_max - col_min + 1e-9)
                    colors = [
                        f"rgb({int(255*v)}, {int(30*(1-v)+30*v)}, {int(255*(1-v))})"
                        for v in norm
                    ]
                    y_jitter = np.full(len(shap_vals), rank) + np.random.default_rng(i).uniform(-0.3, 0.3, len(shap_vals))
                    fig2.add_trace(go.Scatter(
                        x=shap_vals[:, i],
                        y=y_jitter,
                        mode="markers",
                        marker=dict(color=colors, size=4, opacity=0.7),
                        name=shap_features[i],
                        showlegend=False,
                        hovertemplate=f"<b>{shap_features[i]}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
                    ))
                fig2.add_vline(x=0, line_dash="dash", line_color="gray")
                fig2.update_layout(
                    xaxis_title="SHAP value",
                    yaxis=dict(
                        tickvals=list(range(top_n_shap)),
                        ticktext=top_names[::-1],
                        showgrid=False,
                    ),
                    height=max(380, top_n_shap * 26),
                    margin=dict(l=10, r=10),
                )
                st.plotly_chart(fig2, use_container_width=True, config={"scrollZoom": False})

    # ── Metric chart — always visible after any training ─────────────────────
    history = [r for r in st.session_state.get("training_history", [])
               if r.get("target_column") == target]

    if len(history) >= 1:
        st.divider()
        n_runs = len(history)
        st.subheader("Metric Chart")
        st.caption(
            f"{n_runs} run(s) this session for **{target}**. "
            "Train again with different settings to compare runs."
        )

        hist_df = pd.DataFrame(history)
        hist_df["run"] = [f"Run {i+1}" for i in range(n_runs)]
        numeric_cols = hist_df.select_dtypes("number").columns.tolist()
        relevant_cols = [c for c in numeric_cols if c in list(metrics.keys())]

        if relevant_cols:
            metric_to_plot = st.selectbox("Select metric to plot over time", relevant_cols)

            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
            fig = px.line(
                hist_df, x="timestamp", y=metric_to_plot,
                markers=True,
                hover_data=["algorithm"],
                title=f"{metric_to_plot.upper()} over time — {target}",
            )
            fig.update_traces(line_color="steelblue", marker_size=9)
            fig.update_layout(xaxis_title="Time", yaxis_title=metric_to_plot, height=340)
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})
            if n_runs == 1:
                st.caption("Train again with different settings to see how this metric changes over time.")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export")
    st.caption("Download a full report you can open in Excel or Numbers on your Mac.")

    import io
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:

        # Sheet 1 — Summary
        summary_rows = [
            {"Field": "Algorithm",    "Value": str(algorithm)},
            {"Field": "Problem Type", "Value": problem_type},
            {"Field": "Target Column","Value": target},
        ]
        for k, v in metrics.items():
            summary_rows.append({"Field": k.upper(), "Value": round(v, 6)})
        if compare_metrics:
            for k, v in compare_metrics.items():
                summary_rows.append({"Field": f"{compare_algorithm} — {k.upper()}", "Value": round(v, 6)})
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2 — Feature Importances
        if fi and isinstance(fi, dict):
            fi_df = pd.DataFrame([
                {"Feature": k.replace("num__", "").replace("cat__", ""), "Importance": round(v, 6)}
                for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)
            ])
            fi_df.to_excel(writer, sheet_name="Feature Importances", index=False)

        # Sheet 3 — Per-class report (classification only)
        if eval_data and is_classification:
            _y_test = np.array(eval_data["y_test"])
            _y_pred = np.array(eval_data["y_pred"])
            report  = classification_report(_y_test, _y_pred, output_dict=True, zero_division=0)
            report_rows = []
            for cls, vals in report.items():
                if isinstance(vals, dict):
                    row = {"Class": cls}
                    row.update({k: round(v, 4) for k, v in vals.items()})
                    report_rows.append(row)
            if report_rows:
                pd.DataFrame(report_rows).to_excel(writer, sheet_name="Per-Class Report", index=False)

        # Sheet 4 — Training history
        history = [r for r in st.session_state.get("training_history", [])
                   if r.get("target_column") == target]
        if history:
            pd.DataFrame(history).to_excel(writer, sheet_name="Training History", index=False)

    excel_buf.seek(0)
    st.download_button(
        label="Download full report (.xlsx)",
        data=excel_buf,
        file_name=f"{algorithm}_{target}_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Opens in Excel or Numbers. Contains metrics, feature importances, and per-class breakdown.",
    )
