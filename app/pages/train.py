import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import tempfile

from src.core.algorithm_registry import ALGORITHM_REGISTRY, list_algorithms
from src.core.metrics import get_available_metrics, DEFAULT_METRICS
from src.core.problem_detector import detect_problem_type, describe_problem
from src.data.load_data import clean_raw_data


def render():
    if not st.session_state.get("_page_scrolled"):
        st.markdown("<script>window.parent.document.querySelector('section[data-testid=\"stMain\"]').scrollTo(0,0);</script>", unsafe_allow_html=True)
        st.session_state["_page_scrolled"] = True
    st.title("Train a Model")
    st.caption("Upload any dataset, pick your algorithm, set your parameters, and train.")

    # ── Step 1: Dataset ──────────────────────────────────────────────────────
    st.header("Step 1 — Dataset")

    if "train_df_raw" in st.session_state:
        if st.button("Clear dataset & start over"):
            _clear_all()
            st.rerun()
        st.caption("Use **Reset everything** in the sidebar to also clear Results and Predict.")

    col1, col2 = st.columns([2, 1])
    with col1:
        upload = st.file_uploader("Upload a CSV file", type=["csv"])
        dataset_path_input = st.text_input(
            "Or enter a file path",
            placeholder="data/raw/dataset.csv",
            value=st.session_state.get("train_path_input", ""),
        )

    if upload:
        df_raw = pd.read_csv(upload)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_raw.to_csv(tmp.name, index=False)
        st.session_state["train_df_raw"]       = df_raw
        st.session_state["train_dataset_path"] = tmp.name
        st.session_state["train_path_input"]   = ""

    elif dataset_path_input and dataset_path_input != st.session_state.get("train_path_input", ""):
        try:
            df_raw = pd.read_csv(dataset_path_input)
            st.session_state["train_df_raw"]       = df_raw
            st.session_state["train_dataset_path"] = dataset_path_input
            st.session_state["train_path_input"]   = dataset_path_input
        except Exception as e:
            st.error(f"Could not load file: {e}")

    df_raw       = st.session_state.get("train_df_raw")
    dataset_path = st.session_state.get("train_dataset_path")

    if df_raw is not None:
        st.success(f"Loaded {len(df_raw):,} rows and {len(df_raw.columns)} columns.")
        with st.expander("Preview data"):
            st.dataframe(df_raw.head(10), use_container_width=True)

        # ── Step 2: Target column ────────────────────────────────────────────
        st.header("Step 2 — Target Column")
        cols_list = df_raw.columns.tolist()
        saved_target_idx = cols_list.index(st.session_state["train_target_column"]) \
            if st.session_state.get("train_target_column") in cols_list else len(cols_list) - 1

        target_column = st.selectbox(
            "Which column are you predicting?",
            options=cols_list,
            index=saved_target_idx,
            key="train_target_column",
        )

        if target_column:
            try:
                df_clean = clean_raw_data(df_raw.copy())
                if target_column not in df_clean.columns:
                    df_clean[target_column] = df_raw[target_column]

                problem_type = detect_problem_type(df_clean[target_column])
                st.info(f"Auto-detected: {describe_problem(problem_type)}")

                # ── Step 3: Algorithm ────────────────────────────────────────
                st.header("Step 3 — Algorithm")

                algo_mode = st.radio(
                    "How do you want to choose the algorithm?",
                    ["Auto", "Manual"],
                    horizontal=True,
                    key="train_algo_mode",
                )

                available_algos = list_algorithms(problem_type)

                if algo_mode == "Auto":
                    algorithm = "auto"
                    from src.core.algorithm_registry import AUTO_DEFAULTS
                    auto_name = AUTO_DEFAULTS[problem_type]
                    st.caption(f"Will use: {auto_name}")
                else:
                    saved_algo = st.session_state.get("train_algorithm")
                    algo_idx   = available_algos.index(saved_algo) if saved_algo in available_algos else 0
                    algorithm  = st.selectbox("Select algorithm", available_algos, index=algo_idx, key="train_algorithm")

                # ── Step 3b: Comparison algorithm (optional) ─────────────────
                with st.expander("Compare against a second algorithm (optional)"):
                    compare_enabled = st.checkbox("Enable comparison", key="train_compare_enabled")
                    compare_algorithm = None
                    if compare_enabled:
                        other_algos = [a for a in available_algos if a != algorithm]
                        compare_algorithm = st.selectbox(
                            "Second algorithm", other_algos, key="train_compare_algorithm"
                        )
                        st.caption(
                            "Both algorithms will be trained on the same data with default "
                            "parameters. Results appear side by side on the Results page."
                        )

                # ── Step 4: Hyperparameters ──────────────────────────────────
                st.header("Step 4 — Hyperparameters")

                tuning_mode = st.radio(
                    "Hyperparameter control",
                    ["Auto-tune (Optuna)", "Set manually", "Use defaults"],
                    horizontal=True,
                    key="train_tuning_mode",
                )

                manual_params = {}
                run_tune      = False

                if tuning_mode == "Auto-tune (Optuna)":
                    run_tune = True
                    n_trials = st.slider("Number of tuning trials", 5, 100, 20, key="train_n_trials")
                    st.caption("Optuna will search for the best hyperparameter combination automatically.")

                elif tuning_mode == "Set manually" and algorithm != "auto":
                    algo_entry = ALGORITHM_REGISTRY[algorithm]
                    hyperparams = algo_entry["hyperparameters"]
                    st.caption(f"Configuring hyperparameters for: {algorithm}")

                    cols = st.columns(2)
                    for i, (param_name, param_info) in enumerate(hyperparams.items()):
                        with cols[i % 2]:
                            ptype   = param_info["type"]
                            default = param_info["default"]
                            label   = f"{param_name} — {param_info['description']}"
                            skey    = f"train_param_{param_name}"

                            if ptype == "int":
                                val = st.number_input(label,
                                    min_value=param_info["min"], max_value=param_info["max"],
                                    value=default if default is not None else param_info["min"],
                                    step=1, key=skey)
                                manual_params[param_name] = int(val)
                            elif ptype == "float":
                                val = st.number_input(label,
                                    min_value=float(param_info["min"]), max_value=float(param_info["max"]),
                                    value=float(default if default is not None else param_info["min"]),
                                    format="%.4f", key=skey)
                                manual_params[param_name] = float(val)
                            elif ptype == "cat":
                                choices     = [str(c) for c in param_info["choices"]]
                                default_str = str(default) if default is not None else choices[0]
                                idx         = choices.index(default_str) if default_str in choices else 0
                                val         = st.selectbox(label, choices, index=idx, key=skey)
                                manual_params[param_name] = val
                else:
                    st.caption("Default parameters will be used.")

                # ── Step 5: Metrics ──────────────────────────────────────────
                st.header("Step 5 — Evaluation Metrics")

                metrics_mode = st.radio(
                    "Metrics selection",
                    ["Auto (recommended)", "Manual selection"],
                    horizontal=True,
                    key="train_metrics_mode",
                )

                selected_metrics = None
                if metrics_mode == "Manual selection":
                    available = get_available_metrics(problem_type)
                    defaults  = DEFAULT_METRICS[problem_type]
                    selected_metrics = st.multiselect(
                        "Choose metrics", options=available, default=defaults,
                        key="train_selected_metrics",
                    )
                    if not selected_metrics:
                        st.warning("Select at least one metric.")
                else:
                    st.caption(f"Will evaluate: {', '.join(DEFAULT_METRICS[problem_type])}")

                # ── Run ──────────────────────────────────────────────────────
                st.divider()
                run_col, _ = st.columns([1, 3])
                with run_col:
                    run_button = st.button("Train Model", type="primary", use_container_width=True)

                if run_button:
                    if selected_metrics is not None and len(selected_metrics) == 0:
                        st.error("Please select at least one metric.")
                    else:
                        compare_algo = st.session_state.get("train_compare_algorithm") \
                            if st.session_state.get("train_compare_enabled") else None
                        _run_training(
                            dataset_path=dataset_path,
                            target_column=target_column,
                            algorithm=algorithm,
                            manual_params=manual_params if tuning_mode == "Set manually" else {},
                            run_tune=run_tune,
                            selected_metrics=selected_metrics,
                            n_trials=st.session_state.get("train_n_trials", 20),
                            compare_algorithm=compare_algo,
                        )

                if "last_metrics" in st.session_state:
                    st.divider()
                    st.subheader("Last Training Result")
                    metrics = st.session_state["last_metrics"]
                    cols = st.columns(len(metrics))
                    for col, (name, value) in zip(cols, metrics.items()):
                        col.metric(name.upper(), f"{value:.4f}")
                    st.info("Go to Results for full charts and feature importances.")

            except Exception as e:
                st.error(f"Error reading target column: {e}")


def _clear_all():
    """Wipe every trace of the current user's training data — session state and disk."""
    import shutil
    from pathlib import Path

    all_keys = [
        "train_df_raw", "train_dataset_path", "train_target_column",
        "train_algorithm", "train_tuning_mode", "train_algo_mode",
        "train_metrics_mode", "train_selected_metrics", "train_path_input",
        "training_history", "last_metrics", "last_problem_type", "last_algorithm",
        "last_preprocessor", "last_target", "last_feature_importances",
        "last_cm", "last_eval_data", "last_shap_values", "last_shap_features",
        "last_shap_X", "compare_metrics", "compare_algorithm",
    ]
    for key in all_keys:
        st.session_state.pop(key, None)

    # Delete the user's saved state from disk so Results also shows empty
    user_dir = _user_dir()
    if user_dir.exists():
        shutil.rmtree(user_dir)


def _run_training(dataset_path, target_column, algorithm, manual_params, run_tune, selected_metrics, n_trials, compare_algorithm=None):
    from src.models.train import train
    from src.utils.config import load_config

    config = load_config()
    config["tuning"]["n_trials"] = n_trials

    with st.status("Training in progress...", expanded=True) as status:
        st.write("Loading and validating dataset...")
        try:
            model, preprocessor, metrics, problem_type, eval_data = train(
                config=config,
                dataset_path=dataset_path,
                target_column=target_column,
                algorithm=algorithm,
                params=manual_params if manual_params else None,
                metrics=selected_metrics,
                tune=run_tune,
            )
            if compare_algorithm:
                st.write(f"Training comparison model: {compare_algorithm}...")
                _, _, compare_metrics, _, _ = train(
                    config=config,
                    dataset_path=dataset_path,
                    target_column=target_column,
                    algorithm=compare_algorithm,
                    params=None,
                    metrics=selected_metrics,
                    tune=False,
                )
                st.session_state["compare_metrics"]   = compare_metrics
                st.session_state["compare_algorithm"] = compare_algorithm
            else:
                st.session_state.pop("compare_metrics",   None)
                st.session_state.pop("compare_algorithm", None)
            status.update(label="Training complete", state="complete")
        except Exception as e:
            status.update(label="Training failed", state="error")
            st.error(str(e))
            return

    from src.models.evaluate import get_feature_importances
    import json
    from pathlib import Path

    fi = get_feature_importances(model, preprocessor)

    # Read the confusion matrix the model just wrote to disk and store in session
    # state so Results always shows data from the same run as the metrics.
    cm_data = {}
    metrics_path = Path("logs/metrics/eval_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            cm_data = json.load(f)

    st.session_state["last_metrics"]             = metrics
    st.session_state["last_problem_type"]        = problem_type
    st.session_state["last_algorithm"]           = algorithm
    st.session_state["last_preprocessor"]        = preprocessor
    st.session_state["last_target"]              = target_column
    st.session_state["last_feature_importances"] = fi
    st.session_state["last_cm"]                  = cm_data
    # Extract X_test before storing eval_data (it's a numpy array, not JSON-safe)
    X_test_arr = eval_data.pop("X_test", None)
    st.session_state["last_eval_data"] = eval_data

    # Compute SHAP values on the test set
    if X_test_arr is not None:
        try:
            from src.explainability.shap_explainer import compute_shap_summary
            shap_vals, shap_features, shap_X = compute_shap_summary(model, X_test_arr, preprocessor)
            st.session_state["last_shap_values"]   = shap_vals
            st.session_state["last_shap_features"] = shap_features
            st.session_state["last_shap_X"]        = shap_X
        except Exception as shap_err:
            st.session_state["last_shap_values"] = None
            print(f"SHAP skipped: {shap_err}")

    # Build session-scoped training history.
    # Clear the history if the target column changed since the last run.
    from datetime import datetime
    history = st.session_state.get("training_history", [])
    if history and history[-1].get("target_column") != target_column:
        history = []

    history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "target_column": target_column,
        "problem_type": problem_type,
        "algorithm": algorithm,
        **metrics,
    })
    st.session_state["training_history"] = history

    st.success("Model trained and saved successfully.")
    _save_state_to_disk()


def _user_dir() -> "Path":
    from pathlib import Path
    username = st.session_state.get("username", "default")
    d = Path("models") / username
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_state_to_disk():
    """Persist the current session training results to disk so Results survives a server restart."""
    import json
    import numpy as np
    import shutil
    from pathlib import Path

    state_dir = _user_dir()

    # Copy the freshly-trained model files into the user's own folder
    for fname in ("model.joblib", "preprocessor.joblib", "label_encoder.joblib"):
        src = Path("models") / fname
        if src.exists():
            shutil.copy2(src, state_dir / fname)

    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _make_serializable(d):
        if isinstance(d, dict):
            return {k: _make_serializable(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_make_serializable(i) for i in d]
        return _json_safe(d)

    state = {
        "metrics":             _make_serializable(st.session_state.get("last_metrics", {})),
        "problem_type":        st.session_state.get("last_problem_type", ""),
        "algorithm":           str(st.session_state.get("last_algorithm", "")),
        "target_column":       st.session_state.get("last_target", ""),
        "feature_importances": _make_serializable(st.session_state.get("last_feature_importances")),
        "cm_data":             _make_serializable(st.session_state.get("last_cm", {})),
        "eval_data":           _make_serializable(st.session_state.get("last_eval_data")),
        "compare_metrics":     _make_serializable(st.session_state.get("compare_metrics")),
        "compare_algorithm":   st.session_state.get("compare_algorithm"),
        "training_history":    _make_serializable(st.session_state.get("training_history", [])),
    }

    with open(state_dir / "last_run_state.json", "w") as f:
        json.dump(state, f)

    shap_vals = st.session_state.get("last_shap_values")
    shap_X    = st.session_state.get("last_shap_X")
    shap_feat = st.session_state.get("last_shap_features")

    if shap_vals is not None:
        np.save(str(state_dir / "last_shap_values.npy"), shap_vals)
    if shap_X is not None:
        np.save(str(state_dir / "last_shap_X.npy"), shap_X)
    if shap_feat is not None:
        with open(state_dir / "last_shap_features.json", "w") as f:
            json.dump(shap_feat, f)
