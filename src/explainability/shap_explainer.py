import numpy as np
import shap

shap.initjs()

_TREE_MODELS = {
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
}
_LINEAR_MODELS = {
    "LogisticRegression", "Ridge", "Lasso", "LinearRegression",
}


def _get_explainer(model, X_background):
    name = type(model).__name__
    if name in _TREE_MODELS:
        return shap.TreeExplainer(model)
    if name in _LINEAR_MODELS:
        return shap.LinearExplainer(model, X_background)
    # Fallback: KernelExplainer with k-means background (slow — capped at 50 samples)
    bg = shap.kmeans(X_background, min(10, X_background.shape[0]))
    fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    return shap.KernelExplainer(fn, bg)


def _clean_names(preprocessor):
    try:
        return [
            n.replace("num__", "").replace("cat__", "")
            for n in preprocessor.get_feature_names_out()
        ]
    except Exception:
        return None


def _to_2d(raw):
    """Collapse any SHAP output into a 2-D (samples × features) array."""
    if isinstance(raw, list):
        # Binary: list of 2 arrays → take positive class
        # Multiclass: list of N arrays → mean absolute
        return raw[1] if len(raw) == 2 else np.mean([np.abs(r) for r in raw], axis=0)
    if hasattr(raw, "values"):          # shap.Explanation object (shap >= 0.40)
        v = raw.values
        if v.ndim == 3:                 # (samples, features, classes)
            return np.mean(np.abs(v), axis=2)
        return v
    if raw.ndim == 3:                   # (classes, samples, features)
        return np.mean(np.abs(raw), axis=0)
    return raw


def compute_shap_summary(model, X_test, preprocessor, max_samples: int = 200):
    """
    Compute SHAP values on a random sample of the test set.

    Returns
    -------
    shap_values : ndarray (n_samples, n_features)
    feature_names : list[str]
    X_sample : ndarray  — the rows used (needed for beeswarm colouring)
    """
    rng = np.random.default_rng(42)
    n = min(max_samples, X_test.shape[0])
    idx = rng.choice(X_test.shape[0], n, replace=False)
    X_sample = X_test[idx]

    # Densify — sparse matrices come from OneHotEncoder and break arithmetic later
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()

    explainer = _get_explainer(model, X_sample)

    # KernelExplainer is very slow — limit further
    if type(explainer).__name__ == "KernelExplainer":
        X_sample = X_sample[:50]

    raw = explainer.shap_values(X_sample)
    shap_vals = _to_2d(raw)

    feature_names = _clean_names(preprocessor) or [f"f{i}" for i in range(shap_vals.shape[1])]
    return shap_vals, feature_names, X_sample


def compute_shap_single(model, X_single, preprocessor):
    """
    Compute SHAP values for one row (used by the Predict page waterfall chart).

    Returns
    -------
    shap_vals : ndarray (n_features,)
    feature_names : list[str]
    expected_value : float
    """
    if hasattr(X_single, "toarray"):
        X_single = X_single.toarray()
    explainer = _get_explainer(model, X_single)
    raw = explainer.shap_values(X_single)
    shap_2d = _to_2d(raw)
    shap_vals = shap_2d[0]  # first (only) row

    ev = explainer.expected_value
    expected_value = float(np.array(ev).ravel()[0])

    feature_names = _clean_names(preprocessor) or [f"f{i}" for i in range(len(shap_vals))]
    return shap_vals, feature_names, expected_value
