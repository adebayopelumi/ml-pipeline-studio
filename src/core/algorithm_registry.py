from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


# Each algorithm entry defines:
#   model_class      : the sklearn-compatible class
#   problem_type     : "classification", "regression", or "both"
#   default_params   : sensible out-of-the-box parameters
#   hyperparameters  : all tunable params with type, range, and description
#   tune_space       : Optuna suggest calls for auto-tuning

ALGORITHM_REGISTRY = {

    # ── Classification ──────────────────────────────────────────────────────
    "random_forest_classifier": {
        "model_class": RandomForestClassifier,
        "problem_type": "classification",
        "default_params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        "hyperparameters": {
            "n_estimators":      {"type": "int",   "min": 50,   "max": 500,  "default": 100,  "description": "Number of trees"},
            "max_depth":         {"type": "int",   "min": 2,    "max": 30,   "default": None, "description": "Max depth of each tree (None = unlimited)"},
            "min_samples_split": {"type": "int",   "min": 2,    "max": 20,   "default": 2,    "description": "Min samples to split a node"},
            "min_samples_leaf":  {"type": "int",   "min": 1,    "max": 10,   "default": 1,    "description": "Min samples in a leaf node"},
            "max_features":      {"type": "cat",   "choices": ["sqrt", "log2", None], "default": "sqrt", "description": "Features to consider per split"},
        },
        "tune_space": lambda t: {
            "n_estimators":      t.suggest_int("n_estimators", 50, 500),
            "max_depth":         t.suggest_int("max_depth", 2, 30),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  t.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      t.suggest_categorical("max_features", ["sqrt", "log2"]),
        },
    },

    "logistic_regression": {
        "model_class": LogisticRegression,
        "problem_type": "classification",
        "default_params": {"max_iter": 1000, "random_state": 42},
        "hyperparameters": {
            "C":       {"type": "float", "min": 0.001, "max": 100.0, "default": 1.0, "description": "Regularization strength (smaller = stronger)"},
            "penalty": {"type": "cat",   "choices": ["l1", "l2"],    "default": "l2", "description": "Regularization type"},
            "solver":  {"type": "cat",   "choices": ["liblinear", "saga"], "default": "liblinear", "description": "Optimization algorithm"},
        },
        "tune_space": lambda t: {
            "C":       t.suggest_float("C", 0.001, 100.0, log=True),
            "penalty": t.suggest_categorical("penalty", ["l1", "l2"]),
            "solver":  t.suggest_categorical("solver", ["liblinear", "saga"]),
        },
    },

    "gradient_boosting_classifier": {
        "model_class": GradientBoostingClassifier,
        "problem_type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "hyperparameters": {
            "n_estimators":  {"type": "int",   "min": 50,    "max": 500,  "default": 100,  "description": "Number of boosting stages"},
            "max_depth":     {"type": "int",   "min": 2,     "max": 10,   "default": 3,    "description": "Max depth of each tree"},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.5,  "default": 0.1,  "description": "Shrinks contribution of each tree"},
            "subsample":     {"type": "float", "min": 0.5,   "max": 1.0,  "default": 1.0,  "description": "Fraction of samples per tree"},
        },
        "tune_space": lambda t: {
            "n_estimators":  t.suggest_int("n_estimators", 50, 500),
            "max_depth":     t.suggest_int("max_depth", 2, 10),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "subsample":     t.suggest_float("subsample", 0.5, 1.0),
        },
    },

    "xgboost_classifier": {
        "model_class": XGBClassifier,
        "problem_type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42, "eval_metric": "logloss", "verbosity": 0},
        "hyperparameters": {
            "n_estimators":  {"type": "int",   "min": 50,    "max": 500,  "default": 100,  "description": "Number of boosting rounds"},
            "max_depth":     {"type": "int",   "min": 2,     "max": 12,   "default": 6,    "description": "Max depth of each tree"},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.5,  "default": 0.1,  "description": "Step size shrinkage"},
            "subsample":     {"type": "float", "min": 0.5,   "max": 1.0,  "default": 1.0,  "description": "Fraction of samples per tree"},
            "colsample_bytree": {"type": "float", "min": 0.3, "max": 1.0, "default": 1.0,  "description": "Fraction of features per tree"},
        },
        "tune_space": lambda t: {
            "n_estimators":     t.suggest_int("n_estimators", 50, 500),
            "max_depth":        t.suggest_int("max_depth", 2, 12),
            "learning_rate":    t.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "subsample":        t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        },
    },

    "lgbm_classifier": {
        "model_class": LGBMClassifier,
        "problem_type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": -1},
        "hyperparameters": {
            "n_estimators":  {"type": "int",   "min": 50,    "max": 500,  "default": 100,  "description": "Number of boosting rounds"},
            "max_depth":     {"type": "int",   "min": 2,     "max": 15,   "default": -1,   "description": "Max depth (-1 = unlimited)"},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.5,  "default": 0.1,  "description": "Step size shrinkage"},
            "num_leaves":    {"type": "int",   "min": 20,    "max": 300,  "default": 31,   "description": "Max number of leaves per tree"},
        },
        "tune_space": lambda t: {
            "n_estimators":  t.suggest_int("n_estimators", 50, 500),
            "max_depth":     t.suggest_int("max_depth", 2, 15),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "num_leaves":    t.suggest_int("num_leaves", 20, 300),
        },
    },

    "svm_classifier": {
        "model_class": SVC,
        "problem_type": "classification",
        "default_params": {"probability": True, "random_state": 42},
        "hyperparameters": {
            "C":      {"type": "float", "min": 0.01, "max": 100.0, "default": 1.0,  "description": "Regularization parameter"},
            "kernel": {"type": "cat",   "choices": ["rbf", "linear", "poly"],       "default": "rbf", "description": "Kernel function"},
            "gamma":  {"type": "cat",   "choices": ["scale", "auto"],               "default": "scale", "description": "Kernel coefficient"},
        },
        "tune_space": lambda t: {
            "C":      t.suggest_float("C", 0.01, 100.0, log=True),
            "kernel": t.suggest_categorical("kernel", ["rbf", "linear"]),
            "gamma":  t.suggest_categorical("gamma", ["scale", "auto"]),
        },
    },

    # ── Regression ──────────────────────────────────────────────────────────
    "random_forest_regressor": {
        "model_class": RandomForestRegressor,
        "problem_type": "regression",
        "default_params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        "hyperparameters": {
            "n_estimators":      {"type": "int",   "min": 50,  "max": 500, "default": 100,  "description": "Number of trees"},
            "max_depth":         {"type": "int",   "min": 2,   "max": 30,  "default": None, "description": "Max depth of each tree"},
            "min_samples_split": {"type": "int",   "min": 2,   "max": 20,  "default": 2,    "description": "Min samples to split a node"},
            "min_samples_leaf":  {"type": "int",   "min": 1,   "max": 10,  "default": 1,    "description": "Min samples in a leaf node"},
        },
        "tune_space": lambda t: {
            "n_estimators":      t.suggest_int("n_estimators", 50, 500),
            "max_depth":         t.suggest_int("max_depth", 2, 30),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  t.suggest_int("min_samples_leaf", 1, 10),
        },
    },

    "linear_regression": {
        "model_class": LinearRegression,
        "problem_type": "regression",
        "default_params": {},
        "hyperparameters": {
            "fit_intercept": {"type": "cat", "choices": [True, False], "default": True, "description": "Whether to fit an intercept"},
        },
        "tune_space": lambda t: {
            "fit_intercept": t.suggest_categorical("fit_intercept", [True, False]),
        },
    },

    "ridge_regression": {
        "model_class": Ridge,
        "problem_type": "regression",
        "default_params": {"alpha": 1.0},
        "hyperparameters": {
            "alpha": {"type": "float", "min": 0.001, "max": 100.0, "default": 1.0, "description": "Regularization strength"},
        },
        "tune_space": lambda t: {
            "alpha": t.suggest_float("alpha", 0.001, 100.0, log=True),
        },
    },

    "lasso_regression": {
        "model_class": Lasso,
        "problem_type": "regression",
        "default_params": {"alpha": 1.0, "max_iter": 10000},
        "hyperparameters": {
            "alpha": {"type": "float", "min": 0.001, "max": 100.0, "default": 1.0, "description": "Regularization strength (zeros out weak features)"},
        },
        "tune_space": lambda t: {
            "alpha": t.suggest_float("alpha", 0.001, 100.0, log=True),
        },
    },

    "gradient_boosting_regressor": {
        "model_class": GradientBoostingRegressor,
        "problem_type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "hyperparameters": {
            "n_estimators":  {"type": "int",   "min": 50,    "max": 500, "default": 100, "description": "Number of boosting stages"},
            "max_depth":     {"type": "int",   "min": 2,     "max": 10,  "default": 3,   "description": "Max depth of each tree"},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.5, "default": 0.1, "description": "Shrinks contribution of each tree"},
            "subsample":     {"type": "float", "min": 0.5,   "max": 1.0, "default": 1.0, "description": "Fraction of samples per tree"},
        },
        "tune_space": lambda t: {
            "n_estimators":  t.suggest_int("n_estimators", 50, 500),
            "max_depth":     t.suggest_int("max_depth", 2, 10),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "subsample":     t.suggest_float("subsample", 0.5, 1.0),
        },
    },

    "xgboost_regressor": {
        "model_class": XGBRegressor,
        "problem_type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": 0},
        "hyperparameters": {
            "n_estimators":  {"type": "int",   "min": 50,    "max": 500, "default": 100, "description": "Number of boosting rounds"},
            "max_depth":     {"type": "int",   "min": 2,     "max": 12,  "default": 6,   "description": "Max depth of each tree"},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.5, "default": 0.1, "description": "Step size shrinkage"},
            "subsample":     {"type": "float", "min": 0.5,   "max": 1.0, "default": 1.0, "description": "Fraction of samples per tree"},
        },
        "tune_space": lambda t: {
            "n_estimators":  t.suggest_int("n_estimators", 50, 500),
            "max_depth":     t.suggest_int("max_depth", 2, 12),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "subsample":     t.suggest_float("subsample", 0.5, 1.0),
        },
    },

    "lgbm_regressor": {
        "model_class": LGBMRegressor,
        "problem_type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": -1},
        "hyperparameters": {
            "n_estimators":  {"type": "int",   "min": 50,    "max": 500, "default": 100, "description": "Number of boosting rounds"},
            "max_depth":     {"type": "int",   "min": 2,     "max": 15,  "default": -1,  "description": "Max depth (-1 = unlimited)"},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.5, "default": 0.1, "description": "Step size shrinkage"},
            "num_leaves":    {"type": "int",   "min": 20,    "max": 300, "default": 31,  "description": "Max number of leaves per tree"},
        },
        "tune_space": lambda t: {
            "n_estimators":  t.suggest_int("n_estimators", 50, 500),
            "max_depth":     t.suggest_int("max_depth", 2, 15),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "num_leaves":    t.suggest_int("num_leaves", 20, 300),
        },
    },

    "svr": {
        "model_class": SVR,
        "problem_type": "regression",
        "default_params": {},
        "hyperparameters": {
            "C":      {"type": "float", "min": 0.01, "max": 100.0, "default": 1.0,   "description": "Regularization parameter"},
            "kernel": {"type": "cat",   "choices": ["rbf", "linear", "poly"],        "default": "rbf", "description": "Kernel function"},
            "epsilon": {"type": "float", "min": 0.01, "max": 1.0,  "default": 0.1,   "description": "Insensitive loss tube width"},
        },
        "tune_space": lambda t: {
            "C":       t.suggest_float("C", 0.01, 100.0, log=True),
            "kernel":  t.suggest_categorical("kernel", ["rbf", "linear"]),
            "epsilon": t.suggest_float("epsilon", 0.01, 1.0),
        },
    },
}


# Default algorithm per problem type when user says "auto"
AUTO_DEFAULTS = {
    "binary_classification":   "xgboost_classifier",
    "multiclass_classification": "xgboost_classifier",
    "regression":              "xgboost_regressor",
}


def get_algorithm(name: str, problem_type: str = None):
    """Resolve algorithm name, handling 'auto' and validating problem type compatibility."""
    if name == "auto":
        if problem_type is None:
            raise ValueError("problem_type required when algorithm='auto'")
        name = AUTO_DEFAULTS[problem_type]

    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {list(ALGORITHM_REGISTRY.keys())}")

    entry = ALGORITHM_REGISTRY[name]
    if problem_type and entry["problem_type"] != "both":
        expected = "classification" if "classification" in problem_type else "regression"
        if entry["problem_type"] != expected:
            raise ValueError(
                f"Algorithm '{name}' is for {entry['problem_type']} but problem is {problem_type}"
            )
    return entry


def list_algorithms(problem_type: str = None) -> list:
    """Return available algorithm names, optionally filtered by problem type."""
    if problem_type is None:
        return list(ALGORITHM_REGISTRY.keys())
    kind = "classification" if "classification" in problem_type else "regression"
    return [k for k, v in ALGORITHM_REGISTRY.items() if v["problem_type"] in (kind, "both")]
