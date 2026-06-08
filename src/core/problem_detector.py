import pandas as pd


def detect_problem_type(y: pd.Series) -> str:
    """
    Auto-detect whether the target column is regression or classification.
    - Object/string target → classification
    - Numeric with ≤20 unique values → classification
    - Numeric with >20 unique values → regression
    """
    if y.dtype == object or str(y.dtype) == "category":
        unique = y.nunique()
        if unique == 2:
            return "binary_classification"
        return "multiclass_classification"

    unique = y.nunique()
    if unique == 2:
        return "binary_classification"
    if unique <= 20:
        return "multiclass_classification"
    return "regression"


def describe_problem(problem_type: str) -> str:
    return {
        "binary_classification": "Binary Classification (two outcomes, e.g. yes/no, pass/fail)",
        "multiclass_classification": "Multiclass Classification (multiple categories)",
        "regression": "Regression (predicting a continuous number, e.g. house price)",
    }[problem_type]
