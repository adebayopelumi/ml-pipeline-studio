import pytest
import pandas as pd
from src.data.validate_data import validate


def make_valid_df(n=5):
    return pd.DataFrame({
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "Yes"],
        "tenure": [12, 34, 2, 60, 1],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No", "No internet service", "Yes", "No"],
        "OnlineBackup": ["No", "Yes", "No internet service", "No", "Yes"],
        "DeviceProtection": ["Yes", "No", "No internet service", "Yes", "No"],
        "TechSupport": ["No", "Yes", "No internet service", "No", "Yes"],
        "StreamingTV": ["Yes", "No", "No internet service", "Yes", "No"],
        "StreamingMovies": ["No", "Yes", "No internet service", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check"
        ],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
        "TotalCharges": [29.85, 1889.5, 108.15, 1840.75, 151.65],
        "Churn": [0, 0, 1, 0, 1],
    })


def test_valid_data_passes():
    df = make_valid_df()
    result = validate(df, target_column="Churn")
    assert len(result) == 5


def test_missing_target_raises():
    df = make_valid_df().drop(columns=["Churn"])
    with pytest.raises(ValueError, match="Target column"):
        validate(df, target_column="Churn")


def test_non_binary_target_valid_as_multiclass():
    # A target with values 0,1,2 is valid multiclass — should NOT raise
    df = make_valid_df()
    df.loc[0, "Churn"] = 2
    result = validate(df, target_column="Churn")
    assert len(result) == 5


def test_all_null_column_raises():
    df = make_valid_df()
    df["tenure"] = None
    with pytest.raises(ValueError, match="entirely null"):
        validate(df, target_column="Churn")


def test_null_numeric_raises():
    df = make_valid_df()
    df.loc[0, "MonthlyCharges"] = None
    with pytest.raises(ValueError, match="nulls"):
        validate(df, target_column="Churn")


def test_id_like_column_dropped():
    df = make_valid_df(n=5)
    # Add a column where every value is unique — looks like an ID; should be silently dropped
    df["customerID"] = ["a", "b", "c", "d", "e"]
    result = validate(df, target_column="Churn")
    assert "customerID" not in result.columns


def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="empty"):
        validate(pd.DataFrame(), target_column="Churn")
