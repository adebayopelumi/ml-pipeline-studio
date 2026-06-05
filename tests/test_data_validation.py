import pytest
import pandas as pd
from src.data.validate_data import validate


def make_valid_df():
    return pd.DataFrame({
        "tenure": [12, 24, 6],
        "monthly_charges": [75.5, 50.0, 90.0],
        "total_charges": [900.0, 1200.0, 540.0],
        "contract": ["Month-to-month", "One year", "Two year"],
        "churn": [1, 0, 1],
    })


def test_valid_data_passes():
    df = make_valid_df()
    result = validate(df)
    assert len(result) == 3


def test_missing_column_raises():
    df = make_valid_df().drop(columns=["tenure"])
    with pytest.raises(ValueError):
        validate(df)


def test_invalid_contract_raises():
    df = make_valid_df()
    df.loc[0, "contract"] = "Weekly"
    with pytest.raises(ValueError):
        validate(df)


def test_negative_monthly_charges_raises():
    df = make_valid_df()
    df.loc[0, "monthly_charges"] = -10.0
    with pytest.raises(ValueError):
        validate(df)


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        validate(pd.DataFrame())
