import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


churn_schema = DataFrameSchema(
    {
        "tenure": Column(int, Check.greater_than_or_equal_to(0)),
        "monthly_charges": Column(float, Check.greater_than(0)),
        "total_charges": Column(float, Check.greater_than_or_equal_to(0)),
        "contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "churn": Column(int, Check.isin([0, 1])),
    },
    coerce=True,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Cannot validate an empty DataFrame.")
    try:
        validated = churn_schema.validate(df)
        print("Data validation passed.")
        return validated
    except pa.errors.SchemaError as e:
        raise ValueError(f"Data validation failed: {e}")


if __name__ == "__main__":
    from src.utils.config import load_config
    from src.data.load_data import load_raw_data

    config = load_config()
    df = load_raw_data(config["data"]["raw_path"])
    validate(df)
