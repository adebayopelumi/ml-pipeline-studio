import pandas as pd


def validate(df: pd.DataFrame, target_column: str = "target") -> pd.DataFrame:
    """
    Dynamically validate a cleaned DataFrame without hardcoded column names.
    Rules are inferred from the data's own structure.
    """
    if df.empty:
        raise ValueError("Cannot validate an empty DataFrame.")

    errors = []

    # Target column must exist
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {df.columns.tolist()}")

    from src.core.problem_detector import detect_problem_type
    problem_type = detect_problem_type(df[target_column])
    # Only enforce 0/1 check for numeric binary targets — string binary targets
    # (e.g. 'AllPub'/'NoSeWa') are valid and will be encoded by LabelEncoder
    if problem_type == "binary_classification" and pd.api.types.is_numeric_dtype(df[target_column]):
        target_values = set(df[target_column].dropna().unique())
        if not target_values.issubset({0, 1}):
            errors.append(f"Target '{target_column}' is numeric binary but has unexpected values: {target_values}")

    feature_cols = [c for c in df.columns if c != target_column]

    # No column should be entirely null
    all_null = [c for c in df.columns if df[c].isnull().all()]
    if all_null:
        errors.append(f"Columns are entirely null: {all_null}")

    # Numeric columns must have no nulls after cleaning
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    null_numeric = [c for c in numeric_cols if df[c].isnull().any()]
    if null_numeric:
        errors.append(f"Numeric columns still have nulls: {null_numeric}")

    # Silently drop any feature column that looks like an identifier (>90% unique values).
    # This is a safety net — clean_raw_data should have already removed these.
    categorical_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    id_like = [c for c in categorical_cols if df[c].nunique() > 0.9 * len(df)]
    if id_like:
        print(f"Validation: dropping identifier-like columns: {id_like}")
        df = df.drop(columns=id_like)

    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    tgt = df[target_column]
    if pd.api.types.is_numeric_dtype(tgt) and tgt.nunique() == 2:
        target_summary = f"positive rate={tgt.mean():.2%}"
    elif pd.api.types.is_numeric_dtype(tgt):
        target_summary = f"mean={tgt.mean():.4g}"
    else:
        target_summary = f"classes={tgt.nunique()}"
    print(
        f"Data validation passed. "
        f"{len(df)} rows | {len(numeric_cols)} numeric | {len(categorical_cols)} categorical | "
        f"target='{target_column}' {target_summary}"
    )
    return df


if __name__ == "__main__":
    from src.utils.config import load_config
    from src.data.load_data import load_raw_data, clean_raw_data

    config = load_config()
    df = load_raw_data(config["data"]["raw_path"])
    df = clean_raw_data(df)
    validate(df, target_column=config["data"]["target_column"])
