import pandas as pd
from pathlib import Path


def load_raw_data(raw_path: str) -> pd.DataFrame:
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {raw_path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {raw_path}")
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop columns that are obviously identifiers — either by name or by having
    # nearly all unique values (e.g. Name, Ticket, Cabin in Titanic).
    # This runs before the target column is known, so we drop conservatively:
    # only string columns where > 90 % of non-null values are unique.
    id_name_cols = [c for c in df.columns if c.lower() in ("id", "customerid", "customer_id", "passengerid")]
    high_card_cols = [
        c for c in df.select_dtypes(include="object").columns
        if c not in id_name_cols and df[c].nunique() / max(len(df), 1) > 0.9
    ]
    drop_cols = list(set(id_name_cols + high_card_cols))
    if drop_cols:
        print(f"Auto-dropped identifier/high-cardinality columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Convert any object column that is actually numeric (e.g. TotalCharges stored as string)
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only replace if at least 80% of values converted cleanly
        if converted.notna().mean() >= 0.8:
            df[col] = converted

    # Fill missing numeric values with column median
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with the most frequent value
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Convert binary Yes/No columns to 1/0
    for col in df.select_dtypes(include="object").columns:
        unique_vals = set(df[col].dropna().str.strip().str.lower().unique())
        if unique_vals == {"yes", "no"}:
            df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)

    print(f"Cleaned data: {len(df)} rows, {len(df.columns)} columns.")
    return df
