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
