import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from src.core.problem_detector import detect_problem_type


def infer_features(df: pd.DataFrame, target_column: str):
    """Detect numeric and categorical feature columns from dtypes."""
    feature_cols = [c for c in df.columns if c != target_column]
    numeric = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Inferred {len(numeric)} numeric features: {numeric}")
    print(f"Inferred {len(categorical)} categorical features: {categorical}")
    return numeric, categorical


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def preprocess(df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    numeric_features, categorical_features = infer_features(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode classification targets so labels are always 0-indexed integers.
    # Applies to string targets (e.g. "Normal"→0) AND numeric targets whose
    # values aren't already [0, 1, ..., n-1] (e.g. [2,3,4,14] → [0,1,2,3]).
    label_encoder = None
    problem_type = detect_problem_type(y)
    if "classification" in problem_type:
        unique_sorted = sorted(y.dropna().unique().tolist())
        already_zero_indexed = unique_sorted == list(range(len(unique_sorted)))
        if not already_zero_indexed:
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            print(f"Target encoded: {dict(enumerate(label_encoder.classes_))}")

    # Only stratify when all classes have at least 2 members.
    # Rare classes (e.g. only 1 sample) make stratified splitting impossible.
    min_class_count = y.value_counts().min()
    stratify = y if (y.nunique() <= 20 and min_class_count >= 2) else None
    if stratify is None and y.nunique() <= 20:
        print(f"Stratification skipped: minority class has only {min_class_count} sample(s).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Train size: {X_train_processed.shape}, Test size: {X_test_processed.shape}")
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, label_encoder


def save_preprocessor(preprocessor: ColumnTransformer, path: str = "models/preprocessor.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")


def load_preprocessor(path: str = "models/preprocessor.joblib") -> ColumnTransformer:
    return joblib.load(path)
