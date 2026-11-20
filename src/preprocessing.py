"""Home Credit preprocessing.

Inputs: application_train.csv. Outputs: CSR matrices, NumPy targets/IDs,
the fitted ColumnTransformer, and JSON metadata under data/processed/.
All fitting is performed only on the 80% training split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42
EMPLOYMENT_PLACEHOLDER = 365243


def load_application_data(path: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load and validate the 122-column application table.

    Returns: DataFrame; nrows=None reads the full dataset.
    Guards: SK_ID_CURR must be present, unique, and non-missing; TARGET binary.
    """
    data_path = path or RAW_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Home Credit training file not found: {data_path}")

    data = pd.read_csv(data_path, nrows=nrows)
    required = {"SK_ID_CURR", "TARGET"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data["SK_ID_CURR"].isna().any():
        raise ValueError("SK_ID_CURR contains missing values")
    if data["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR contains duplicate applicant IDs")
    if set(data["TARGET"].dropna().unique()) - {0, 1}:
        raise ValueError("TARGET must contain only 0 and 1")
    if data["TARGET"].isna().any():
        raise ValueError("TARGET contains missing values")
    return data


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate IDs/target and return model features.

    SK_ID_CURR and TARGET are never model inputs. DAYS_EMPLOYED=365243 is
    converted to missing and retained as DAYS_EMPLOYED_IS_PLACEHOLDER.
    """
    applicant_ids = data["SK_ID_CURR"].copy()
    target = data["TARGET"].astype("int8").copy()
    features = data.drop(columns=["SK_ID_CURR", "TARGET"]).copy()

    if "DAYS_EMPLOYED" in features:
        features["DAYS_EMPLOYED_IS_PLACEHOLDER"] = (
            features["DAYS_EMPLOYED"] == EMPLOYMENT_PLACEHOLDER
        ).astype("int8")
        features.loc[
            features["DAYS_EMPLOYED"] == EMPLOYMENT_PLACEHOLDER,
            "DAYS_EMPLOYED",
        ] = np.nan

    return features, target, applicant_ids


def identify_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical model columns."""
    numerical = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number"]).columns.tolist()
    return numerical, categorical


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build training-fitted imputers, encoder, and numerical scaler."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def save_artifacts(
    preprocessor: ColumnTransformer,
    X_train: sparse.spmatrix,
    X_test: sparse.spmatrix,
    y_train: pd.Series,
    y_test: pd.Series,
    applicant_ids_train: pd.Series,
    applicant_ids_test: pd.Series,
    numerical_features: list[str],
    categorical_features: list[str],
    feature_names: list[str],
) -> None:
    """Persist reusable outputs.

    X_train/X_test are CSR float matrices; y and applicant arrays are 1-D NumPy
    arrays; preprocessor.joblib is fitted only on training data.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(PROCESSED_DIR / "X_train.npz", sparse.csr_matrix(X_train))
    sparse.save_npz(PROCESSED_DIR / "X_test.npz", sparse.csr_matrix(X_test))
    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())
    np.save(PROCESSED_DIR / "applicant_ids_train.npy", applicant_ids_train.to_numpy())
    np.save(PROCESSED_DIR / "applicant_ids_test.npy", applicant_ids_test.to_numpy())
    joblib.dump(preprocessor, PROCESSED_DIR / "preprocessor.joblib")

    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "target": "TARGET",
        "target_definition": (
            "1 indicates payment difficulties in early installments; "
            "not necessarily a confirmed charge-off"
        ),
        "applicant_id": "SK_ID_CURR",
        "employment_placeholder": EMPLOYMENT_PLACEHOLDER,
        "employment_placeholder_indicator": "DAYS_EMPLOYED_IS_PLACEHOLDER",
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "transformed_feature_names": feature_names,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "transformed_columns": int(X_train.shape[1]),
    }
    with (PROCESSED_DIR / "preprocessing_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    feature_metadata = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "feature_names": feature_names,
        "excluded_features": ["SK_ID_CURR", "TARGET"],
    }
    with (PROCESSED_DIR / "feature_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(feature_metadata, handle, indent=2)


def preprocess_application_data(
    nrows: int | None = None,
) -> dict[str, object]:
    """Run Home Credit preprocessing and persist all reusable outputs."""
    data = load_application_data(nrows=nrows)
    features, target, applicant_ids = prepare_features(data)
    numerical_features, categorical_features = identify_feature_types(features)

    (
        features_train,
        features_test,
        y_train,
        y_test,
        applicant_ids_train,
        applicant_ids_test,
    ) = train_test_split(
        features,
        target,
        applicant_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_train = preprocessor.fit_transform(features_train)
    X_test = preprocessor.transform(features_test)
    if sparse.issparse(X_train):
        X_train = sparse.csr_matrix(X_train)
    if sparse.issparse(X_test):
        X_test = sparse.csr_matrix(X_test)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names = [
            f"{index}" for index in range(X_train.shape[1])
        ]

    save_artifacts(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
        applicant_ids_train,
        applicant_ids_test,
        numerical_features,
        categorical_features,
        feature_names,
    )

    return {
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "transformed_columns": X_train.shape[1],
        "numerical_features": len(numerical_features),
        "categorical_features": len(categorical_features),
        "train_target_counts": y_train.value_counts().sort_index().to_dict(),
        "test_target_counts": y_test.value_counts().sort_index().to_dict(),
        "output_dir": str(PROCESSED_DIR),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Home Credit application data")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row count for a smoke test",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(preprocess_application_data(args.sample_size), indent=2))