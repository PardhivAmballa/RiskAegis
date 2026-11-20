"""Baseline and PCA model training.

Inputs: data/processed/X_train.npz and y_train.npy.
Outputs: baseline/PCA joblib models and training JSON metadata under models/.
No test matrix or test target is read by this module during fitting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
AVAILABLE_MODELS = (
    "decision_tree",
    "random_forest",
    "adaboost",
    "gradient_boosting",
    "xgboost",
)


def load_training_data() -> tuple[sparse.csr_matrix, np.ndarray]:
    """Load and validate only the training split.

    Returns: CSR shape (246008, 307) full-data matrix and 1-D binary target.
    The test split is intentionally never loaded or fitted here.
    """
    X_path = PROCESSED_DIR / "X_train.npz"
    y_path = PROCESSED_DIR / "y_train.npy"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Missing processed training artifacts. Run preprocessing.py first."
        )

    X_train = sparse.load_npz(X_path).tocsr()
    y_train = np.load(y_path)
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train contain different row counts")
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training target must contain binary values 0 and 1")
    if np.isnan(X_train.data).any():
        raise ValueError("Training matrix contains missing values")
    return X_train, y_train


def build_baseline_models(y_train: np.ndarray) -> dict[str, Any]:
    """Create fixed-seed classifiers.

    Decision Tree/Random Forest use balanced weights; Gradient Boosting uses
    inverse-frequency sample weights; XGBoost uses scale_pos_weight; AdaBoost
    uses its boosting-round adaptive sample weighting.
    """
    negative_count = int(np.sum(y_train == 0))
    positive_count = int(np.sum(y_train == 1))
    scale_pos_weight = negative_count / positive_count

    return {
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=20,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def fit_model(model: Any, name: str, X: sparse.csr_matrix | np.ndarray, y: np.ndarray) -> None:
    """Fit with class balancing for models that support sample weights."""
    if name == "gradient_boosting":
        negative = int(np.sum(y == 0))
        positive = int(np.sum(y == 1))
        weights = np.where(y == 1, negative / positive, 1.0)
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)


def fit_sparse_pca(X_train: sparse.csr_matrix, n_components: int = 50) -> PCA:
    """Fit randomized 50-component PCA using only 50,000 training rows.

    The test matrix is not used for fitting; batch transformation limits peak RAM.
    """
    fit_size = min(50_000, X_train.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    fit_indices = rng.choice(X_train.shape[0], size=fit_size, replace=False)
    X_fit = X_train[fit_indices].toarray()
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    )
    pca.fit(X_fit)
    joblib.dump(pca, MODELS_DIR / "pca.joblib")
    return pca


def transform_in_batches(pca: PCA, matrix: sparse.csr_matrix, batch_size: int = 10_000) -> np.ndarray:
    """Transform sparse data in chunks to reduce peak RAM use."""
    chunks = [
        pca.transform(matrix[start:start + batch_size].toarray())
        for start in range(0, matrix.shape[0], batch_size)
    ]
    return np.vstack(chunks)


def train_pca_models(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    model_names: list[str] | None = None,
) -> dict[str, object]:
    """Train the five baselines with 50 PCA components, using training-only PCA fitting."""
    X_test = sparse.load_npz(PROCESSED_DIR / "X_test.npz").tocsr()
    pca = fit_sparse_pca(X_train)
    X_train_pca = transform_in_batches(pca, X_train)
    X_test_pca = transform_in_batches(pca, X_test)

    np.save(PROCESSED_DIR / "X_train_pca.npy", X_train_pca)
    np.save(PROCESSED_DIR / "X_test_pca.npy", X_test_pca)

    results: dict[str, object] = {
        "method": "PCA randomized",
        "n_components": int(pca.n_components),
        "random_state": RANDOM_STATE,
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "training_fit_size": int(min(50_000, X_train.shape[0])),
        "models": {},
    }
    pca_metadata_path = MODELS_DIR / "pca_training_metadata.json"
    if pca_metadata_path.exists():
        with pca_metadata_path.open("r", encoding="utf-8") as handle:
            existing_pca = json.load(handle)
        results["models"] = existing_pca.get("models", {})

    models = build_baseline_models(y_train)
    requested = model_names or list(models)
    for name in requested:
        model = models[name]
        start = time.perf_counter()
        fit_model(model, name, X_train_pca, y_train)
        duration = time.perf_counter() - start
        model_path = MODELS_DIR / f"{name}_pca.joblib"
        joblib.dump(model, model_path)
        joblib.load(model_path)
        results["models"][name] = {
            "status": "trained_and_loaded",
            "duration_seconds": round(duration, 3),
            "artifact": str(model_path),
        }
        print(f"{name}_pca: trained and loaded in {duration:.3f} seconds -> {model_path}")

    with (MODELS_DIR / "pca_training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def train_models(model_names: tuple[str, ...] | list[str] | None = None) -> dict[str, object]:
    """Train baseline models on the saved Phase 1 training matrix."""
    X_train, y_train = load_training_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = build_baseline_models(y_train)
    requested = list(model_names or AVAILABLE_MODELS)
    unknown = sorted(set(requested).difference(AVAILABLE_MODELS))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")

    results: dict[str, object] = {
        "random_state": RANDOM_STATE,
        "data_source": "data/processed/X_train.npz",
        "train_rows": int(X_train.shape[0]),
        "train_columns": int(X_train.shape[1]),
        "train_class_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y_train, return_counts=True))
        },
        "models": {},
    }
    if (MODELS_DIR / "training_metadata.json").exists():
        with (MODELS_DIR / "training_metadata.json").open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        results["models"] = existing.get("models", {})
        results["pca"] = existing.get("pca", {})

    for name in requested:
        model = models[name]
        start = time.perf_counter()
        fit_model(model, name, X_train, y_train)
        duration_seconds = time.perf_counter() - start

        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)
        if not hasattr(loaded, "predict"):
            raise RuntimeError(f"Saved model cannot be loaded for {name}")

        results["models"][name] = {
            "status": "trained_and_loaded",
            "duration_seconds": round(duration_seconds, 3),
            "artifact": str(model_path),
            "parameters": model.get_params(),
        }
        print(
            f"{name}: trained and loaded in {duration_seconds:.3f} seconds "
            f"-> {model_path}"
        )

    metadata_path = MODELS_DIR / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    pca_results = train_pca_models(X_train, y_train, requested)
    results["pca"] = pca_results
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    return results


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline Home Credit classifiers"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(AVAILABLE_MODELS),
        default=None,
        help="Optional subset of baseline models",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    print(json.dumps(train_models(args.models), indent=2, default=str))