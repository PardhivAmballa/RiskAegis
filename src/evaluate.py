"""Held-out evaluation for saved RiskAegis models.

Inputs: data/processed/X_test.npz and y_test.npy plus models/*.joblib.
Outputs: reports/model_comparison.{csv,json} and per-model classification,
confusion-matrix, and plot artifacts under reports/evaluation/.
TARGET=1 metrics are highlighted as payment-difficulty recall/precision.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42


def evaluate_models() -> dict[str, object]:
    """Evaluate every saved model on the exact same 61,503-row test split.

    Baseline models read raw 307-feature X_test; PCA models read
    X_test_pca.npy (61,503 x 50). No training or refitting occurs here.
    """
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    X_test = sparse.load_npz(PROCESSED_DIR / "X_test.npz").tocsr()
    X_test_pca = (
        np.load(PROCESSED_DIR / "X_test_pca.npy")
        if (PROCESSED_DIR / "X_test_pca.npy").exists()
        else None
    )

    names = [
        "decision_tree",
        "random_forest",
        "adaboost",
        "gradient_boosting",
        "xgboost",
    ]
    artifacts = [
        (name, MODELS_DIR / f"{name}.joblib", X_test)
        for name in names
    ]
    artifacts.extend(
        [
            (f"{name}_pca", MODELS_DIR / f"{name}_pca.joblib", X_test_pca)
            for name in names
            if X_test_pca is not None
            and (MODELS_DIR / f"{name}_pca.joblib").exists()
        ]
    )

    results: dict[str, object] = {}
    reports_dir = REPORTS_DIR / "evaluation"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for name, artifact, test_matrix in artifacts:
        if not artifact.exists():
            raise FileNotFoundError(f"Saved model missing: {artifact}")
        model = joblib.load(artifact)
        y_pred = model.predict(test_matrix)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(test_matrix)[:, 1]
        else:
            y_score = y_pred

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["0 - no difficulty", "1 - payment difficulty"],
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "default_class_1_recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
            "pr_auc": float(average_precision_score(y_test, y_score)),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        results[name] = metrics

        pd.DataFrame(report).transpose().to_csv(
            reports_dir / f"{name}_classification_report.csv"
        )
        np.save(reports_dir / f"{name}_confusion_matrix.npy", cm)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["No difficulty", "Difficulty"])
        ax.set_yticks([0, 1], ["No difficulty", "Difficulty"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, cm[row, col], ha="center", va="center")
        fig.tight_layout()
        fig.savefig(reports_dir / f"{name}_confusion_matrix.png", dpi=150)
        plt.close(fig)

    with (REPORTS_DIR / "model_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    comparison_rows = [
        {"model": name, **{key: value for key, value in metrics.items()
                           if key != "classification_report" and key != "confusion_matrix"}}
        for name, metrics in results.items()
    ]
    pd.DataFrame(comparison_rows).to_csv(
        REPORTS_DIR / "model_comparison.csv", index=False
    )
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate_models(), indent=2))