# RiskAegis
Intelligent Credit Risk Assessment &amp; Loan Default Prediction Engine with Model Benchmarking, SHAP Explainability, and Streamlit UI.

## Pipeline Handoff &amp; Artifacts

Run all commands from the project root with the project virtual environment:

```powershell
python src/preprocessing.py
python src/train_models.py
python src/evaluate.py
```

- Preprocessing reads `data/raw/application_train.csv`, excludes `SK_ID_CURR`,
  separates `TARGET`, fits only on the stratified 80% training split, and writes
  sparse/NumPy outputs plus `preprocessor.joblib` to `data/processed/`.
- Training reads `data/processed/X_train.npz` and `y_train.npy`, saves baseline
  and 50-component PCA models as `*.joblib` in `models/`, and writes
  `training_metadata.json` and `pca_training_metadata.json`.
- Evaluation reads the saved test split and all matching model artifacts, then
  writes `reports/model_comparison.{csv,json}` and per-model reports in
  `reports/evaluation/`. The test split has 61,503 rows; class 1 is the
  payment-difficulty target.

Downstream ablation, explainability, error analysis, and Streamlit code should
read `data/processed/preprocessing_metadata.json` and
`data/processed/feature_metadata.json` for schemas, `models/*.joblib` for
models, and `reports/model_comparison.json` for held-out metrics. Use
`SK_ID_CURR` from `applicant_ids_*.npy` only for traceability, never as a model
feature.
