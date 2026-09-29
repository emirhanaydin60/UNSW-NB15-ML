# UNSW-NB15 Multiclass Intrusion Benchmark (Nested CV + GWO)

This project implements a full, reproducible experimental software system for controlled comparison of five machine learning classifiers on the UNSW-NB15 multiclass intrusion detection task.

## 1. Purpose

The benchmark compares:
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM, RBF)
- Logistic Regression (LR)
- XGBoost

All models share one experimental protocol:
- Official train/test partition
- Nested stratified CV (5 outer, 5 inner)
- Fold-local preprocessing
- Fold-local Random Forest feature selection (top 20)
- Fold-local SMOTENC balancing
- Grey Wolf Optimizer hyperparameter tuning
- Common fitness and metrics
- Paired statistical analysis

No model-selection signal is taken from the official test set.

## 2. Dataset Preparation

Expected CSV files:
- Training: 175,341 rows
- Official test: 82,332 rows

Configure paths in config.yaml:
- dataset.train_path
- dataset.test_path

Required fields include:
- target: attack_cat
- excluded predictors: id, label
- categorical predictors: proto, service, state

Validation checks enforce:
- required columns
- target presence
- missing values
- categorical column sanity
- class-label validation
- duplicate warning

## 3. Installation

1. Create/activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 4. Configuration

Edit config.yaml for:
- dataset paths
- seeds
- CV settings
- SMOTENC settings
- feature selection settings
- GWO settings
- search spaces
- logging/report settings

The effective run config is copied to:
- runs/<run_id>/config/config_used.yaml

## 5. Running the Experiment

Start a new full run:

```bash
python main.py --config config.yaml --run
```

Resume latest or specific run:

```bash
python main.py --config config.yaml --resume
python main.py --config config.yaml --resume --run-id <run_id>
```

Restart as a new run directory (does not overwrite old runs):

```bash
python main.py --config config.yaml --restart
```

Show run status:

```bash
python main.py --config config.yaml --status --run-id <run_id>
```

Generate tables/figures from saved results only:

```bash
python main.py --config config.yaml --generate-reports --run-id <run_id>
```

## 6. Checkpoint and Resume

Checkpointing is persisted under each run directory and includes:
- experiment state JSON
- per-GWO state files
- progress metadata (phase/model/fold/iteration/wolf)
- completed fold/model markers

Checkpoint writes are atomic (temporary file + rename).

Resume behavior:
- Loads checkpoint and saved run config
- Continues from last completed state
- Skips already completed model/fold/final stages

## 7. Smoke Test

Run a lightweight smoke execution on class-stratified subsets with smaller CV/GWO budgets:

```bash
python main.py --config config.yaml --run --smoke-test
```

Smoke runs are isolated via dedicated run IDs (suffix _smoke) and do not modify normal run checkpoints.

## 8. Output Structure

Each run:

- runs/<run_id>/config/
- runs/<run_id>/logs/
- runs/<run_id>/checkpoints/
- runs/<run_id>/results/
- runs/<run_id>/figures/
- runs/<run_id>/tables/
- runs/<run_id>/models/
- runs/<run_id>/metadata/

## 9. Core Result Files

Generated under results/:
- outer_fold_metrics.csv
- outer_fold_predictions.csv
- gwo_results.csv
- gwo_convergence.csv
- selected_features.csv
- feature_importances.csv
- class_distributions.csv
- final_hyperparameters.csv
- statistical_results.csv
- final_test_metrics.csv
- final_test_predictions.csv
- confusion_matrices/*.csv
- classification_report_<MODEL>.json

## 10. Figures and Tables

Figures (PNG + PDF, configurable DPI):
- class distribution before/after SMOTENC
- GWO convergence curves
- model comparison (Balanced Accuracy)
- model comparison (Macro-F1)
- confusion matrices per final model

Tables (CSV):
- cross-validation performance summary
- final hyperparameters
- selected final features
- Friedman test summary
- Wilcoxon pairwise comparisons with Holm correction
- final official test performance

## 11. Reproducibility

Reproducibility controls include:
- base seed (default 42)
- deterministic derived seeds by phase/model/fold
- saved effective config per run
- environment metadata (Python, package versions, OS, CPU)
- dataset filename and SHA-256 hashes

## 12. Data Leakage Safeguards

The implementation includes explicit safeguards:
- outer and inner split overlap assertions
- preprocessing fit only on training partitions
- feature selection fit only on outer training partitions
- SMOTENC applied only to training partitions
- outer validation never used by GWO or model fitting
- official test evaluation only after nested CV completion

## 13. Running Tests

```bash
pytest -q
```

Test coverage includes:
- dataset loading and validation
- preprocessing/categorical handling
- feature selection
- SMOTENC strategy and resampling
- GWO decoding and boundaries
- fitness calculation
- checkpoint save/load and resume path
- statistical analysis and aggregation
- smoke subset construction

## 14. Notes

- If a single candidate configuration fails during GWO, the run logs the error and assigns poor fitness while continuing optimization.
- Serious structural failures are logged, checkpointed, and raised safely.
- Report generation reads saved result files and does not rerun model training.
