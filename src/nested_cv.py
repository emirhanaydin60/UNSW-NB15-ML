from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.balancing import apply_smotenc
from src.checkpoint import CheckpointManager
from src.data_loader import load_test_dataset, split_xy
from src.evaluation import (
    classification_report_dict,
    compute_balanced_accuracy_macro_f1,
    confusion_matrix_df,
)
from src.feature_selection import select_top_features
from src.gwo import GreyWolfOptimizer
from src.models import MODEL_ORDER, build_model, model_requires_scaling
from src.preprocessing import FoldPreprocessor, ModelTransformer, get_selected_categorical_columns
from src.reporting import generate_reports
from src.statistics import run_statistical_analysis
from src.utils import (
    append_csv_rows,
    atomic_write_json,
    derive_seed,
    ensure_dir,
    get_system_metadata,
    json_hash,
    upsert_csv_rows,
    utc_now_iso,
)


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    smoke_test: bool


class NestedCVExperiment:
    def __init__(
        self,
        config: dict[str, Any],
        run_context: RunContext,
        logger: Any,
        checkpoint_manager: CheckpointManager,
    ):
        self.config = config
        self.ctx = run_context
        self.logger = logger
        self.ckpt = checkpoint_manager

        self.dataset_cfg = self.config["dataset"]
        self.base_seed = int(self.config["experiment"]["base_seed"])
        self.model_names = list(self.config["models"]["enabled"])

        self.expected_classes = list(self.dataset_cfg["expected_classes"])
        self.class_to_idx = {c: i for i, c in enumerate(self.expected_classes)}

        self.paths = self._init_paths()
        self.state = self._load_or_init_state()

        self._seen_gwo_eval_keys: set[str] = set()
        self._seen_gwo_conv_keys: set[str] = set()
        self._bootstrap_seen_gwo_keys()

    def _init_paths(self) -> dict[str, Path]:
        run_dir = self.ctx.run_dir
        paths = {
            "run": run_dir,
            "config": ensure_dir(run_dir / "config"),
            "logs": ensure_dir(run_dir / "logs"),
            "checkpoints": ensure_dir(run_dir / "checkpoints"),
            "results": ensure_dir(run_dir / "results"),
            "figures": ensure_dir(run_dir / "figures"),
            "tables": ensure_dir(run_dir / "tables"),
            "models": ensure_dir(run_dir / "models"),
            "metadata": ensure_dir(run_dir / "metadata"),
        }
        ensure_dir(paths["results"] / "confusion_matrices")
        return paths

    def _load_or_init_state(self) -> dict[str, Any]:
        # Validate checkpoint compatibility with the current methodology fingerprint
        expected_fingerprint = {
            "outer_folds": int(self.config["cv"]["outer_folds"]),
            "hpo_validation_size": float(self.config.get("hpo", {}).get("validation_size", 0.20)),
            "gwo_population": int(self.config.get("gwo", {}).get("population_size", 5)),
            "gwo_iterations": int(self.config.get("gwo", {}).get("iterations", 10)),
            "inner_cv": False,
        }

        existing = self.ckpt.load_state()
        if existing is not None:
            try:
                self.ckpt.validate_compatibility(expected_fingerprint)
            except Exception as exc:
                raise RuntimeError(f"Checkpoint compatibility failure: {exc}")
            self.logger.info("Loaded existing experiment checkpoint state")
            return existing

        state = {
            "run_id": self.ctx.run_id,
            "created_at": utc_now_iso(),
            "smoke_test": self.ctx.smoke_test,
            "config_hash": json_hash(self.config),
            "methodology_fingerprint": expected_fingerprint,
            "completed_outer_models": [],
            "completed_outer_folds": [],
            "outer_feature_selection_done": [],
            "completed_final_models": [],
            "outer_cv_complete": False,
            "final_training_complete": False,
            "reports_generated": False,
            "official_test_used": False,
            "active_progress": {
                "phase": None,
                "model": None,
                "outer_fold": None,
                "inner_fold": None,
                "gwo_iteration": None,
                "wolf_index": None,
            },
            "last_error": None,
        }
        self.ckpt.save_state(state)
        return state

    def _bootstrap_seen_gwo_keys(self) -> None:
        eval_path = self.paths["results"] / "gwo_results.csv"
        if eval_path.exists():
            df = pd.read_csv(eval_path)
            for _, row in df.iterrows():
                key = self._gwo_eval_key(
                    str(row["phase"]),
                    str(row["model"]),
                    int(row["outer_fold"]),
                    int(row["iteration"]),
                    int(row["wolf_index"]),
                )
                self._seen_gwo_eval_keys.add(key)

        conv_path = self.paths["results"] / "gwo_convergence.csv"
        if conv_path.exists():
            df = pd.read_csv(conv_path)
            for _, row in df.iterrows():
                key = self._gwo_conv_key(
                    str(row["phase"]),
                    str(row["model"]),
                    int(row["outer_fold"]),
                    int(row["iteration"]),
                )
                self._seen_gwo_conv_keys.add(key)

    def _save_state(self) -> None:
        self.ckpt.save_state(self.state)

    def _mark_error(self, error: str) -> None:
        self.state["last_error"] = {"timestamp": utc_now_iso(), "error": error}
        self._save_state()

    def write_metadata(self, dataset_hashes: dict[str, str]) -> None:
        import importlib.metadata

        package_versions = {}
        for pkg in [
            "numpy",
            "pandas",
            "scikit-learn",
            "imbalanced-learn",
            "xgboost",
            "scipy",
            "matplotlib",
            "seaborn",
            "pyyaml",
            "joblib",
        ]:
            try:
                package_versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                package_versions[pkg] = "not-installed"

        metadata = {
            "run_id": self.ctx.run_id,
            "created_at": self.state["created_at"],
            "smoke_test": self.ctx.smoke_test,
            "system": get_system_metadata(),
            "base_seed": self.base_seed,
            "dataset": dataset_hashes,
            "package_versions": package_versions,
            "config_hash": self.state["config_hash"],
        }
        atomic_write_json(self.paths["metadata"] / "run_metadata.json", metadata)

    def _encode_y(self, y: pd.Series) -> np.ndarray:
        unknown = sorted(set(y.astype(str).unique()) - set(self.expected_classes))
        if unknown:
            raise ValueError(f"Unknown labels encountered: {unknown}")
        return y.astype(str).map(self.class_to_idx).to_numpy()

    def _decode_y(self, y_encoded: np.ndarray) -> list[str]:
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        return [idx_to_class[int(i)] for i in y_encoded]

    def _outer_model_key(self, model: str, outer_fold: int) -> dict[str, Any]:
        return {"model": model, "outer_fold": int(outer_fold)}

    def _is_outer_model_completed(self, model: str, outer_fold: int) -> bool:
        key = self._outer_model_key(model, outer_fold)
        return key in self.state["completed_outer_models"]

    def _mark_outer_model_completed(self, model: str, outer_fold: int) -> None:
        key = self._outer_model_key(model, outer_fold)
        if key not in self.state["completed_outer_models"]:
            self.state["completed_outer_models"].append(key)
            self._save_state()

    def _is_final_model_completed(self, model: str) -> bool:
        return model in self.state["completed_final_models"]

    def _mark_final_model_completed(self, model: str) -> None:
        if model not in self.state["completed_final_models"]:
            self.state["completed_final_models"].append(model)
            self._save_state()

    def _append_outer_fold_feature_rows(self, rank_df: pd.DataFrame, outer_fold: int, phase: str) -> None:
        feature_rows = []
        selected_rows = []
        for _, row in rank_df.iterrows():
            feature_rows.append(
                {
                    "phase": phase,
                    "outer_fold": outer_fold,
                    "feature": row["feature"],
                    "importance": float(row["importance"]),
                    "rank": int(row["rank"]),
                    "selected": bool(row["selected"]),
                }
            )
            if bool(row["selected"]):
                selected_rows.append(
                    {
                        "phase": phase,
                        "outer_fold": outer_fold,
                        "rank": int(row["rank"]),
                        "feature": row["feature"],
                        "importance": float(row["importance"]),
                    }
                )

        upsert_csv_rows(
            self.paths["results"] / "feature_importances.csv",
            feature_rows,
            fieldnames=["phase", "outer_fold", "feature", "importance", "rank", "selected"],
            key_fields=["phase", "outer_fold", "feature"],
        )
        upsert_csv_rows(
            self.paths["results"] / "selected_features.csv",
            selected_rows,
            fieldnames=["phase", "outer_fold", "rank", "feature", "importance"],
            key_fields=["phase", "outer_fold", "rank", "feature"],
        )

    def _emit_runtime_event(self, event: dict[str, Any]) -> None:
        """Append a runtime event record to metadata/runtime_events.jsonl"""
        path = self.paths["metadata"] / "runtime_events.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True) + "\n")
        except Exception:
            # best-effort only; do not fail pipeline for logging issues
            return

    def _record_class_distribution(
        self,
        phase: str,
        model: str,
        outer_fold: int,
        before: dict[str, int],
        after: dict[str, int],
    ) -> None:
        rows = []
        classes = sorted(set(before) | set(after))
        for cls in classes:
            rows.append(
                {
                    "phase": phase,
                    "model": model,
                    "outer_fold": outer_fold,
                    "stage": "before",
                    "class": cls,
                    "count": int(before.get(cls, 0)),
                }
            )
            rows.append(
                {
                    "phase": phase,
                    "model": model,
                    "outer_fold": outer_fold,
                    "stage": "after",
                    "class": cls,
                    "count": int(after.get(cls, 0)),
                }
            )
        upsert_csv_rows(
            self.paths["results"] / "class_distributions.csv",
            rows,
            fieldnames=["phase", "model", "outer_fold", "stage", "class", "count"],
            key_fields=["phase", "model", "outer_fold", "stage", "class"],
        )

    def _gwo_eval_key(self, phase: str, model: str, outer_fold: int, iteration: int, wolf_index: int) -> str:
        return f"{phase}|{model}|{outer_fold}|{iteration}|{wolf_index}"

    def _gwo_conv_key(self, phase: str, model: str, outer_fold: int, iteration: int) -> str:
        return f"{phase}|{model}|{outer_fold}|{iteration}"

    def _persist_gwo_result_rows(
        self,
        phase: str,
        model: str,
        outer_fold: int,
        evaluations: list[dict[str, Any]],
        convergence: list[dict[str, Any]],
    ) -> None:
        eval_rows = []
        for row in evaluations:
            k = self._gwo_eval_key(phase, model, outer_fold, int(row["iteration"]), int(row["wolf_index"]))
            if k in self._seen_gwo_eval_keys:
                continue
            self._seen_gwo_eval_keys.add(k)
            eval_rows.append(
                {
                    "phase": phase,
                    "model": model,
                    "outer_fold": outer_fold,
                    "iteration": int(row["iteration"]),
                    "wolf_index": int(row["wolf_index"]),
                    "hyperparameters_json": json.dumps(row["params"], sort_keys=True),
                    "fitness": float(row["fitness"]),
                    "mean_inner_balanced_accuracy": float(row.get("mean_inner_balanced_accuracy", row.get("mean_balanced_accuracy", row["fitness"]))),
                    "mean_inner_macro_f1": float(row.get("mean_inner_macro_f1", row.get("mean_macro_f1", row["fitness"]))),
                    "best_fitness_so_far": float(row["best_fitness_so_far"]),
                    "became_alpha": bool(row["became_alpha"]),
                    "failed": bool(row.get("failed", False)),
                    "error": str(row.get("error", "")),
                }
            )

        if eval_rows:
            upsert_csv_rows(
                self.paths["results"] / "gwo_results.csv",
                eval_rows,
                fieldnames=[
                    "phase",
                    "model",
                    "outer_fold",
                    "iteration",
                    "wolf_index",
                    "hyperparameters_json",
                    "fitness",
                    "mean_inner_balanced_accuracy",
                    "mean_inner_macro_f1",
                    "best_fitness_so_far",
                    "became_alpha",
                    "failed",
                    "error",
                ],
                key_fields=["phase", "model", "outer_fold", "iteration", "wolf_index"],
            )

        conv_rows = []
        for row in convergence:
            k = self._gwo_conv_key(phase, model, outer_fold, int(row["iteration"]))
            if k in self._seen_gwo_conv_keys:
                continue
            self._seen_gwo_conv_keys.add(k)
            conv_rows.append(
                {
                    "phase": phase,
                    "model": model,
                    "outer_fold": outer_fold,
                    "iteration": int(row["iteration"]),
                    "best_fitness": float(row["best_fitness"]),
                    "alpha_score": float(row["alpha_score"]),
                    "beta_score": float(row["beta_score"]),
                    "delta_score": float(row["delta_score"]),
                }
            )

        if conv_rows:
            upsert_csv_rows(
                self.paths["results"] / "gwo_convergence.csv",
                conv_rows,
                fieldnames=[
                    "phase",
                    "model",
                    "outer_fold",
                    "iteration",
                    "best_fitness",
                    "alpha_score",
                    "beta_score",
                    "delta_score",
                ],
                key_fields=["phase", "model", "outer_fold", "iteration"],
            )

    def _run_gwo(
        self,
        phase: str,
        model_name: str,
        outer_fold: int,
        fitness_fn,
        model_seed: int,
    ) -> dict[str, Any]:
        search_space = self.config["models"]["search_spaces"][model_name]
        gwo_cfg = self.config["gwo"]
        optimizer = GreyWolfOptimizer(
            model_name=model_name,
            search_space_cfg=search_space,
            population_size=int(gwo_cfg["population_size"]),
            iterations=int(gwo_cfg["iterations"]),
            seed=model_seed,
            poor_fitness_value=-1.0,
        )

        gwo_key = f"{phase}_{model_name}_fold_{outer_fold}"
        resume_state = self.ckpt.load_gwo_state(gwo_key)

        def checkpoint_callback(event_type: str, state: dict[str, Any], _row: dict[str, Any] | None) -> None:
            self.state["active_progress"] = {
                "phase": phase,
                "model": model_name,
                "outer_fold": outer_fold,
                "inner_fold": None,
                "gwo_iteration": int(state.get("iteration", 0)) + 1,
                "wolf_index": int(state.get("next_wolf_index", 0)),
                "event_type": event_type,
            }
            self.ckpt.save_gwo_state(gwo_key, state)
            self._save_state()

        result = optimizer.optimize(
            fitness_fn=fitness_fn,
            checkpoint_callback=checkpoint_callback,
            resume_state=resume_state,
            logger=self.logger,
        )

        self._persist_gwo_result_rows(
            phase=phase,
            model=model_name,
            outer_fold=outer_fold,
            evaluations=result["evaluations"],
            convergence=result["convergence_history"],
        )
        return result

    def _build_sampling_strategy_for_stage(self, y: pd.Series) -> dict[str, int]:
        target_count = int(self.config["smotenc"]["target_count"])
        return {str(cls): target_count for cls, count in y.value_counts().items() if int(count) < target_count}

    def _apply_balancing_with_policy(
        self,
        X_selected: pd.DataFrame,
        y_selected: pd.Series,
        categorical_indices: list[int],
        stage_random_state: int,
        log_context: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, dict[str, int], dict[str, int], dict[str, int], str]:
        before = y_selected.value_counts().sort_index().to_dict()
        strategy = self._build_sampling_strategy_for_stage(y_selected)
        if not strategy:
            return X_selected.copy(), y_selected.copy(), before, before.copy(), strategy, "none"

        # Build a diagnostic record and persist it to metadata/diagnostics JSONL
        diag_dir = self.paths["metadata"] / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag_file = diag_dir / "balancing_events.jsonl"

        # enrich log_context with selected-features info if present
        lc = dict(log_context or {})
        try:
            inner_cv_seed = None
            if lc.get("model") is not None and lc.get("outer_fold") is not None:
                inner_cv_seed = derive_seed(self.base_seed, lc.get("model"), int(lc.get("outer_fold")), "inner_cv")
        except Exception:
            inner_cv_seed = None

        base_diag = {
            "timestamp": utc_now_iso(),
            "model": lc.get("model"),
            "phase": lc.get("phase", "outer_cv"),
            "outer_fold": lc.get("outer_fold"),
            "inner_fold": lc.get("inner_fold"),
            "gwo_iteration": lc.get("gwo_iteration"),
            "wolf_index": lc.get("wolf_index"),
            "stage_random_state": int(stage_random_state) if stage_random_state is not None else None,
            "inner_cv_seed": inner_cv_seed,
            "k_neighbors": int(self.config["smotenc"]["k_neighbors"]),
            "target_count": int(self.config["smotenc"]["target_count"]),
            "y_counts_before": before,
            "targeted_classes": strategy,
            "selected_categorical_columns": lc.get("selected_categorical_columns"),
            "selected_features": lc.get("selected_features"),
            "n_rows_X": int(len(X_selected)),
            "n_rows_y": int(len(y_selected)),
        }

        # Persist an effective-run metadata file (best-effort; don't overwrite once present)
        eff_path = self.paths["metadata"] / "effective_run_metadata.json"
        if not eff_path.exists():
            eff_meta = {
                "config_used": self.config,
                "base_seed": int(self.base_seed),
                "dataset_path": self.config.get("dataset", {}).get("train_path"),
                "sample_size": None,
                "sampling_seed": None,
                "outer_inner_seed_example": base_diag.get("inner_cv_seed"),
                "model": lc.get("model"),
                "worker_threads": None,
            }
            try:
                atomic_write_json(eff_path, eff_meta)
            except Exception:
                # best-effort only
                pass

        if categorical_indices:
            # SMOTENC path: catch exceptions to persist diagnostics then re-raise
            try:
                X_bal, y_bal, before, after, strategy = apply_smotenc(
                    X_selected,
                    y_selected,
                    categorical_feature_indices=categorical_indices,
                    smote_config=self.config["smotenc"],
                    random_state=stage_random_state,
                )

                # augment diag with after counts and method
                entry = dict(base_diag)
                entry.update({"balancing_method": "smotenc", "y_counts_after": after, "exception": None})
                with diag_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, sort_keys=True) + "\n")

                return X_bal, y_bal, before, after, strategy, "smotenc"
            except Exception as exc:
                entry = dict(base_diag)
                entry.update({"balancing_method": "smotenc", "y_counts_after": None, "exception": str(exc)})
                try:
                    with diag_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, sort_keys=True) + "\n")
                except Exception:
                    pass
                raise

        self.logger.info(
            "SMOTE_USED_BECAUSE_NO_SELECTED_CATEGORICAL_FEATURES",
            extra=log_context or {},
        )
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(
            sampling_strategy=strategy,
            k_neighbors=int(self.config["smotenc"]["k_neighbors"]),
            random_state=stage_random_state,
        )
        X_resampled, y_resampled = sampler.fit_resample(X_selected, y_selected)
        X_bal = pd.DataFrame(X_resampled, columns=X_selected.columns)
        y_bal = pd.Series(y_resampled, name=y_selected.name)
        after = y_bal.value_counts().sort_index().to_dict()
        return X_bal, y_bal, before, after, strategy, "smote"

    def _validate_outer_fold_structure(self, outer_fold_metrics: pd.DataFrame) -> None:
        required_folds = {1, 2, 3, 4, 5}
        if outer_fold_metrics.empty:
            raise ValueError("Statistical analysis requires outer-fold results for all five folds and five models")

        if outer_fold_metrics.duplicated(subset=["model", "outer_fold"]).any():
            raise ValueError("Statistical analysis requires exactly one row per model and outer fold")

        observed_folds = set(map(int, outer_fold_metrics["outer_fold"].unique()))
        if observed_folds != required_folds:
            raise ValueError(f"Statistical analysis requires outer folds {sorted(required_folds)}; got {sorted(observed_folds)}")

        model_names = sorted(map(str, outer_fold_metrics["model"].unique()))
        if len(model_names) != 5:
            raise ValueError(f"Statistical analysis requires results for exactly five models; got {len(model_names)}")

        for model_name in model_names:
            model_folds = set(map(int, outer_fold_metrics.loc[outer_fold_metrics["model"] == model_name, "outer_fold"].tolist()))
            if model_folds != required_folds:
                raise ValueError(f"Model {model_name} must have exactly one observation for folds 1-5; got {sorted(model_folds)}")

    def _prepare_smoke_subsets_from_loaded_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        target = self.config["dataset"]["target_column"]
        expected = self.config["dataset"]["expected_classes"]
        smoke_cfg = self.config["smoke_test"]

        train_n = int(smoke_cfg.get("rows_per_class_train", 60))
        test_n = int(smoke_cfg.get("rows_per_class_test", 30))
        seed = int(self.config["experiment"]["base_seed"])

        train_parts = []
        test_parts = []
        for cls in expected:
            cls_train = train_df[train_df[target].astype(str) == cls]
            cls_test = test_df[test_df[target].astype(str) == cls]
            if cls_train.empty or cls_test.empty:
                raise ValueError(f"Smoke subset cannot be built; class missing: {cls}")
            train_parts.append(cls_train.sample(n=min(train_n, len(cls_train)), random_state=seed))
            test_parts.append(cls_test.sample(n=min(test_n, len(cls_test)), random_state=seed))

        train_out = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_out = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return train_out, test_out

    def _build_fitness_fn(
        self,
        model_name: str,
        model_idx: int,
        outer_fold: int,
        X_outer_train_raw: pd.DataFrame,
        y_outer_train: pd.Series,
    ):
        # Build a fitness function that uses a single deterministic stratified HPO validation split
        validation_size = float(self.config.get("hpo", {}).get("validation_size", 0.20))

        def fitness_fn(params: dict[str, Any], iteration: int, wolf_idx: int) -> dict[str, float]:
            # derive deterministic split seed
            split_seed = derive_seed(self.base_seed, model_name, outer_fold, "hpo_split")

            X_hpo_train_raw, X_hpo_val_raw, y_hpo_train, y_hpo_val = train_test_split(
                X_outer_train_raw,
                y_outer_train,
                test_size=validation_size,
                stratify=y_outer_train,
                random_state=split_seed,
            )

            if set(X_hpo_train_raw.index).intersection(set(X_hpo_val_raw.index)):
                raise AssertionError("Leakage check failed: HPO train/val overlap")

            self.state["active_progress"] = {
                "phase": "outer_cv",
                "model": model_name,
                "outer_fold": outer_fold,
                "inner_fold": None,
                "gwo_iteration": iteration + 1,
                "wolf_index": wolf_idx,
            }

            timings: dict[str, float] = {}
            t0 = time.perf_counter()

            # preprocessing
            t_prep0 = time.perf_counter()
            preproc = FoldPreprocessor(categorical_columns=self.dataset_cfg["categorical_columns"])
            preproc.fit(X_hpo_train_raw)
            X_hpo_train_encoded = preproc.transform(X_hpo_train_raw)
            X_hpo_val_encoded = preproc.transform(X_hpo_val_raw)
            timings["preprocessing"] = time.perf_counter() - t_prep0

            # feature selection (fit only on HPO-train)
            t_fs0 = time.perf_counter()
            fs_seed = 42
            selected_features, _ = select_top_features(
                X_hpo_train_encoded,
                y_hpo_train,
                config=self.config,
                random_state=fs_seed,
            )
            selected_categorical_columns = get_selected_categorical_columns(
                selected_features,
                self.dataset_cfg["categorical_columns"],
            )
            timings["feature_selection"] = time.perf_counter() - t_fs0

            X_hpo_train_selected = X_hpo_train_encoded[selected_features].reset_index(drop=True)
            X_hpo_val_selected = X_hpo_val_encoded[selected_features].reset_index(drop=True)

            # balancing: only fit on HPO-train
            t_bal0 = time.perf_counter()
            cat_indices = [i for i, col in enumerate(X_hpo_train_selected.columns) if col in selected_categorical_columns]
            bal_seed = derive_seed(self.base_seed, "balancing", model_name, outer_fold, iteration, wolf_idx)
            X_bal, y_bal, before, after, strategy, balancing_method = self._apply_balancing_with_policy(
                X_hpo_train_selected,
                y_hpo_train.reset_index(drop=True),
                cat_indices,
                stage_random_state=bal_seed,
                log_context={
                    "model": model_name,
                    "phase": "outer_cv",
                    "outer_fold": outer_fold,
                    "gwo_iteration": iteration + 1,
                    "wolf_index": wolf_idx,
                    "selected_features": selected_features,
                    "selected_categorical_columns": selected_categorical_columns,
                },
            )
            timings["balancing"] = time.perf_counter() - t_bal0

            self.logger.info(
                "HPO balancing completed",
                extra={"model": model_name, "outer_fold": outer_fold, "wolf": wolf_idx},
            )

            # model fitting
            t_fit0 = time.perf_counter()
            transformer = ModelTransformer(
                categorical_columns=selected_categorical_columns,
                scale=model_requires_scaling(model_name),
            )
            X_train_proc = transformer.fit_transform(X_bal)
            X_val_proc = transformer.transform(X_hpo_val_selected)

            y_train_enc = self._encode_y(y_bal)
            model_seed = derive_seed(self.base_seed, "model", model_name, outer_fold, iteration, wolf_idx)
            model = build_model(
                model_name=model_name,
                params=params,
                random_state=model_seed,
                num_classes=len(self.expected_classes),
            )
            model.fit(X_train_proc, y_train_enc)
            timings["model_fit"] = time.perf_counter() - t_fit0

            # validation prediction
            t_pred0 = time.perf_counter()
            pred_enc = model.predict(X_val_proc)
            timings["validation_prediction"] = time.perf_counter() - t_pred0
            pred = self._decode_y(pred_enc)

            # metric calculation
            t_metric0 = time.perf_counter()
            ba, macro_f1 = compute_balanced_accuracy_macro_f1(y_hpo_val.tolist(), pred)
            timings["metric_calculation"] = time.perf_counter() - t_metric0

            total_duration = time.perf_counter() - t0

            # emit structured runtime event for this candidate evaluation
            try:
                event = {
                    "event": "candidate_evaluation",
                    "timestamp_start": utc_now_iso(),
                    "duration_seconds": total_duration,
                    "model": model_name,
                    "phase": "gwo_fitness",
                    "outer_fold": outer_fold,
                    "gwo_iteration": iteration + 1,
                    "wolf_index": wolf_idx,
                    "stage_durations": timings,
                    "sample_size": int(len(X_outer_train_raw) + len(X_outer_val_raw)),
                    "n_rows_hpo_train": int(len(X_hpo_train_raw)),
                    "n_rows_hpo_validation": int(len(X_hpo_val_raw)),
                    "n_features": int(len(selected_features)),
                    "workers": int(self.config.get("experiment", {}).get("workers", 1)),
                    "threads": int(self.config.get("experiment", {}).get("threads", 1)),
                    "random_state": int(model_seed),
                }
                self._emit_runtime_event(event)
            except Exception:
                pass

            fitness = (float(ba) + float(macro_f1)) / 2.0
            return {
                "fitness": float(fitness),
                "mean_balanced_accuracy": float(ba),
                "mean_macro_f1": float(macro_f1),
            }

        return fitness_fn

    def _run_single_outer_model(
        self,
        model_name: str,
        model_idx: int,
        outer_fold: int,
        X_outer_train_raw: pd.DataFrame,
        y_outer_train: pd.Series,
        X_outer_val_raw: pd.DataFrame,
        y_outer_val: pd.Series,
    ) -> None:
        fitness_fn = self._build_fitness_fn(
            model_name=model_name,
            model_idx=model_idx,
            outer_fold=outer_fold,
            X_outer_train_raw=X_outer_train_raw,
            y_outer_train=y_outer_train.reset_index(drop=True),
        )

        gwo_seed = derive_seed(self.base_seed, "gwo", model_name, outer_fold)
        t0 = time.perf_counter()
        gwo_result = self._run_gwo(
            phase="outer_cv",
            model_name=model_name,
            outer_fold=outer_fold,
            fitness_fn=fitness_fn,
            model_seed=gwo_seed,
        )
        gwo_runtime = time.perf_counter() - t0

        best_params = gwo_result["best_params"]
        best_fitness = float(gwo_result["best_fitness"])

        preproc = FoldPreprocessor(categorical_columns=self.dataset_cfg["categorical_columns"])
        preproc.fit(X_outer_train_raw)
        X_outer_train_encoded = preproc.transform(X_outer_train_raw)
        X_outer_val_encoded = preproc.transform(X_outer_val_raw)

        fs_seed = 42
        selected_features, rank_df = select_top_features(
            X_outer_train_encoded,
            y_outer_train,
            config=self.config,
            random_state=fs_seed,
        )
        self._append_outer_fold_feature_rows(rank_df, outer_fold, phase="outer_cv")

        selected_categorical_columns = get_selected_categorical_columns(
            selected_features,
            self.dataset_cfg["categorical_columns"],
        )

        X_train_sel = X_outer_train_encoded[selected_features].reset_index(drop=True)
        X_val_sel = X_outer_val_encoded[selected_features].reset_index(drop=True)

        cat_indices = [i for i, col in enumerate(X_train_sel.columns) if col in selected_categorical_columns]
        X_bal, y_bal, before_dist, after_dist, _, balancing_method = self._apply_balancing_with_policy(
            X_train_sel,
            y_outer_train.reset_index(drop=True),
            cat_indices,
            stage_random_state=42,
            log_context={
                "model": model_name,
                "outer_fold": outer_fold,
                "selected_features": selected_features,
                "selected_categorical_columns": selected_categorical_columns,
            },
        )
        self._record_class_distribution(
            phase="outer_cv",
            model=model_name,
            outer_fold=outer_fold,
            before=before_dist,
            after=after_dist,
        )

        transformer = ModelTransformer(
            categorical_columns=selected_categorical_columns,
            scale=model_requires_scaling(model_name),
        )
        X_train_proc = transformer.fit_transform(X_bal)
        X_val_proc = transformer.transform(X_val_sel)

        y_train_enc = self._encode_y(y_bal)
        model_seed = derive_seed(self.base_seed, "fit", model_name, outer_fold)
        model = build_model(
            model_name=model_name,
            params=best_params,
            random_state=model_seed,
            num_classes=len(self.expected_classes),
        )

        t_train_0 = time.perf_counter()
        model.fit(X_train_proc, y_train_enc)
        train_time = time.perf_counter() - t_train_0

        t_pred_0 = time.perf_counter()
        pred_enc = model.predict(X_val_proc)
        pred_time = time.perf_counter() - t_pred_0
        pred = self._decode_y(pred_enc)

        ba, macro_f1 = compute_balanced_accuracy_macro_f1(y_outer_val.tolist(), pred)

        metrics_row = {
            "phase": "outer_cv",
            "model": model_name,
            "outer_fold": outer_fold,
            "balanced_accuracy": ba,
            "macro_f1": macro_f1,
            "n_train_before_smotenc": int(len(X_train_sel)),
            "n_train_after_smotenc": int(len(X_bal)),
            "selected_features": ";".join(selected_features),
            "training_time_sec": train_time,
            "prediction_time_sec": pred_time,
            "best_gwo_fitness": best_fitness,
            "best_hyperparameters_json": json.dumps(best_params, sort_keys=True),
            "gwo_runtime_sec": gwo_runtime,
        }

        upsert_csv_rows(
            self.paths["results"] / "outer_fold_metrics.csv",
            [metrics_row],
            fieldnames=[
                "phase",
                "model",
                "outer_fold",
                "balanced_accuracy",
                "macro_f1",
                "n_train_before_smotenc",
                "n_train_after_smotenc",
                "selected_features",
                "training_time_sec",
                "prediction_time_sec",
                "best_gwo_fitness",
                "best_hyperparameters_json",
                "gwo_runtime_sec",
            ],
            key_fields=["phase", "model", "outer_fold"],
        )

        pred_rows = []
        for i, (truth, p) in enumerate(zip(y_outer_val.tolist(), pred)):
            pred_rows.append(
                {
                    "phase": "outer_cv",
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "row_in_fold": i,
                    "y_true": truth,
                    "y_pred": p,
                }
            )
        upsert_csv_rows(
            self.paths["results"] / "outer_fold_predictions.csv",
            pred_rows,
            fieldnames=["phase", "model", "outer_fold", "row_in_fold", "y_true", "y_pred"],
            key_fields=["phase", "model", "outer_fold", "row_in_fold"],
        )

        self._mark_outer_model_completed(model_name, outer_fold)

    def run_outer_cv(self, train_df: pd.DataFrame) -> None:
        if self.state.get("outer_cv_complete", False):
            self.logger.info("Outer CV already completed, skipping")
            return

        X_train_all, y_train_all = split_xy(train_df, self.config)
        skf_outer = StratifiedKFold(
            n_splits=int(self.config["cv"]["outer_folds"]),
            shuffle=bool(self.config["cv"].get("shuffle", True)),
            random_state=self.base_seed,
        )

        for outer_fold_idx, (outer_tr_idx, outer_val_idx) in enumerate(
            skf_outer.split(X_train_all, y_train_all),
            start=1,
        ):
            if outer_fold_idx in self.state["completed_outer_folds"]:
                self.logger.info("Skipping completed outer fold %d", outer_fold_idx)
                continue

            if set(outer_tr_idx).intersection(set(outer_val_idx)):
                raise AssertionError("Leakage check failed: outer train/val overlap")

            if self.state.get("official_test_used", False):
                raise AssertionError("Leakage check failed: official test set was marked as used before outer CV completion")

            X_outer_train_raw = X_train_all.iloc[outer_tr_idx].reset_index(drop=True)
            y_outer_train = y_train_all.iloc[outer_tr_idx].reset_index(drop=True)
            X_outer_val_raw = X_train_all.iloc[outer_val_idx].reset_index(drop=True)
            y_outer_val = y_train_all.iloc[outer_val_idx].reset_index(drop=True)

            self.logger.info(
                "Leakage checks passed for outer split",
                extra={"outer_fold": outer_fold_idx},
            )

            # Per revised methodology: do not perform feature-selection on full outer-training here.
            # Feature selection is performed on HPO-train during GWO fitness evaluations
            # and on full outer-training only after best hyperparameters are selected.

            for model_idx, model_name in enumerate(self.model_names):
                if self._is_outer_model_completed(model_name, outer_fold_idx):
                    self.logger.info(
                        "Skipping completed model/fold: %s fold=%d",
                        model_name,
                        outer_fold_idx,
                    )
                    continue

                self.logger.info(
                    "Running model in outer fold",
                    extra={
                        "model": model_name,
                        "outer_fold": outer_fold_idx,
                    },
                )
                self._run_single_outer_model(
                    model_name=model_name,
                    model_idx=model_idx,
                    outer_fold=outer_fold_idx,
                    X_outer_train_raw=X_outer_train_raw,
                    y_outer_train=y_outer_train,
                    X_outer_val_raw=X_outer_val_raw,
                    y_outer_val=y_outer_val,
                )

            self.state["completed_outer_folds"].append(outer_fold_idx)
            self._save_state()

        self.state["outer_cv_complete"] = True
        self._save_state()

        outer_metrics_file = self.paths["results"] / "outer_fold_metrics.csv"
        if outer_metrics_file.exists():
            outer_metrics = pd.read_csv(outer_metrics_file)
            self._validate_outer_fold_structure(outer_metrics)
            friedman_df, pairwise_df = run_statistical_analysis(outer_metrics, alpha=0.05)
            stat_rows = []
            for _, row in friedman_df.iterrows():
                stat_rows.append(
                    {
                        "test_type": "friedman",
                        "metric": row["metric"],
                        "model_a": "",
                        "model_b": "",
                        "friedman_statistic": row["friedman_statistic"],
                        "raw_p_value": row["p_value"],
                        "holm_adjusted_p_value": "",
                        "significant": row["significant"],
                    }
                )
            for _, row in pairwise_df.iterrows():
                stat_rows.append(
                    {
                        "test_type": "wilcoxon_holm",
                        "metric": row["metric"],
                        "model_a": row["model_a"],
                        "model_b": row["model_b"],
                        "friedman_statistic": "",
                        "raw_p_value": row["raw_p_value"],
                        "holm_adjusted_p_value": row["holm_adjusted_p_value"],
                        "significant": row["significant"],
                    }
                )
            pd.DataFrame(stat_rows).to_csv(self.paths["results"] / "statistical_results.csv", index=False)
            if not pairwise_df.empty:
                pairwise_df.to_csv(self.paths["results"] / "pairwise_wilcoxon_results.csv", index=False)

    def run_final_training_and_test(self, train_df: pd.DataFrame) -> None:
        if self.state.get("final_training_complete", False):
            self.logger.info("Final training and test evaluation already completed, skipping")
            return

        if not self.state.get("outer_cv_complete", False):
            raise RuntimeError("Outer CV must complete before final test evaluation")

        X_train_raw, y_train = split_xy(train_df, self.config)
        test_df = load_test_dataset(self.config, logger=self.logger)
        self.state["official_test_used"] = True
        self._save_state()

        X_test_raw, y_test = split_xy(test_df, self.config)

        if self.ctx.smoke_test:
            _, test_df = self._prepare_smoke_subsets_from_loaded_data(train_df, test_df)
            X_test_raw, y_test = split_xy(test_df, self.config)

        preproc = FoldPreprocessor(categorical_columns=self.dataset_cfg["categorical_columns"])
        preproc.fit(X_train_raw)
        X_train_encoded = preproc.transform(X_train_raw)
        X_test_encoded = preproc.transform(X_test_raw)

        fs_seed = derive_seed(self.base_seed, "feature_selection", "final_train")
        selected_features, rank_df = select_top_features(
            X_train_encoded,
            y_train,
            config=self.config,
            random_state=fs_seed,
        )
        self._append_outer_fold_feature_rows(rank_df, outer_fold=-1, phase="final_train")

        selected_categorical_columns = get_selected_categorical_columns(
            selected_features,
            self.dataset_cfg["categorical_columns"],
        )

        X_train_sel = X_train_encoded[selected_features].reset_index(drop=True)
        X_test_sel = X_test_encoded[selected_features].reset_index(drop=True)

        for model_idx, model_name in enumerate(self.model_names):
            if self._is_final_model_completed(model_name):
                self.logger.info("Skipping completed final model: %s", model_name)
                continue

            self.logger.info("Running final model optimization and test", extra={"model": model_name})

            fitness_fn = self._build_fitness_fn(
                model_name=model_name,
                model_idx=model_idx,
                outer_fold=0,
                X_outer_train_raw=X_train_raw,
                y_outer_train=y_train,
            )

            gwo_seed = derive_seed(self.base_seed, "gwo", model_name, "final_train")
            t0 = time.perf_counter()
            gwo_result = self._run_gwo(
                phase="final_train",
                model_name=model_name,
                outer_fold=0,
                fitness_fn=fitness_fn,
                model_seed=gwo_seed,
            )
            gwo_runtime = time.perf_counter() - t0

            best_params = gwo_result["best_params"]
            best_fitness = float(gwo_result["best_fitness"])

            cat_indices = [i for i, col in enumerate(X_train_sel.columns) if col in selected_categorical_columns]
            X_bal, y_bal, before_dist, after_dist, _, balancing_method = self._apply_balancing_with_policy(
                X_train_sel,
                y_train,
                cat_indices,
                stage_random_state=42,
                log_context={
                    "model": model_name,
                    "phase": "final_train",
                    "selected_features": selected_features,
                    "selected_categorical_columns": selected_categorical_columns,
                },
            )
            self._record_class_distribution(
                phase="final_train",
                model=model_name,
                outer_fold=0,
                before=before_dist,
                after=after_dist,
            )

            transformer = ModelTransformer(
                categorical_columns=selected_categorical_columns,
                scale=model_requires_scaling(model_name),
            )
            X_train_proc = transformer.fit_transform(X_bal)
            X_test_proc = transformer.transform(X_test_sel)

            y_train_enc = self._encode_y(y_bal)
            model_seed = derive_seed(self.base_seed, "fit", model_name, "final_train")
            model = build_model(
                model_name=model_name,
                params=best_params,
                random_state=model_seed,
                num_classes=len(self.expected_classes),
            )

            t_train = time.perf_counter()
            model.fit(X_train_proc, y_train_enc)
            train_time = time.perf_counter() - t_train

            t_pred = time.perf_counter()
            pred_enc = model.predict(X_test_proc)
            pred_time = time.perf_counter() - t_pred
            pred = self._decode_y(pred_enc)

            ba, macro_f1 = compute_balanced_accuracy_macro_f1(y_test.tolist(), pred)

            upsert_csv_rows(
                self.paths["results"] / "final_hyperparameters.csv",
                [
                    {
                        "model": model_name,
                        "phase": "final_train",
                        "best_fitness": best_fitness,
                        "gwo_runtime_sec": gwo_runtime,
                        "hyperparameters_json": json.dumps(best_params, sort_keys=True),
                    }
                ],
                fieldnames=["model", "phase", "best_fitness", "gwo_runtime_sec", "hyperparameters_json"],
                key_fields=["model", "phase"],
            )

            upsert_csv_rows(
                self.paths["results"] / "final_test_metrics.csv",
                [
                    {
                        "model": model_name,
                        "balanced_accuracy": ba,
                        "macro_f1": macro_f1,
                        "training_time_sec": train_time,
                        "prediction_time_sec": pred_time,
                    }
                ],
                fieldnames=[
                    "model",
                    "balanced_accuracy",
                    "macro_f1",
                    "training_time_sec",
                    "prediction_time_sec",
                ],
                key_fields=["model"],
            )

            pred_rows = []
            for i, (truth, p) in enumerate(zip(y_test.tolist(), pred)):
                pred_rows.append(
                    {
                        "model": model_name,
                        "row_in_test": i,
                        "y_true": truth,
                        "y_pred": p,
                    }
                )
            upsert_csv_rows(
                self.paths["results"] / "final_test_predictions.csv",
                pred_rows,
                fieldnames=["model", "row_in_test", "y_true", "y_pred"],
                key_fields=["model", "row_in_test"],
            )

            cm_df = confusion_matrix_df(y_test.tolist(), pred, self.expected_classes)
            cm_df.to_csv(self.paths["results"] / "confusion_matrices" / f"{model_name}.csv")

            cls_rep = classification_report_dict(y_test.tolist(), pred, self.expected_classes)
            atomic_write_json(
                self.paths["results"] / f"classification_report_{model_name}.json",
                cls_rep,
            )

            if bool(self.config["experiment"].get("save_models", True)):
                joblib.dump(
                    {
                        "model": model,
                        "transformer": transformer,
                        "preprocessor": preproc,
                        "selected_features": selected_features,
                        "model_name": model_name,
                        "best_params": best_params,
                    },
                    self.paths["models"] / f"final_model_{model_name}.joblib",
                )

            self._mark_final_model_completed(model_name)

        self.state["final_training_complete"] = True
        self._save_state()

    def run_reports(self) -> None:
        generate_reports(self.ctx.run_dir, self.config)
        self.state["reports_generated"] = True
        self._save_state()

    def status(self) -> dict[str, Any]:
        return {
            "run_id": self.ctx.run_id,
            "smoke_test": self.ctx.smoke_test,
            "outer_cv_complete": self.state.get("outer_cv_complete", False),
            "final_training_complete": self.state.get("final_training_complete", False),
            "completed_outer_models": len(self.state.get("completed_outer_models", [])),
            "completed_outer_folds": self.state.get("completed_outer_folds", []),
            "completed_final_models": self.state.get("completed_final_models", []),
            "reports_generated": self.state.get("reports_generated", False),
            "last_error": self.state.get("last_error"),
        }

    def execute_full_pipeline(self, train_df: pd.DataFrame) -> None:
        try:
            self.run_outer_cv(train_df)
            self.run_final_training_and_test(train_df)
            self.run_reports()
            self.state["last_error"] = None
            self._save_state()
        except Exception as exc:
            self._mark_error(str(exc))
            self.logger.exception("Fatal structural error, checkpoint saved")
            raise
