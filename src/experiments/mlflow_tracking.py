"""
Objective 7 – Machine Learning Experimentation with MLflow
Systematically track parameters, metrics and model versions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None
import pandas as pd

from config.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> None:
    """Initialise MLflow tracking."""
    if not HAS_MLFLOW:
        logger.warning("MLflow not installed – skipping experiment tracking")
        return
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s | experiment: %s", MLFLOW_TRACKING_URI, experiment_name)


def log_baseline_run(
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model_path: Optional[str] = None,
    tags: Optional[dict] = None,
) -> str:
    """Log a classical ML baseline experiment."""
    setup_mlflow()
    if not HAS_MLFLOW:
        return "mlflow-not-available"
    with mlflow.start_run(run_name=f"baseline_{model_name}") as run:
        mlflow.set_tags({"model_family": "baseline", "model_name": model_name, **(tags or {})})
        mlflow.log_params(params)
        # Flatten nested metrics if needed
        flat = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for mk, mv in v.items():
                    flat[f"{k}_{mk}"] = mv
            else:
                flat[k] = v
        mlflow.log_metrics(flat)
        if model_path and Path(model_path).exists():
            mlflow.log_artifact(model_path)
        logger.info("Logged baseline run: %s (id=%s)", model_name, run.info.run_id)
        return run.info.run_id


def log_bert_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    model_dir: Optional[str] = None,
    tags: Optional[dict] = None,
) -> str:
    """Log a BERT fine-tuning experiment."""
    setup_mlflow()
    if not HAS_MLFLOW:
        return "mlflow-not-available"
    with mlflow.start_run(run_name="bert_sentiment") as run:
        mlflow.set_tags({"model_family": "transformer", "model_name": "bert", **(tags or {})})
        mlflow.log_params(params)
        flat = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for mk, mv in v.items():
                    flat[f"{k}_{mk}"] = mv
            else:
                flat[k] = v
        mlflow.log_metrics(flat)
        if model_dir and Path(model_dir).exists():
            # Log the directory as artifacts (weights can be large – optional)
            mlflow.log_artifacts(model_dir, artifact_path="bert_model")
        logger.info("Logged BERT run (id=%s)", run.info.run_id)
        return run.info.run_id


def compare_runs(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> pd.DataFrame:
    """Return a comparison table of all runs in the experiment."""
    setup_mlflow(experiment_name)
    if not HAS_MLFLOW:
        return pd.DataFrame()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning("Experiment '%s' not found", experiment_name)
        return pd.DataFrame()

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
    if runs.empty:
        return runs

    # Select useful columns
    cols = [c for c in runs.columns if c.startswith("metrics.") or c.startswith("params.") or c in (
        "run_id", "tags.model_name", "tags.model_family", "start_time", "status"
    )]
    comparison = runs[cols].copy()
    comparison.to_csv(REPORTS_DIR / "mlflow_run_comparison.csv", index=False)
    logger.info("Run comparison saved → %s (%d runs)", REPORTS_DIR / "mlflow_run_comparison.csv", len(comparison))
    return comparison


def run_experiment_suite(
    baseline_results: list[dict],
    bert_results: Optional[dict] = None,
    bert_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Log every experiment result produced by the modelling pipeline
    and return the comparison table.
    """
    setup_mlflow()

    for res in baseline_results:
        params = {"model_name": res["model_name"]}
        metrics = {
            "val": res.get("val_metrics", {}),
            "test": res.get("test_metrics", {}),
        }
        log_baseline_run(
            model_name=res["model_name"],
            params=params,
            metrics=metrics,
            model_path=res.get("model_path"),
        )

    if bert_results is not None:
        log_bert_run(
            params=bert_params or {},
            metrics={
                "test": bert_results.get("test_metrics", {}),
                "train": bert_results.get("train_metrics", {}),
            },
            model_dir=bert_results.get("model_path"),
        )

    return compare_runs()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    setup_mlflow()
    print("MLflow ready. Tracking URI:", MLFLOW_TRACKING_URI)
