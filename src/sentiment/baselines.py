"""
Objective 6 (part) – Baseline sentiment classifiers
Logistic Regression, Naive Bayes, SVM, Random Forest with TF-IDF features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config.config import (
    LABEL2ID,
    MODELS_DIR,
    RANDOM_SEED,
    SENTIMENT_LABELS,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


def make_baseline_pipeline(model_name: str) -> Pipeline:
    """Return a sklearn Pipeline of TF-IDF + classifier."""
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )
    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "naive_bayes": MultinomialNB(alpha=0.5),
        "svm": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, dual="auto"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=30, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_SEED,
        ),
    }
    if model_name not in classifiers:
        raise ValueError(f"Unknown baseline: {model_name}. Choose from {list(classifiers)}")
    return Pipeline([("tfidf", vectorizer), ("clf", classifiers[model_name])])


def prepare_splits(
    df: pd.DataFrame,
    text_col: str = "review_text_ml",
    label_col: str = "sentiment_label",
) -> dict[str, Any]:
    """Stratified train / validation / test split."""
    X = df[text_col].values
    y = df[label_col].map(LABEL2ID).values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=RANDOM_SEED
    )
    logger.info(
        "Splits → train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def train_and_evaluate_baseline(
    model_name: str,
    splits: dict[str, Any],
    save: bool = True,
) -> dict[str, Any]:
    """Train one baseline, evaluate on val & test, optionally persist."""
    logger.info("Training baseline: %s", model_name)
    pipe = make_baseline_pipeline(model_name)
    pipe.fit(splits["X_train"], splits["y_train"])

    results = {"model_name": model_name}
    for split_name in ["val", "test"]:
        X = splits[f"X_{split_name}"]
        y = splits[f"y_{split_name}"]
        preds = pipe.predict(X)
        metrics = evaluate_predictions(y, preds)
        results[f"{split_name}_metrics"] = metrics
        results[f"{split_name}_report"] = classification_report(
            y, preds, target_names=SENTIMENT_LABELS, zero_division=0
        )
        results[f"{split_name}_cm"] = confusion_matrix(y, preds).tolist()
        logger.info(
            "  %s | acc=%.4f  f1_macro=%.4f",
            split_name, metrics["accuracy"], metrics["f1_macro"],
        )

    if save:
        path = MODELS_DIR / f"baseline_{model_name}.joblib"
        joblib.dump(pipe, path)
        logger.info("Saved → %s", path)
        results["model_path"] = str(path)

    return results


def run_all_baselines(df: pd.DataFrame) -> list[dict]:
    """Train and evaluate every baseline model."""
    splits = prepare_splits(df)
    all_results = []
    for name in ["logistic_regression", "naive_bayes", "svm", "random_forest"]:
        res = train_and_evaluate_baseline(name, splits)
        all_results.append(res)
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data.extract import load_or_extract
    from src.data.clean import clean_dataset

    raw, _ = load_or_extract()
    cleaned = clean_dataset(raw)
    results = run_all_baselines(cleaned)
    for r in results:
        print(r["model_name"], r["test_metrics"])
