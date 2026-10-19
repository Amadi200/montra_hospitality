"""
Objective 6 – BERT Sentiment Classification
Fine-tune a pre-trained BERT model for Positive / Neutral / Negative classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config.config import (
    BERT_CONFIG,
    ID2LABEL,
    LABEL2ID,
    MODELS_DIR,
    RANDOM_SEED,
    SENTIMENT_LABELS,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Metrics helper for Trainer
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def prepare_bert_splits(
    df: pd.DataFrame,
    text_col: str = "review_text_clean",
    label_col: str = "sentiment_label",
) -> dict[str, Any]:
    texts = df[text_col].tolist()
    labels = df[label_col].map(LABEL2ID).tolist()

    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_SEED
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=RANDOM_SEED
    )
    logger.info("BERT splits → train: %d | val: %d | test: %d", len(X_train), len(X_val), len(X_test))
    return {
        "train_texts": X_train, "train_labels": y_train,
        "val_texts": X_val, "val_labels": y_val,
        "test_texts": X_test, "test_labels": y_test,
    }


def train_bert(
    splits: dict[str, Any],
    model_name: str = None,
    max_length: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    epochs: int = None,
    output_dir: Optional[Path] = None,
) -> tuple[Any, dict]:
    """
    Fine-tune BERT and return (trainer, metrics_dict).
    """
    cfg = BERT_CONFIG
    model_name = model_name or cfg["model_name"]
    max_length = max_length or cfg["max_length"]
    batch_size = batch_size or cfg["batch_size"]
    learning_rate = learning_rate or cfg["learning_rate"]
    epochs = epochs or cfg["epochs"]
    output_dir = output_dir or (MODELS_DIR / "bert_sentiment")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training BERT on device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(SENTIMENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = ReviewDataset(splits["train_texts"], splits["train_labels"], tokenizer, max_length)
    val_ds = ReviewDataset(splits["val_texts"], splits["val_labels"], tokenizer, max_length)
    test_ds = ReviewDataset(splits["test_texts"], splits["test_labels"], tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",          # MLflow handled separately
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting BERT fine-tuning…")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # Evaluate on test set
    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_metrics = {
        "accuracy": float(accuracy_score(splits["test_labels"], test_preds)),
        "f1_macro": float(f1_score(splits["test_labels"], test_preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(splits["test_labels"], test_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(splits["test_labels"], test_preds, average="macro", zero_division=0)),
    }
    report = classification_report(
        splits["test_labels"], test_preds, target_names=SENTIMENT_LABELS, zero_division=0
    )
    cm = confusion_matrix(splits["test_labels"], test_preds).tolist()

    logger.info("Test metrics: %s", test_metrics)
    logger.info("\n%s", report)

    # Persist best model + tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("BERT model saved → %s", output_dir)

    return trainer, {
        "test_metrics": test_metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "train_metrics": train_result.metrics,
        "model_path": str(output_dir),
    }


def predict_sentiment(
    texts: list[str],
    model_dir: Optional[Path] = None,
    batch_size: int = 64,
) -> list[str]:
    """Inference helper – returns list of label strings."""
    model_dir = model_dir or (MODELS_DIR / "bert_sentiment")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True,
            max_length=BERT_CONFIG["max_length"], return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        labels.extend([ID2LABEL[p] for p in preds])
    return labels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data.extract import load_or_extract
    from src.data.clean import clean_dataset

    raw, _ = load_or_extract()
    cleaned = clean_dataset(raw)
    # For a quick smoke-test use a sample; remove sample for full training
    sample = cleaned.sample(min(8000, len(cleaned)), random_state=RANDOM_SEED)
    splits = prepare_bert_splits(sample)
    trainer, results = train_bert(splits, epochs=2)
    print(results["test_metrics"])
