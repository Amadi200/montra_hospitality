"""
Objective 3 – Data Cleaning and Preparation
Handle missing values, duplicates, clean review text, and produce a reliable corpus.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.config import DATA_DIR, MIN_REVIEW_LENGTH, RANDOM_SEED, SENTIMENT_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def clean_review_text(text: str) -> str:
    """
    Lightweight cleaning suitable for both classical ML and transformer models.
    Does NOT lower-case (BERT is case-sensitive) or remove stop-words here;
    those decisions live in the feature-extraction stage.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    # Keep apostrophes and basic punctuation that carry sentiment
    text = re.sub(r"[^\w\s'.,!?-]", " ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_for_classical_ml(text: str) -> str:
    """More aggressive cleaning for TF-IDF / bag-of-words baselines."""
    text = clean_review_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------
def clean_dataset(
    df: pd.DataFrame,
    min_length: int = MIN_REVIEW_LENGTH,
) -> pd.DataFrame:
    """
    Full cleaning pipeline for the consolidated guest-experience dataset.

    Steps:
      1. Drop exact duplicate review_ids
      2. Drop rows with missing / empty review_text
      3. Clean review text
      4. Filter by minimum length
      5. Validate sentiment labels & rating_score
      6. Fill missing loyalty_status
      7. Standardise categorical columns
    """
    logger.info("Starting data cleaning | input shape: %s", df.shape)
    out = df.copy()

    # 1. Deduplicate on review_id
    n_before = len(out)
    out = out.drop_duplicates(subset=["review_id"], keep="first")
    logger.info("Dropped %d duplicate review_ids", n_before - len(out))

    # 2. Missing / empty text
    out = out[out["review_text"].notna() & (out["review_text"].str.strip() != "")]
    logger.info("After removing empty reviews: %d", len(out))

    # 3. Text cleaning
    out["review_text_clean"] = out["review_text"].apply(clean_review_text)
    out["review_text_ml"] = out["review_text"].apply(clean_for_classical_ml)

    # 4. Length filter
    out["text_length"] = out["review_text_clean"].str.len()
    out = out[out["text_length"] >= min_length]
    logger.info("After min-length filter (%d chars): %d", min_length, len(out))

    # 5. Sentiment & rating validation
    out["sentiment_label"] = out["sentiment_label"].str.lower().str.strip()
    valid_mask = out["sentiment_label"].isin(SENTIMENT_LABELS)
    if (~valid_mask).any():
        logger.warning("Dropping %d rows with invalid sentiment labels", (~valid_mask).sum())
        out = out[valid_mask]

    # Ensure rating is numeric and in plausible range
    out["rating_score"] = pd.to_numeric(out["rating_score"], errors="coerce")
    out = out[out["rating_score"].between(1.0, 5.0)]

    # 6. Loyalty status – treat NaN as "None" (non-member)
    if "gt_loyalty_status" in out.columns:
        out["gt_loyalty_status"] = out["gt_loyalty_status"].fillna("None")

    # 7. Standardise a few categoricals
    cat_cols = [
        "gt_customer_type", "gt_loyalty_status", "gt_country",
        "bk_room_type", "bk_booking_channel",
        "ht_location", "ht_hotel_name",
    ]
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype(str)

    # Word count for EDA
    out["word_count"] = out["review_text_clean"].str.split().str.len()

    logger.info("Cleaning complete | output shape: %s", out.shape)
    return out.reset_index(drop=True)


def run_data_quality_checks(df: pd.DataFrame) -> dict:
    """
    Lightweight data-quality report (Great Expectations can be layered on later).
    Returns a dictionary of check results.
    """
    checks = {
        "n_rows": len(df),
        "n_unique_reviews": df["review_id"].nunique(),
        "null_review_text": int(df["review_text_clean"].isna().sum()),
        "empty_review_text": int((df["review_text_clean"].str.strip() == "").sum()),
        "invalid_sentiment": int((~df["sentiment_label"].isin(SENTIMENT_LABELS)).sum()),
        "rating_out_of_range": int((~df["rating_score"].between(1, 5)).sum()),
        "duplicate_review_ids": int(df["review_id"].duplicated().sum()),
        "min_text_length": int(df["text_length"].min()),
        "max_text_length": int(df["text_length"].max()),
        "mean_text_length": float(df["text_length"].mean()),
    }
    logger.info("Data quality checks: %s", checks)
    return checks


def save_cleaned(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or (DATA_DIR / "cleaned_reviews.parquet")
    try:
        df.to_parquet(path, index=False)
    except Exception:
        path = path.with_suffix(".pkl")
        df.to_pickle(path)
    logger.info("Saved cleaned dataset → %s (%d rows)", path, len(df))
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data.extract import load_or_extract

    raw_df, _ = load_or_extract()
    cleaned = clean_dataset(raw_df)
    checks = run_data_quality_checks(cleaned)
    save_cleaned(cleaned)
    print(cleaned[["review_id", "sentiment_label", "rating_score", "text_length"]].head(10))
    print(checks)
