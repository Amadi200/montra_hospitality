"""
Objective 2 – Exploratory Data Analysis
Understand structure, distributions, relationships and textual patterns
before modelling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.config import FIGURES_DIR, REPORTS_DIR, SENTIMENT_LABELS

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


def _save_fig(name: str, fig: Optional[plt.Figure] = None) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Figure saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# Individual analyses
# ---------------------------------------------------------------------------
def plot_sentiment_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = SENTIMENT_LABELS
    counts = df["sentiment_label"].value_counts().reindex(order)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, legend=False)
    ax.set_title("Sentiment Label Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, f"{v:,}", ha="center", fontsize=9)
    return _save_fig("01_sentiment_distribution", fig)


def plot_rating_distribution(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["rating_score"], bins=20, kde=True, ax=axes[0])
    axes[0].set_title("Rating Score Distribution")
    sns.boxplot(x="sentiment_label", y="rating_score", data=df, order=SENTIMENT_LABELS, ax=axes[1])
    axes[1].set_title("Rating by Sentiment Label")
    return _save_fig("02_rating_distribution", fig)


def plot_text_length(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["text_length"], bins=30, ax=axes[0])
    axes[0].set_title("Review Character Length")
    sns.boxplot(x="sentiment_label", y="word_count", data=df, order=SENTIMENT_LABELS, ax=axes[1])
    axes[1].set_title("Word Count by Sentiment")
    return _save_fig("03_text_length", fig)


def plot_reviews_over_time(df: pd.DataFrame) -> Path:
    if "review_date" not in df.columns:
        return Path()
    ts = df.set_index("review_date").resample("ME").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ts.plot(ax=ax, marker="o")
    ax.set_title("Review Volume Over Time (Monthly)")
    ax.set_ylabel("Number of Reviews")
    return _save_fig("04_reviews_over_time", fig)


def plot_hotel_patterns(df: pd.DataFrame) -> Path:
    hotel_col = "ht_hotel_name" if "ht_hotel_name" in df.columns else "hotel_id"
    loc_col = "ht_location" if "ht_location" in df.columns else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Average rating per hotel
    avg = df.groupby(hotel_col)["rating_score"].mean().sort_values()
    avg.plot(kind="barh", ax=axes[0])
    axes[0].set_title("Avg Rating by Hotel")
    axes[0].set_xlabel("Average Rating")

    # Sentiment mix by location
    if loc_col:
        ct = pd.crosstab(df[loc_col], df["sentiment_label"], normalize="index")
        ct[SENTIMENT_LABELS].plot(kind="bar", stacked=True, ax=axes[1])
        axes[1].set_title("Sentiment Mix by Location")
        axes[1].set_ylabel("Proportion")
        axes[1].legend(title="Sentiment", bbox_to_anchor=(1.02, 1))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    return _save_fig("05_hotel_patterns", fig)


def plot_guest_segments(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "gt_customer_type" in df.columns:
        ct = pd.crosstab(df["gt_customer_type"], df["sentiment_label"], normalize="index")
        ct[SENTIMENT_LABELS].plot(kind="bar", ax=axes[0])
        axes[0].set_title("Sentiment by Customer Type")
        axes[0].set_ylabel("Proportion")
        axes[0].tick_params(axis="x", rotation=30)
    if "gt_loyalty_status" in df.columns:
        ct = pd.crosstab(df["gt_loyalty_status"], df["sentiment_label"], normalize="index")
        ct[SENTIMENT_LABELS].plot(kind="bar", ax=axes[1])
        axes[1].set_title("Sentiment by Loyalty Status")
        axes[1].tick_params(axis="x", rotation=30)
    return _save_fig("06_guest_segments", fig)


def plot_rating_vs_text_length(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sample = df.sample(min(5000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample, x="text_length", y="rating_score",
        hue="sentiment_label", hue_order=SENTIMENT_LABELS,
        alpha=0.4, ax=ax, s=15,
    )
    ax.set_title("Review Length vs Rating Score")
    return _save_fig("07_rating_vs_length", fig)


def correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric correlations useful for reporting."""
    num_cols = ["rating_score", "text_length", "word_count"]
    if "bk_total_amount" in df.columns:
        num_cols.append("bk_total_amount")
    if "bk_stay_duration" in df.columns:
        num_cols.append("bk_stay_duration")
    if "ht_star_rating" in df.columns:
        num_cols.append("ht_star_rating")
    if "gt_previous_stays" in df.columns:
        num_cols.append("gt_previous_stays")
    available = [c for c in num_cols if c in df.columns]
    return df[available].corr().round(3)


def generate_eda_report(df: pd.DataFrame) -> Path:
    """
    Run the full EDA suite, save figures, and write a short markdown summary.
    """
    logger.info("Running full EDA suite…")
    figures = []
    figures.append(plot_sentiment_distribution(df))
    figures.append(plot_rating_distribution(df))
    figures.append(plot_text_length(df))
    figures.append(plot_reviews_over_time(df))
    figures.append(plot_hotel_patterns(df))
    figures.append(plot_guest_segments(df))
    figures.append(plot_rating_vs_text_length(df))

    corr = correlation_summary(df)

    # Summary stats
    summary = {
        "n_reviews": len(df),
        "n_hotels": df["hotel_id"].nunique() if "hotel_id" in df.columns else "N/A",
        "n_guests": df["guest_id"].nunique() if "guest_id" in df.columns else "N/A",
        "date_range": (
            f"{df['review_date'].min().date()} → {df['review_date'].max().date()}"
            if "review_date" in df.columns else "N/A"
        ),
        "sentiment_counts": df["sentiment_label"].value_counts().to_dict(),
        "mean_rating": round(df["rating_score"].mean(), 2),
        "mean_word_count": round(df["word_count"].mean(), 1),
    }

    report_path = REPORTS_DIR / "eda_report.md"
    with open(report_path, "w") as f:
        f.write("# Exploratory Data Analysis Report\n\n")
        f.write("## Dataset Overview\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Correlation Matrix\n\n")
        f.write(corr.to_markdown())
        f.write("\n\n## Figures Generated\n\n")
        for p in figures:
            if p and p.exists():
                f.write(f"- `{p.name}`\n")
    logger.info("EDA report written → %s", report_path)
    return report_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data.extract import load_or_extract
    from src.data.clean import clean_dataset, save_cleaned

    raw, _ = load_or_extract()
    cleaned = clean_dataset(raw)
    save_cleaned(cleaned)
    generate_eda_report(cleaned)
