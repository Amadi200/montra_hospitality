"""
Objective 5 – Topic-level Sentiment Analysis
Measure sentiment per discovered topic rather than only per review.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SENTIMENT_LABELS

logger = logging.getLogger(__name__)


def compute_topic_sentiment(
    df: pd.DataFrame,
    topic_col: str = "topic_id",
    label_col: str = "topic_label",
    sentiment_col: str = "sentiment_label",
    min_docs: int = 20,
) -> pd.DataFrame:
    """
    Aggregate sentiment distribution for each topic.

    Returns a DataFrame with columns:
      topic_id, topic_label, n_docs,
      pct_negative, pct_neutral, pct_positive,
      dominant_sentiment, mean_rating (if available)
    """
    if topic_col not in df.columns:
        raise ValueError(f"Column '{topic_col}' not found – run topic modelling first.")

    # Exclude outliers
    work = df[df[topic_col] != -1].copy()

    agg = (
        work.groupby([topic_col, label_col])
        .agg(
            n_docs=(sentiment_col, "size"),
            **{f"n_{s}": (sentiment_col, lambda x, s=s: (x == s).sum()) for s in SENTIMENT_LABELS},
        )
        .reset_index()
    )

    for s in SENTIMENT_LABELS:
        agg[f"pct_{s}"] = (agg[f"n_{s}"] / agg["n_docs"] * 100).round(1)

    # Dominant sentiment
    pct_cols = [f"pct_{s}" for s in SENTIMENT_LABELS]
    agg["dominant_sentiment"] = agg[pct_cols].idxmax(axis=1).str.replace("pct_", "")

    # Optional mean rating
    if "rating_score" in work.columns:
        rating_agg = work.groupby(topic_col)["rating_score"].mean().round(2).rename("mean_rating")
        agg = agg.merge(rating_agg, left_on=topic_col, right_index=True, how="left")

    # Filter small topics
    agg = agg[agg["n_docs"] >= min_docs].sort_values("n_docs", ascending=False)
    logger.info("Topic-sentiment table built for %d topics", len(agg))
    return agg.reset_index(drop=True)


def plot_topic_sentiment(topic_sent: pd.DataFrame, top_n: int = 15) -> Path:
    """Stacked bar chart of sentiment proportions for the top-N topics."""
    plot_df = topic_sent.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = None
    colors = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#27ae60"}
    for s in SENTIMENT_LABELS:
        vals = plot_df[f"pct_{s}"].values
        ax.barh(plot_df["topic_label"], vals, left=bottom, label=s, color=colors[s])
        bottom = vals if bottom is None else bottom + vals
    ax.set_xlabel("Percentage of Reviews")
    ax.set_title(f"Sentiment Distribution by Topic (Top {top_n})")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    path = FIGURES_DIR / "10_topic_sentiment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure saved → %s", path)
    return path


def identify_priority_topics(
    topic_sent: pd.DataFrame,
    negative_threshold: float = 40.0,
) -> pd.DataFrame:
    """
    Flag topics that deserve operational attention:
    high volume + high negative share.
    """
    priority = topic_sent[
        (topic_sent["pct_negative"] >= negative_threshold)
        | (
            (topic_sent["dominant_sentiment"] == "negative")
            & (topic_sent["n_docs"] >= topic_sent["n_docs"].median())
        )
    ].copy()
    priority = priority.sort_values(["pct_negative", "n_docs"], ascending=[False, False])
    logger.info("Identified %d priority (negative) topics", len(priority))
    return priority


def generate_topic_sentiment_report(
    df: pd.DataFrame,
    topic_sent: Optional[pd.DataFrame] = None,
) -> Path:
    """Write a markdown business insight report for topic-level sentiment."""
    if topic_sent is None:
        topic_sent = compute_topic_sentiment(df)

    plot_topic_sentiment(topic_sent)
    priority = identify_priority_topics(topic_sent)

    report_path = REPORTS_DIR / "topic_sentiment_report.md"
    with open(report_path, "w") as f:
        f.write("# Topic-Level Sentiment Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Topics analysed: **{len(topic_sent)}**\n")
        f.write(f"- Total reviews with assigned topics: **{topic_sent['n_docs'].sum():,}**\n\n")

        f.write("## Top Topics by Volume\n\n")
        cols = ["topic_label", "n_docs", "pct_negative", "pct_neutral", "pct_positive", "dominant_sentiment"]
        if "mean_rating" in topic_sent.columns:
            cols.append("mean_rating")
        f.write(topic_sent[cols].head(12).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Priority Topics (High Negative Sentiment)\n\n")
        if len(priority):
            f.write(priority[cols].to_markdown(index=False))
        else:
            f.write("No topics exceeded the negative-sentiment threshold.\n")
        f.write("\n")

    logger.info("Topic-sentiment report → %s", report_path)
    return report_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    def _load(stem: Path):
        for suffix in (".parquet", ".pkl"):
            path = stem.with_suffix(suffix)
            if path.exists() and path.stat().st_size > 0:
                return pd.read_parquet(path) if suffix == ".parquet" else pd.read_pickle(path)
        return None

    cleaned = _load(Path("data/cleaned_reviews"))
    topics = _load(MODELS_DIR / "document_topics")
    if cleaned is not None and topics is not None:
        merged = cleaned.merge(topics, on="review_id", how="inner")
        ts = compute_topic_sentiment(merged)
        generate_topic_sentiment_report(merged, ts)
        print(ts.head(10))
    else:
        print("Run topic modelling first (python main.py --topics)")
