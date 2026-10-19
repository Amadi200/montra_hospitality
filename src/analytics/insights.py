"""
Guest Experience Analytics & Business Insight Reporting
Combines sentiment, topics, hotels, segments and booking context.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.config import FIGURES_DIR, REPORTS_DIR, SENTIMENT_LABELS

logger = logging.getLogger(__name__)


def hotel_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-hotel average rating, sentiment mix and review volume."""
    hotel_col = "ht_hotel_name" if "ht_hotel_name" in df.columns else "hotel_id"
    loc_col = "ht_location" if "ht_location" in df.columns else None

    agg = (
        df.groupby([hotel_col] + ([loc_col] if loc_col else []))
        .agg(
            n_reviews=("review_id", "count"),
            mean_rating=("rating_score", "mean"),
            pct_positive=("sentiment_label", lambda x: (x == "positive").mean() * 100),
            pct_negative=("sentiment_label", lambda x: (x == "negative").mean() * 100),
        )
        .round(2)
        .reset_index()
        .sort_values("mean_rating")
    )
    return agg


def segment_sentiment_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Sentiment breakdown by guest segment dimensions."""
    results = {}
    for col in ["gt_customer_type", "gt_loyalty_status", "bk_booking_channel", "bk_room_type"]:
        if col not in df.columns:
            continue
        ct = (
            pd.crosstab(df[col], df["sentiment_label"], normalize="index") * 100
        ).round(1)
        ct["n"] = df.groupby(col).size()
        results[col] = ct
    return results


def trend_over_time(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Monthly (or other) sentiment & rating trends."""
    if "review_date" not in df.columns:
        return pd.DataFrame()
    ts = df.set_index("review_date").sort_index()
    monthly = ts.resample(freq).agg(
        n_reviews=("review_id", "count"),
        mean_rating=("rating_score", "mean"),
        pct_positive=("sentiment_label", lambda x: (x == "positive").mean() * 100),
        pct_negative=("sentiment_label", lambda x: (x == "negative").mean() * 100),
    ).round(2)
    return monthly


def plot_hotel_leaderboard(hotel_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    hotel_col = [c for c in hotel_summary.columns if "hotel" in c.lower() or c == "hotel_id"][0]
    plot_df = hotel_summary.sort_values("mean_rating")
    colors = ["#e74c3c" if r < 3.0 else "#f39c12" if r < 3.8 else "#27ae60" for r in plot_df["mean_rating"]]
    ax.barh(plot_df[hotel_col], plot_df["mean_rating"], color=colors)
    ax.set_xlabel("Average Rating")
    ax.set_title("Hotel Rating Leaderboard")
    ax.axvline(plot_df["mean_rating"].mean(), color="grey", linestyle="--", label="Overall mean")
    ax.legend()
    path = FIGURES_DIR / "11_hotel_leaderboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_sentiment_trends(trends: pd.DataFrame) -> Path:
    if trends.empty:
        return Path()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    trends["mean_rating"].plot(ax=axes[0], marker="o", color="#3498db")
    axes[0].set_ylabel("Mean Rating")
    axes[0].set_title("Rating & Sentiment Trends Over Time")
    trends[["pct_positive", "pct_negative"]].plot(ax=axes[1], marker="o")
    axes[1].set_ylabel("% of Reviews")
    axes[1].legend(["Positive %", "Negative %"])
    path = FIGURES_DIR / "12_sentiment_trends.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def generate_business_insight_report(
    df: pd.DataFrame,
    topic_sent: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Produce the final business-level insight report combining
    hotel performance, segments, trends and (optionally) topics.
    """
    logger.info("Generating business insight report…")

    hotel_sum = hotel_sentiment_summary(df)
    segments = segment_sentiment_summary(df)
    trends = trend_over_time(df)

    plot_hotel_leaderboard(hotel_sum)
    plot_sentiment_trends(trends)

    report_path = REPORTS_DIR / "business_insight_report.md"
    with open(report_path, "w") as f:
        f.write("# Guest Experience Business Insight Report\n")
        f.write("## Montra Hospitality Intel\n\n")

        f.write("### 1. Executive Summary\n\n")
        f.write(f"- Total reviews analysed: **{len(df):,}**\n")
        f.write(f"- Overall mean rating: **{df['rating_score'].mean():.2f}**\n")
        sent_pct = (df["sentiment_label"].value_counts(normalize=True) * 100).round(1)
        for s in SENTIMENT_LABELS:
            f.write(f"- {s.capitalize()}: **{sent_pct.get(s, 0)}%**\n")
        f.write("\n")

        f.write("### 2. Hotel Performance Leaderboard\n\n")
        f.write(hotel_sum.to_markdown(index=False))
        f.write("\n\n")

        f.write("### 3. Sentiment by Guest Segment\n\n")
        for name, table in segments.items():
            f.write(f"#### {name}\n\n")
            f.write(table.to_markdown())
            f.write("\n\n")

        if topic_sent is not None and len(topic_sent):
            f.write("### 4. Topic-Level Insights\n\n")
            cols = ["topic_label", "n_docs", "pct_negative", "pct_positive", "dominant_sentiment"]
            if "mean_rating" in topic_sent.columns:
                cols.append("mean_rating")
            f.write(topic_sent[cols].head(15).to_markdown(index=False))
            f.write("\n\n")

            neg = topic_sent[topic_sent["dominant_sentiment"] == "negative"].head(5)
            if len(neg):
                f.write("**Priority improvement areas (dominant negative sentiment):**\n\n")
                for _, row in neg.iterrows():
                    f.write(f"- **{row['topic_label']}** – {row['pct_negative']}% negative "
                            f"({row['n_docs']} reviews)\n")
                f.write("\n")

        f.write("### 5. Recommendations\n\n")
        f.write(
            "1. Focus operational resources on hotels and topics with the highest negative share.\n"
            "2. Investigate root causes behind low-rated properties (staffing, maintenance, cleanliness).\n"
            "3. Share positive-topic best practices across the portfolio.\n"
            "4. Monitor monthly sentiment trends for early warning of emerging issues.\n"
            "5. Segment-specific actions: tailor service recovery for high-value loyalty tiers.\n"
        )

    logger.info("Business insight report → %s", report_path)
    return report_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data.extract import load_or_extract
    from src.data.clean import clean_dataset

    raw, _ = load_or_extract()
    cleaned = clean_dataset(raw)
    generate_business_insight_report(cleaned)
