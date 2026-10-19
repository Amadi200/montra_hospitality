#!/usr/bin/env python3
"""
Guest Experience Analytics & Sentiment Intelligence
Montra Hospitality Intel (UK)

End-to-end pipeline covering all seven project objectives:

  1. Data Collection
  2. Exploratory Data Analysis
  3. Data Cleaning & Preparation
  4. Topic Modelling (BERTopic)
  5. Topic-level Sentiment Analysis
  6. BERT Sentiment Classification (+ baselines)
  7. Machine Learning Experimentation (MLflow)

Usage
-----
  # Full pipeline (may take a long time for BERT + BERTopic on full data)
  python main.py --all

  # Individual stages
  python main.py --extract
  python main.py --clean
  python main.py --eda
  python main.py --topics [--topic-sample 10000]
  python main.py --baselines
  python main.py --bert [--bert-sample 8000] [--epochs 3]
  python main.py --topic-sentiment
  python main.py --insights
  python main.py --compare-runs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_DIR, MODELS_DIR, RANDOM_SEED, BERT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def cmd_extract(force: bool = False):
    from src.data.extract import load_or_extract
    df, raw = load_or_extract(force_refresh=force)
    logger.info("Consolidated dataset: %s", df.shape)
    return df


def cmd_clean(force_extract: bool = False):
    from src.data.extract import load_or_extract
    from src.data.clean import clean_dataset, run_data_quality_checks, save_cleaned

    raw, _ = load_or_extract(force_refresh=force_extract)
    cleaned = clean_dataset(raw)
    checks = run_data_quality_checks(cleaned)
    save_cleaned(cleaned)
    logger.info("Quality checks: %s", checks)
    return cleaned


def cmd_eda():
    from src.data.clean import clean_dataset
    from src.data.extract import load_or_extract
    from src.eda.explore import generate_eda_report

    import pandas as pd
    cleaned = None
    for name in ("cleaned_reviews.parquet", "cleaned_reviews.pkl"):
        p = DATA_DIR / name
        if p.exists() and p.stat().st_size > 0:
            cleaned = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_pickle(p)
            break
    if cleaned is None:
        raw, _ = load_or_extract()
        cleaned = clean_dataset(raw)

    report = generate_eda_report(cleaned)
    logger.info("EDA complete → %s", report)


def cmd_topics(sample_size: int | None = None):
    from src.data.clean import clean_dataset
    from src.data.extract import load_or_extract
    from src.topic_modeling.bertopic_model import run_topic_modelling, get_topic_info

    import pandas as pd
    cleaned = None
    for name in ("cleaned_reviews.parquet", "cleaned_reviews.pkl"):
        p = DATA_DIR / name
        if p.exists() and p.stat().st_size > 0:
            cleaned = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_pickle(p)
            break
    if cleaned is None:
        raw, _ = load_or_extract()
        cleaned = clean_dataset(raw)

    model, docs = run_topic_modelling(cleaned, sample_size=sample_size)
    info = get_topic_info(model)
    logger.info("Top topics:\n%s", info.head(12).to_string())
    return model, docs


def cmd_baselines():
    from src.data.clean import clean_dataset
    from src.data.extract import load_or_extract
    from src.sentiment.baselines import run_all_baselines
    from src.experiments.mlflow_tracking import run_experiment_suite

    import pandas as pd
    cleaned = None
    for name in ("cleaned_reviews.parquet", "cleaned_reviews.pkl"):
        p = DATA_DIR / name
        if p.exists() and p.stat().st_size > 0:
            cleaned = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_pickle(p)
            break
    if cleaned is None:
        raw, _ = load_or_extract()
        cleaned = clean_dataset(raw)

    results = run_all_baselines(cleaned)
    run_experiment_suite(baseline_results=results)
    for r in results:
        logger.info("%s test metrics: %s", r["model_name"], r["test_metrics"])
    return results


def cmd_bert(sample_size: int | None = None, epochs: int | None = None):
    from src.data.clean import clean_dataset
    from src.data.extract import load_or_extract
    from src.sentiment.bert_classifier import prepare_bert_splits, train_bert
    from src.experiments.mlflow_tracking import log_bert_run
    import pandas as pd

    cleaned_path = DATA_DIR / "cleaned_reviews.parquet"
    if cleaned_path.exists():
        cleaned = pd.read_parquet(cleaned_path) if cleaned_path.suffix == ".parquet" else pd.read_pickle(cleaned_path)
    else:
        raw, _ = load_or_extract()
        cleaned = clean_dataset(raw)

    if sample_size:
        cleaned = cleaned.sample(min(sample_size, len(cleaned)), random_state=RANDOM_SEED)
        logger.info("Using sample of %d reviews for BERT", len(cleaned))

    splits = prepare_bert_splits(cleaned)
    epochs = epochs or BERT_CONFIG["epochs"]
    trainer, results = train_bert(splits, epochs=epochs)

    log_bert_run(
        params={
            "model_name": BERT_CONFIG["model_name"],
            "max_length": BERT_CONFIG["max_length"],
            "batch_size": BERT_CONFIG["batch_size"],
            "learning_rate": BERT_CONFIG["learning_rate"],
            "epochs": epochs,
            "sample_size": len(cleaned),
        },
        metrics={"test": results["test_metrics"]},
        model_dir=results.get("model_path"),
    )
    logger.info("BERT test metrics: %s", results["test_metrics"])
    return results


def cmd_topic_sentiment():
    import pandas as pd
    from src.sentiment.topic_sentiment import (
        compute_topic_sentiment,
        generate_topic_sentiment_report,
    )

    cleaned = None
    for name in ("cleaned_reviews.parquet", "cleaned_reviews.pkl"):
        p = DATA_DIR / name
        if p.exists() and p.stat().st_size > 0:
            import pandas as pd
            cleaned = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_pickle(p)
            break
    if cleaned is None:
        raise FileNotFoundError("Run --clean first")
    topics = None
    for name in ("document_topics.parquet", "document_topics.pkl"):
        p = MODELS_DIR / name
        if p.exists() and p.stat().st_size > 0:
            import pandas as pd
            topics = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_pickle(p)
            break
    if topics is None:
        raise FileNotFoundError("Run --topics first")
    merged = cleaned.merge(topics, on="review_id", how="inner")
    ts = compute_topic_sentiment(merged)
    report = generate_topic_sentiment_report(merged, ts)
    logger.info("Topic-sentiment report → %s", report)
    return ts


def cmd_insights():
    import pandas as pd
    from src.analytics.insights import generate_business_insight_report

    cleaned = None
    for name in ("cleaned_reviews.parquet", "cleaned_reviews.pkl"):
        p = DATA_DIR / name
        if p.exists() and p.stat().st_size > 0:
            import pandas as pd
            cleaned = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_pickle(p)
            break
    if cleaned is None:
        raise FileNotFoundError("Run --clean first")
    topic_sent = None
    topic_path = MODELS_DIR / "document_topics.parquet"  # also checks .pkl below
    if topic_path.exists():
        from src.sentiment.topic_sentiment import compute_topic_sentiment
        topics = None
        for name in ("document_topics.parquet", "document_topics.pkl"):
            p = MODELS_DIR / name
            if p.exists() and p.stat().st_size > 0:
                topics = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_pickle(p)
                break
        merged = cleaned.merge(topics, on="review_id", how="inner")
        topic_sent = compute_topic_sentiment(merged)

    report = generate_business_insight_report(cleaned, topic_sent)
    logger.info("Business insight report → %s", report)


def cmd_compare_runs():
    from src.experiments.mlflow_tracking import compare_runs
    comparison = compare_runs()
    if not comparison.empty:
        print(comparison.to_string())
    else:
        logger.info("No MLflow runs found yet.")


def cmd_all(args):
    """Run the complete pipeline in order."""
    logger.info("=" * 60)
    logger.info("FULL PIPELINE START")
    logger.info("=" * 60)

    cmd_extract(force=args.force)
    cmd_clean()
    cmd_eda()
    cmd_topics(sample_size=args.topic_sample)
    cmd_baselines()
    cmd_bert(sample_size=args.bert_sample, epochs=args.epochs)
    cmd_topic_sentiment()
    cmd_insights()
    cmd_compare_runs()

    logger.info("=" * 60)
    logger.info("FULL PIPELINE COMPLETE")
    logger.info("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Guest Experience Analytics & Sentiment Intelligence – Montra Hospitality Intel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--all", action="store_true", help="Run the full end-to-end pipeline")
    p.add_argument("--extract", action="store_true", help="Extract data from PostgreSQL")
    p.add_argument("--clean", action="store_true", help="Clean and prepare the dataset")
    p.add_argument("--eda", action="store_true", help="Run exploratory data analysis")
    p.add_argument("--topics", action="store_true", help="Run BERTopic topic modelling")
    p.add_argument("--baselines", action="store_true", help="Train baseline sentiment models")
    p.add_argument("--bert", action="store_true", help="Fine-tune BERT sentiment classifier")
    p.add_argument("--topic-sentiment", action="store_true", help="Compute topic-level sentiment")
    p.add_argument("--insights", action="store_true", help="Generate business insight report")
    p.add_argument("--compare-runs", action="store_true", help="Compare MLflow experiment runs")

    p.add_argument("--force", action="store_true", help="Force re-extraction from database")
    p.add_argument("--topic-sample", type=int, default=None, help="Sample size for BERTopic (default: all)")
    p.add_argument("--bert-sample", type=int, default=None, help="Sample size for BERT training (default: all)")
    p.add_argument("--epochs", type=int, default=None, help="BERT fine-tuning epochs")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # If no stage flags, show help
    stage_flags = [
        args.all, args.extract, args.clean, args.eda, args.topics,
        args.baselines, args.bert, args.topic_sentiment, args.insights, args.compare_runs,
    ]
    if not any(stage_flags):
        parser.print_help()
        return

    if args.all:
        cmd_all(args)
        return

    if args.extract:
        cmd_extract(force=args.force)
    if args.clean:
        cmd_clean(force_extract=args.force)
    if args.eda:
        cmd_eda()
    if args.topics:
        cmd_topics(sample_size=args.topic_sample)
    if args.baselines:
        cmd_baselines()
    if args.bert:
        cmd_bert(sample_size=args.bert_sample, epochs=args.epochs)
    if args.topic_sentiment:
        cmd_topic_sentiment()
    if args.insights:
        cmd_insights()
    if args.compare_runs:
        cmd_compare_runs()


if __name__ == "__main__":
    main()
