"""
Objective 4 – Topic Modelling with BERTopic
Discover major themes in guest reviews using transformer embeddings + clustering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

from config.config import BERTOPIC_CONFIG, MODELS_DIR, FIGURES_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


def build_bertopic_model(
    embedding_model_name: str = None,
    min_topic_size: int = None,
    n_gram_range: tuple = None,
    top_n_words: int = None,
) -> BERTopic:
    """
    Construct a BERTopic pipeline with sensible defaults for short hotel reviews.
    """
    cfg = BERTOPIC_CONFIG
    embedding_model_name = embedding_model_name or cfg["embedding_model"]
    min_topic_size = min_topic_size or cfg["min_topic_size"]
    n_gram_range = n_gram_range or cfg["n_gram_range"]
    top_n_words = top_n_words or cfg["top_n_words"]

    logger.info("Loading embedding model: %s", embedding_model_name)
    embedding_model = SentenceTransformer(embedding_model_name)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=n_gram_range,
        min_df=5,
        max_df=0.85,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        calculate_probabilities=cfg["calculate_probabilities"],
        verbose=True,
    )
    return topic_model


def fit_topics(
    documents: list[str],
    model: Optional[BERTopic] = None,
    nr_topics: Optional[int | str] = None,
) -> tuple[BERTopic, list[int], np.ndarray | None]:
    """
    Fit BERTopic on a list of review texts.
    Returns (model, topics, probabilities).
    """
    model = model or build_bertopic_model()
    nr_topics = nr_topics if nr_topics is not None else BERTOPIC_CONFIG["nr_topics"]

    logger.info("Fitting BERTopic on %d documents…", len(documents))
    topics, probs = model.fit_transform(documents)

    if nr_topics != "auto" and isinstance(nr_topics, int):
        logger.info("Reducing to %d topics…", nr_topics)
        model.reduce_topics(documents, nr_topics=nr_topics)
        topics = model.topics_

    logger.info("Discovered %d topics (excl. outliers)", len(set(topics)) - (1 if -1 in topics else 0))
    return model, topics, probs


def get_topic_info(model: BERTopic) -> pd.DataFrame:
    """Return the topic summary table."""
    return model.get_topic_info()


def assign_topic_labels(
    model: BERTopic,
    custom_labels: Optional[dict[int, str]] = None,
) -> dict[int, str]:
    """
    Produce human-readable labels. If custom_labels are supplied they take
    precedence; otherwise the top-3 words are used.
    """
    labels = {}
    for topic_id in model.get_topics():
        if topic_id == -1:
            labels[topic_id] = "Outlier / Mixed"
            continue
        if custom_labels and topic_id in custom_labels:
            labels[topic_id] = custom_labels[topic_id]
        else:
            words = [w for w, _ in model.get_topic(topic_id)[:3]]
            labels[topic_id] = " / ".join(words)
    return labels


def save_topic_model(model: BERTopic, path: Optional[Path] = None) -> Path:
    path = path or (MODELS_DIR / "bertopic_model")
    path = Path(path)
    model.save(str(path), serialization="safetensors", save_ctfidf=True)
    logger.info("BERTopic model saved → %s", path)
    return path


def load_topic_model(path: Optional[Path] = None) -> BERTopic:
    path = path or (MODELS_DIR / "bertopic_model")
    model = BERTopic.load(str(path))
    logger.info("BERTopic model loaded from %s", path)
    return model


def run_topic_modelling(
    df: pd.DataFrame,
    text_col: str = "review_text_clean",
    sample_size: Optional[int] = None,
    save: bool = True,
) -> tuple[BERTopic, pd.DataFrame]:
    """
    End-to-end topic modelling pipeline.
    Adds columns: topic_id, topic_label, topic_prob to a copy of df.
    """
    docs_df = df.copy()
    if sample_size and sample_size < len(docs_df):
        docs_df = docs_df.sample(sample_size, random_state=RANDOM_SEED)
        logger.info("Sampled %d reviews for topic modelling", sample_size)

    documents = docs_df[text_col].tolist()
    model, topics, probs = fit_topics(documents)

    labels = assign_topic_labels(model)
    docs_df["topic_id"] = topics
    docs_df["topic_label"] = [labels.get(t, "Unknown") for t in topics]
    if probs is not None:
        # Probability of the assigned topic
        docs_df["topic_prob"] = [
            float(probs[i][topics[i]]) if topics[i] != -1 and probs is not None else 0.0
            for i in range(len(topics))
        ]
    else:
        docs_df["topic_prob"] = 0.0

    # Persist topic info
    topic_info = get_topic_info(model)
    topic_info["label"] = topic_info["Topic"].map(labels)
    topic_info.to_csv(MODELS_DIR / "topic_info.csv", index=False)

    if save:
        save_topic_model(model)
        # Also save the document-topic mapping for the full (or sampled) set
        docs_df[["review_id", "topic_id", "topic_label", "topic_prob"]].to_pickle(
            MODELS_DIR / "document_topics.pkl", index=False
        )

    # Visualisations (optional – may fail in headless environments)
    try:
        fig = model.visualize_barchart(top_n_topics=12)
        fig.write_image(str(FIGURES_DIR / "08_topic_barchart.png"))
        fig = model.visualize_heatmap()
        fig.write_image(str(FIGURES_DIR / "09_topic_heatmap.png"))
    except Exception as e:
        logger.warning("Could not save interactive topic plots: %s", e)

    return model, docs_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data.extract import load_or_extract
    from src.data.clean import clean_dataset

    raw, _ = load_or_extract()
    cleaned = clean_dataset(raw)
    # Use a sample for faster iteration during development; set to None for full run
    model, docs = run_topic_modelling(cleaned, sample_size=10000)
    print(get_topic_info(model).head(15))
