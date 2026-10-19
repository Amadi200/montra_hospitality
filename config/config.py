"""
Configuration for Guest Experience Analytics & Sentiment Intelligence
Montra Hospitality Intel (UK)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"
MLRUNS_DIR = OUTPUT_DIR / "mlruns"

for d in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, MLRUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://neondb_owner:npg_qStXmZLBQ5A9@ep-wispy-sunset-axezqxvm-pooler.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require",
)

# ---------------------------------------------------------------------------
# Sentiment labels (ground truth already present in reviews.sentiment_label)
# ---------------------------------------------------------------------------
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# NLP / Preprocessing
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MIN_REVIEW_LENGTH = 20          # characters
MAX_REVIEW_LENGTH = 512         # tokens for BERT
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ---------------------------------------------------------------------------
# Topic Modelling (BERTopic)
# ---------------------------------------------------------------------------
BERTOPIC_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "min_topic_size": 50,
    "nr_topics": "auto",          # or an integer
    "top_n_words": 10,
    "n_gram_range": (1, 2),
    "calculate_probabilities": True,
}

# ---------------------------------------------------------------------------
# BERT Sentiment Classification
# ---------------------------------------------------------------------------
BERT_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 128,            # reviews are short (~172 chars)
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}

# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------
BASELINE_MODELS = ["logistic_regression", "naive_bayes", "svm", "random_forest"]

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "guest_experience_sentiment"
MLFLOW_TRACKING_URI = str(MLRUNS_DIR)
