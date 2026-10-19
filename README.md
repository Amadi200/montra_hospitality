# Guest Experience Analytics & Sentiment Intelligence

**Montra Hospitality Intel (UK)**

Transform unstructured guest review text into measurable, actionable customer-experience intelligence.

---

## Jupyter Notebooks (recommended for VS Code)

All pipeline stages are available as notebooks under `notebooks/`:

| Notebook | Description |
|----------|-------------|
| `00_overview.ipynb` | Start here – environment check & guide |
| `01_data_collection.ipynb` | Extract from PostgreSQL |
| `02_data_cleaning.ipynb` | Clean text & build corpus |
| `03_exploratory_data_analysis.ipynb` | EDA charts |
| `04_topic_modeling.ipynb` | BERTopic |
| `05_topic_sentiment.ipynb` | Sentiment per topic |
| `06_baseline_sentiment.ipynb` | Classical ML baselines |
| `07_bert_sentiment.ipynb` | Fine-tune BERT |
| `08_business_insights.ipynb` | Business reports |

Open `notebooks/00_overview.ipynb` first, then run each notebook **in order**.

---

## Project Overview

Montra’s existing platform already reports structured metrics (revenue, occupancy, bookings).  
This project extends it into **customer-experience analytics** by combining:

| Component | Technology |
|-----------|------------|
| Data extraction | PostgreSQL → pandas / SQLAlchemy |
| Cleaning & EDA | pandas, seaborn, matplotlib |
| Topic modelling | BERTopic + Sentence-Transformers |
| Topic-level sentiment | Aggregation over BERTopic topics |
| Sentiment classification | BERT (Hugging Face) + classical baselines |
| Experiment tracking | MLflow |
| Business insights | Automated markdown reports + figures |

### Objectives

1. **Data Collection** – consolidate reviews with bookings, guests and hotels  
2. **Exploratory Data Analysis** – distributions, trends, segment patterns  
3. **Data Cleaning & Preparation** – text cleaning, quality checks  
4. **Topic Modelling** – BERTopic discovers themes (staff, cleanliness, food, location …)  
5. **Topic Sentiment Analysis** – sentiment *per topic*, not only per review  
6. **BERT Sentiment Classification** – Positive / Neutral / Negative, benchmarked against LR, NB, SVM, RF  
7. **ML Experimentation** – systematic comparison tracked with MLflow  

---

## Database Schema

| Table | Key columns | Rows |
|-------|-------------|------|
| `reviews` | review_id, booking_id, guest_id, hotel_id, review_text, rating_score, review_date, sentiment_label | 35 000 |
| `bookings` | booking_id, guest_id, hotel_id, room_type, booking_channel, check_in/out, total_amount, stay_duration | 35 000 |
| `guests` | guest_id, customer_type, loyalty_status, country, previous_stays | 12 000 |
| `hotels` | hotel_id, hotel_name, location, star_rating, total_rooms | 25 |

Connection string is configured in `config/config.py` (or via the `DATABASE_URL` environment variable).

---

## Project Structure

```
guest_experience_analytics/
├── config/
│   └── config.py              # Paths, DB URL, model hyper-parameters
├── src/
│   ├── data/
│   │   ├── extract.py         # Obj 1 – data collection
│   │   └── clean.py           # Obj 3 – cleaning & preparation
│   ├── eda/
│   │   └── explore.py         # Obj 2 – exploratory analysis
│   ├── topic_modeling/
│   │   └── bertopic_model.py  # Obj 4 – BERTopic
│   ├── sentiment/
│   │   ├── baselines.py       # Obj 6 – LR / NB / SVM / RF
│   │   ├── bert_classifier.py # Obj 6 – BERT fine-tuning
│   │   └── topic_sentiment.py # Obj 5 – topic-level sentiment
│   ├── experiments/
│   │   └── mlflow_tracking.py # Obj 7 – MLflow
│   └── analytics/
│       └── insights.py        # Business insight reports
├── data/                      # Parquet extracts & cleaned corpus
├── outputs/
│   ├── models/                # Saved models & topic mappings
│   ├── reports/               # Markdown reports
│   ├── figures/               # EDA & insight plots
│   └── mlruns/                # MLflow tracking store
├── notebooks/                 # Optional interactive notebooks
├── main.py                    # CLI entry-point
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Create environment & install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) download spaCy / NLTK resources if needed later
python -m spacy download en_core_web_sm

# 3. Run individual stages
python main.py --extract          # pull from PostgreSQL
python main.py --clean            # clean + quality checks
python main.py --eda              # figures + eda_report.md
python main.py --topics --topic-sample 10000
python main.py --baselines
python main.py --bert --bert-sample 8000 --epochs 2
python main.py --topic-sentiment
python main.py --insights
python main.py --compare-runs

# Or run everything
python main.py --all --topic-sample 10000 --bert-sample 8000 --epochs 2
```

> **Tip:** On CPU, start with the sample flags. Remove them for full-data training once the pipeline is validated.

---

## Key Outputs

| Path | Description |
|------|-------------|
| `data/cleaned_reviews.parquet` | Cleaned NLP-ready corpus |
| `outputs/models/bertopic_model/` | Persisted BERTopic model |
| `outputs/models/document_topics.parquet` | review_id → topic mapping |
| `outputs/models/baseline_*.joblib` | Classical ML pipelines |
| `outputs/models/bert_sentiment/` | Fine-tuned BERT + tokenizer |
| `outputs/reports/eda_report.md` | EDA summary |
| `outputs/reports/topic_sentiment_report.md` | Topic-level sentiment |
| `outputs/reports/business_insight_report.md` | Executive insight report |
| `outputs/figures/*.png` | All visualisations |
| `outputs/mlruns/` | MLflow experiment history |

---

## Modelling Notes

- **Reviews are short** (~172 characters). BERT `max_length=128` is sufficient.
- Ground-truth `sentiment_label` already exists in the database (positive / neutral / negative).
- BERTopic uses `all-MiniLM-L6-v2` embeddings + HDBSCAN; `min_topic_size=50` works well for 35 k reviews.
- Baselines use TF-IDF (1-2 grams, 15 k features) + class-weighted classifiers.
- All experiments are logged to a local MLflow store under `outputs/mlruns`.

---

## Out of Scope (by design)

- External review scraping (TripAdvisor, Booking.com, …)
- Real-time / streaming inference
- Production API / MLOps deployment
- Non-English reviews
- Emotion or sarcasm detection beyond the three-class schema

---

## Licence & Contact

Internal project – Montra Hospitality Intel.  
For questions about the analytics pipeline contact the data-science team.
