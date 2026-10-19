# Run this project in VS Code

## 1. Open the folder

1. Unzip `guest_experience_analytics.zip`
2. In VS Code: **File → Open Folder…** → select `guest_experience_analytics`

## 2. Create a virtual environment

Open the integrated terminal (`Ctrl+`` `) and run:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Select the interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Optional (for spaCy, if you extend preprocessing later):

```bash
python -m spacy download en_core_web_sm
```

## 4. Database connection

The Neon PostgreSQL URL is already set in `config/config.py`.

To override it (recommended for secrets), create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+psycopg2://neondb_owner:YOUR_PASSWORD@ep-wispy-sunset-axezqxvm-pooler.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require
```

`config/config.py` already reads `DATABASE_URL` from the environment.

## 5. Run the pipeline

From the project root (with `.venv` activated):

```bash
# Stage by stage (recommended first time)
python main.py --extract
python main.py --clean
python main.py --eda
python main.py --baselines
python main.py --topics --topic-sample 10000
python main.py --topic-sentiment
python main.py --bert --bert-sample 8000 --epochs 2
python main.py --insights
python main.py --compare-runs

# Or everything at once (use samples on CPU)
python main.py --all --topic-sample 10000 --bert-sample 8000 --epochs 2
```

### Tips for CPU machines

- Start with `--topic-sample 5000` and `--bert-sample 4000`
- BERT is much faster with a GPU (`torch.cuda.is_available()`)
- Full 35k BERT fine-tuning can take a long time on CPU

## 6. Where outputs land

| Path | Contents |
|------|----------|
| `data/` | Raw extracts + cleaned corpus |
| `outputs/figures/` | EDA and insight charts (PNG) |
| `outputs/reports/` | Markdown reports |
| `outputs/models/` | Saved baselines, BERTopic, BERT |
| `outputs/mlruns/` | MLflow experiment history |

## 7. VS Code extras (optional)

- Install extensions: **Python**, **Jupyter**, **Pylance**
- For notebooks: create files under `notebooks/` and use the same `.venv` kernel
- Debug: set breakpoints in any `src/` module and run `main.py` with the debugger

## Quick smoke test

```bash
python main.py --extract --clean --eda --baselines
```

You should see ~35,000 reviews extracted, cleaned, figures under `outputs/figures/`, and baseline metrics around **93–94% accuracy**.
