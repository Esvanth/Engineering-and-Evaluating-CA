---
Title: Customer Support Ticket Classifier

Student ID: Esvanth Mohankumar - x24311073@student.ncirl.ie , Karthicksubramanian Muthukkaruppan - x25191489@student.ncirl.ie

A hierarchical multi-label classifier for customer-support tickets, built as a continuous assessment for the **MSc AI — Engineering and Evaluating AI Systems** module at **NCI**. It predicts three label levels using a chained cascade of classifiers:

| Level | Name | Examples |
|-------|------|----------|
| **Type 2** | Top-level category | Suggestion, Problem-Fault, Others |
| **Type 3** | Mid-level category | Payment, Refund, AppGallery-Install/Upgrade |
| **Type 4** | Leaf-level category | Subscription cancellation, Can't install Apps |

## Architecture

The system uses a **chained hierarchical classifier**. Each level in the hierarchy is an independent scikit-learn model, but each level receives the original TF-IDF text features **plus** the one-hot encoded predictions from all preceding levels:

```
Level 0 (y2):  features = TF-IDF(text)
Level 1 (y3):  features = TF-IDF(text) + one-hot(y2)
Level 2 (y4):  features = TF-IDF(text) + one-hot(y2) + one-hot(y3)
```

At training time the cascade uses the **true** parent labels; at inference time it uses its **own predicted** parent labels — which is exactly what the custom chained accuracy metric evaluates.

Two model families are trained and available for comparison:
- **Random Forest** (300 estimators, max depth 30, balanced subsample weighting)
- **Logistic Regression** (L2 regularisation, balanced class weighting)

### Text preprocessing pipeline

1. Deduplicate quoted email chains within each ticket
2. Remove customer-support boilerplate phrases (multilingual)
3. Strip noise: greetings, dates, email addresses, anonymised tokens, punctuation
4. Combine cleaned summary + cleaned content into a single text field
5. Vectorise via **TF-IDF** (up to 2 000 features, 1–2 ngrams, sublinear TF)

The same preprocessing functions are shared between training and inference to prevent training-serving skew.

## Project structure

```
├── app.py                          # Gradio web UI (single message + batch CSV)
├── requirements.txt                # Python dependencies
├── data/
│   ├── AppGallery.csv              # Raw ticket data (AppGallery domain)
│   ├── Purchasing.csv              # Raw ticket data (Purchasing domain)
│   └── new_messages.csv            # Sample input for batch prediction
├── src/
│   ├── config.py                   # Paths, column names, hyperparameters
│   ├── data_loader.py              # CSV loading, schema validation, label cleaning
│   ├── preprocessing.py            # Text cleaning and deduplication
│   ├── features.py                 # TF-IDF vectoriser (fit / save / load / transform)
│   ├── train.py                    # Full training pipeline (load → split → fit → evaluate)
│   ├── predict.py                  # Batch inference CLI
│   ├── metrics.py                  # Chained accuracy, per-level metrics, confusion matrices
│   ├── logging_utils.py            # Logging configuration
│   ├── make_diagrams.py            # Diagram generation utilities
│   └── models/
│       ├── base_model.py           # Abstract base classifier interface
│       ├── random_forest_model.py  # Random Forest wrapper
│       ├── logistic_regression_model.py  # Logistic Regression wrapper
│       └── chained_classifier.py   # Chained hierarchical cascade (fit / predict / save / load)
├── artifacts/                      # Trained model artefacts (vectoriser + classifiers)
├── outputs/                        # Evaluation outputs (metrics, reports, confusion matrices)
│   ├── metrics.json
│   ├── classification_reports.csv
│   ├── chained_accuracy.csv
│   ├── test_predictions.csv
│   ├── error_analysis.csv
│   ├── confusion_matrices/         # Per-model, per-level confusion matrix PNGs
│   └── logs/                       # Training and prediction logs
└── tests/
    └── test_chained_accuracy.py    # Unit tests for the chained accuracy metric
```

## Evaluation on the held-out test set (42 rows)

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| **Chained accuracy** | 0.683 | **0.694** |
| Type 2 macro F1 | **0.833** | 0.769 |
| Type 3 macro F1 | 0.553 | 0.583 |
| Type 4 macro F1 | 0.537 | 0.448 |

The **chained accuracy** metric is specific to this assignment: once the model gets a label wrong at any level, all subsequent levels in that row score zero — even if they happen to match the ground truth. This reflects the real-world constraint that correct downstream predictions depend on correct upstream ones.

## Getting started

### Prerequisites

- Python 3.11+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

For the Gradio web UI, also install Gradio:

```bash
pip install gradio>=4.0
```

### Training

Train both models (Random Forest + Logistic Regression):

```bash
python -m src.train
```

Or train a single model:

```bash
python -m src.train --model rf
python -m src.train --model lr
```

This reads from `data/`, fits the TF-IDF vectoriser on the training split only, trains the chained classifiers, evaluates on the held-out test set, and writes all artefacts to `artifacts/` and evaluation outputs to `outputs/`.

### Batch prediction

```bash
python -m src.predict --input data/new_messages.csv --output outputs/predictions.csv --model rf
```

The input CSV must contain `Ticket Summary` and `Interaction content` columns.

### Web UI

```bash
python app.py
```

Opens a Gradio interface at `http://localhost:7860` with two modes:

- **Single Message** — enter a ticket summary and content, choose a model, and click Classify. Results show the predicted label and confidence for each level.
- **Batch CSV** — upload a CSV with the required columns, get a predictions CSV with per-level confidence scores.

### Tests

```bash
python -m pytest tests/
```

Runs the unit tests for the chained accuracy metric, covering the five worked examples from the assignment brief.

## Limitations

- Trained on only **206 samples** with an 80/20 split. Predictions on out-of-domain text will be unreliable.
- The dataset covers only AppGallery and Purchasing ticket domains.
