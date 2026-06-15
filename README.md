# Customer Interaction Multi-Label Classification

Refactored multi-label classifier for customer-support tickets. Predicts
three label levels (`y2`, `y3`, `y4`) using a cascaded classifier and
reports the **chained accuracy** required by the assessment brief.

## Project layout

```
project/
  src/
    config.py                       Paths, columns, hyperparameters.
    data_loader.py                  CSV loading, schema validation, label tidy.
    preprocessing.py                Deduplication + cleaning (shared with inference).
    features.py                     TF-IDF with serialisation.
    metrics.py                      Chained accuracy + standard metrics.
    train.py                        Training entry point.
    predict.py                      Batch inference entry point.
    logging_utils.py                Logger config.
    make_diagrams.py                Generates the architecture diagrams.
    models/
      base_model.py                 Abstract single-label classifier interface.
      random_forest_model.py        Random Forest wrapper.
      logistic_regression_model.py  Logistic Regression wrapper.
      chained_classifier.py         y2 -> y3 -> y4 cascade.
  data/
    AppGallery.csv                  Provided dataset (Type 1 = AppGallery & Games).
    Purchasing.csv                  Provided dataset (Type 1 = In-App Purchase).
    new_messages.csv                Sample input for batch inference.
  artifacts/                        Saved model + vectorizer (created by train.py).
  outputs/                          Metrics, predictions, error analysis, confusion matrices, logs.
  tests/
    test_chained_accuracy.py        Unit tests reproducing the brief's examples (A..E).
  diagrams/                         Architecture diagrams.
  requirements.txt
  README.md
```

## Install

```bash
pip install -r requirements.txt
```

Python 3.10 or newer recommended.

## Run the training pipeline

```bash
python -m src.train --model all
```

This will:

1. Load `AppGallery.csv` and `Purchasing.csv` from `data/`.
2. Validate columns, drop rows with missing `y2`, sentinel-token any missing `y3` / `y4`.
3. Deduplicate quoted email chains, then strip noise.
4. Stratified 80/20 train-test split.
5. Fit TF-IDF on the **training texts only** (so there's no leakage into the test set).
6. Train two chained classifiers: Random Forest and Logistic Regression.
7. Evaluate using:
   - per-label accuracy, precision, recall, macro F1
   - chained accuracy (Type 3 gated by Type 2, Type 4 gated by both, as in the brief)
   - per-level confusion matrices saved as PNG.
8. Save artefacts (`artifacts/`) and outputs (`outputs/`).

Single-model runs:

```bash
python -m src.train --model rf
python -m src.train --model lr
```

## Run batch inference

```bash
python -m src.predict --input data/new_messages.csv \
                     --output outputs/predictions.csv \
                     --model rf
```

Output columns: `message_id`, `input_summary`, `y2_predicted`, `y2_confidence`,
`y3_predicted`, `y3_confidence`, `y4_predicted`, `y4_confidence`, `model_name`,
`model_version`, `prediction_timestamp_utc`.

The prediction script reloads the vectorizer and classifier saved by training
and uses the same `preprocess_message()` function the trainer used. This
guarantees training-serving consistency by construction.

## Run the tests

```bash
python -m unittest discover tests -v
```

The chained-accuracy tests reproduce the five worked examples (A, B, C, D, E)
in the assessment brief.

## Generated outputs

After `train.py` finishes, `outputs/` will contain:

| File                              | Purpose                                                   |
| --------------------------------- | --------------------------------------------------------- |
| `metrics.json`                    | All per-model metric summaries (machine readable).        |
| `classification_reports.csv`      | Per-class precision/recall/F1 across all levels.          |
| `chained_accuracy.csv`            | Where in the chain each model's accuracy is lost.         |
| `test_predictions.csv`            | Full test-split predictions with true labels.             |
| `error_analysis.csv`              | Misclassification examples for the report.                |
| `predictions.csv`                 | Output of the most recent `predict.py` run.               |
| `confusion_matrices/*.png`        | One confusion matrix per (model, level).                  |
| `logs/training.log`               | Full training log.                                        |
| `logs/prediction.log`             | Full prediction log.                                      |

## How chained accuracy works

For each row, chained accuracy is the fraction of labels in the chain
`(y2, y3, y4)` that are correct **up to the first mistake**. Once the
chain breaks, later labels score 0 even if they match the ground truth,
because in production we'd never have predicted them correctly without
the right parent.

Examples (true = `Suggestion`, `Payment`, `Subscription Cancelled`):

| Pred y2  | Pred y3 | Pred y4               | Chained acc |
| -------- | ------- | --------------------- | ----------- |
| Suggest. | Payment | Subscription Cancelled | 1.00       |
| Suggest. | Payment | Subscription Retained  | 0.67       |
| Suggest. | Refund  | Subscription Retained  | 0.33       |
| Other    | Payment | Subscription Cancelled | 0.00       |
| Suggest. | Refund  | Subscription Cancelled | 0.33       |

See `src/metrics.py` and `tests/test_chained_accuracy.py`.
