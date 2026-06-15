"""
Training pipeline.

Loads the CSVs, preprocesses, splits, fits TF-IDF on the train partition,
trains both chained classifiers, evaluates on the held-out test set, and
writes everything to artifacts/ and outputs/.

Run with:
    python -m src.train               # trains both RF and LR
    python -m src.train --model rf
    python -m src.train --model lr
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS, ARTIFACTS_DIR, COLUMNS, DATA, LOGS_DIR, OUTPUTS_DIR,
)
from src.data_loader import data_summary, filter_for_training, load_raw_dataset
from src.features import fit_vectorizer, save_vectorizer, transform_texts
from src.logging_utils import get_logger
from src.metrics import evaluate_model, save_confusion_matrices
from src.models import (
    ChainedHierarchicalClassifier, LogisticRegressionModel, RandomForestModel,
)
from src.preprocessing import preprocess_dataframe

log = get_logger(__name__, log_file=LOGS_DIR / "training.log")


def _stratification_key(df):
    """
    Build a stratification key for train_test_split. Any class with fewer
    than DATA.min_samples_per_class samples gets folded into a single
    '__rare__' bucket - otherwise sklearn refuses to stratify.
    """
    counts = df["y2"].value_counts()
    rare = counts[counts < DATA.min_samples_per_class].index
    return df["y2"].where(~df["y2"].isin(rare), other="__rare__")


def split_data(df):
    strat = _stratification_key(df)
    # if even the most common class has fewer than 2 samples we can't stratify at all
    if strat.value_counts().min() < 2:
        log.warning("Falling back to non-stratified split")
        return train_test_split(df, test_size=DATA.test_size,
                                random_state=DATA.random_state)

    train_df, test_df = train_test_split(
        df, test_size=DATA.test_size,
        random_state=DATA.random_state, stratify=strat,
    )
    log.info("Split sizes: train=%d test=%d", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_label_matrix(df):
    # shape (n_samples, 3) of strings
    return df[list(COLUMNS.label_cols)].to_numpy().astype(str)


def train_models(model_choice="all"):
    log.info("=" * 70)
    log.info("Training pipeline start")
    log.info("=" * 70)

    # 1. load + filter
    raw = load_raw_dataset()
    raw = filter_for_training(raw)
    summary = data_summary(raw)
    log.info("Data summary: %s", json.dumps(summary, indent=2))

    # 2. preprocess text
    processed = preprocess_dataframe(raw)

    # 3. train/test split
    train_df, test_df = split_data(processed)

    # 4. fit TF-IDF on TRAIN texts only (this is the leakage fix)
    vec = fit_vectorizer(train_df["text_combined"].tolist())
    save_vectorizer(vec)
    X_train = transform_texts(vec, train_df["text_combined"].tolist())
    X_test = transform_texts(vec, test_df["text_combined"].tolist())
    Y_train = build_label_matrix(train_df)
    Y_test = build_label_matrix(test_df)

    # 5. pick which models to train
    factories = {
        "rf": lambda: ChainedHierarchicalClassifier(
            [RandomForestModel(), RandomForestModel(), RandomForestModel()],
            level_names=list(COLUMNS.label_cols),
        ),
        "lr": lambda: ChainedHierarchicalClassifier(
            [LogisticRegressionModel(), LogisticRegressionModel(), LogisticRegressionModel()],
            level_names=list(COLUMNS.label_cols),
        ),
    }
    if model_choice == "all":
        chosen = ["rf", "lr"]
    elif model_choice in factories:
        chosen = [model_choice]
    else:
        raise ValueError(f"Unknown model choice: {model_choice!r}")

    all_summaries = {"data_summary": summary, "models": {}}
    all_class_reports = []
    all_chain_breakdowns = []
    all_predictions = []

    for key in chosen:
        log.info("-" * 70)
        log.info("Training model: %s", key)
        log.info("-" * 70)

        clf = factories[key]()
        clf.fit(X_train, Y_train)

        # save the trained cascade
        clf_path = ARTIFACTS_DIR / ARTIFACTS.classifier_template.format(model_name=key)
        clf.save(clf_path)

        # evaluate on the held-out set
        y_pred, confs = clf.predict_with_confidence(X_test)
        y_true_chain = [Y_test[:, lvl] for lvl in range(Y_test.shape[1])]
        y_pred_chain = [y_pred[:, lvl] for lvl in range(y_pred.shape[1])]

        summ, per_class, breakdown = evaluate_model(
            y_true_chain, y_pred_chain, list(COLUMNS.label_cols), model_name=key,
        )
        summ["model_artefact"] = str(clf_path.relative_to(ARTIFACTS_DIR.parent))
        all_summaries["models"][key] = summ

        per_class["model"] = key
        breakdown["model"] = key
        all_class_reports.append(per_class)
        all_chain_breakdowns.append(breakdown)

        # confusion matrices
        save_confusion_matrices(y_true_chain, y_pred_chain,
                                list(COLUMNS.label_cols), model_name=key)

        # save the full per-row predictions for the error analysis
        preds_df = pd.DataFrame({
            "ticket_id": test_df[COLUMNS.ticket_id].values,
            "ticket_summary": test_df[COLUMNS.ticket_summary].values,
            "text_combined": test_df["text_combined"].values,
            "y2_true": Y_test[:, 0], "y3_true": Y_test[:, 1], "y4_true": Y_test[:, 2],
            "y2_pred": y_pred[:, 0], "y3_pred": y_pred[:, 1], "y4_pred": y_pred[:, 2],
            "y2_conf": confs[:, 0], "y3_conf": confs[:, 1], "y4_conf": confs[:, 2],
            "model": key,
        })
        all_predictions.append(preds_df)

        log.info(
            "Model %s | chained_acc=%.3f | y2_f1=%.3f | y3_f1=%.3f | y4_f1=%.3f",
            key, summ["chained_accuracy"],
            summ["per_level_metrics"][0]["f1_macro"],
            summ["per_level_metrics"][1]["f1_macro"],
            summ["per_level_metrics"][2]["f1_macro"],
        )

    # ----- persist outputs -----
    (OUTPUTS_DIR / ARTIFACTS.metrics_json).write_text(
        json.dumps(all_summaries, indent=2, default=float)
    )
    log.info("Wrote %s", OUTPUTS_DIR / ARTIFACTS.metrics_json)

    pd.concat(all_class_reports, ignore_index=True).to_csv(
        OUTPUTS_DIR / ARTIFACTS.classification_reports_csv, index=False,
    )
    pd.concat(all_chain_breakdowns, ignore_index=True).to_csv(
        OUTPUTS_DIR / ARTIFACTS.chained_accuracy_csv, index=False,
    )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUTPUTS_DIR / "test_predictions.csv", index=False,
    )

    _write_error_analysis(all_predictions)
    log.info("Training pipeline complete.")
    return all_summaries


def _write_error_analysis(prediction_dfs):
    """Pick out a handful of misclassifications per model for the report."""
    if not prediction_dfs:
        return
    rows = []
    for df in prediction_dfs:
        model = df["model"].iloc[0]
        wrong = df[
            (df["y2_pred"] != df["y2_true"])
            | ((df["y3_pred"] != df["y3_true"]) & (df["y3_true"] != DATA.missing_label_token))
            | ((df["y4_pred"] != df["y4_true"]) & (df["y4_true"] != DATA.missing_label_token))
        ].copy()
        for _, r in wrong.head(10).iterrows():
            rows.append({
                "model": model,
                "ticket_id": r["ticket_id"],
                "message_excerpt": (r["ticket_summary"] or r["text_combined"])[:120],
                "y2_true": r["y2_true"], "y2_pred": r["y2_pred"],
                "y3_true": r["y3_true"], "y3_pred": r["y3_pred"],
                "y4_true": r["y4_true"], "y4_pred": r["y4_pred"],
                "y2_conf": round(float(r["y2_conf"]), 3),
            })
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / ARTIFACTS.error_analysis_csv, index=False)
    log.info("Wrote error analysis table with %d rows", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the chained multi-label classifier.")
    ap.add_argument("--model", choices=["rf", "lr", "all"], default="all")
    args = ap.parse_args()
    train_models(args.model)
