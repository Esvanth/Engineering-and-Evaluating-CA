"""
Evaluation metrics.

The main thing here is the chained accuracy described in the assignment
brief. For each row we have a chain of labels (y2, y3, y4). The rule is:
once we get a label wrong, the rest of the chain is automatically "wrong"
even if it happens to match the ground truth, because in production we
would never have predicted those later labels correctly without the right
parent.

Examples (true = Suggestion / Payment / Subscription Cancelled):
    Predicted              | Chained acc
    -----------------------+------------
    same / same / same     | 1.00
    same / same / wrong    | 0.67
    same / wrong / wrong   | 0.33
    wrong / same / same    | 0.00  (chain breaks at level 0)
    same / wrong / same    | 0.33  (level 2 "correct" doesn't count)
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from src.config import ARTIFACTS, DATA, OUTPUTS_DIR
from src.logging_utils import get_logger

log = get_logger(__name__)


# --- Chained accuracy -------------------------------------------------------
def chained_accuracy_per_row(y_true_chain, y_pred_chain,
                             missing_token=DATA.missing_label_token):
    """
    Return a list of chained accuracy scores, one per row.

    y_true_chain and y_pred_chain are sequences of label arrays - one array
    per level in the hierarchy. So for our task they'd be [y2_arr, y3_arr,
    y4_arr], each of length n_samples.
    """
    n_levels = len(y_true_chain)
    if n_levels == 0:
        raise ValueError("Need at least one label level")
    n_samples = len(y_true_chain[0])

    # sanity check that all arrays match in length
    for arr in list(y_true_chain) + list(y_pred_chain):
        if len(arr) != n_samples:
            raise ValueError("All label arrays must have the same length")

    scores = []
    for i in range(n_samples):
        correct = 0
        applicable = 0       # how many levels have a non-missing ground truth
        broken = False       # set once the chain breaks for this row
        for level in range(n_levels):
            true_v = str(y_true_chain[level][i])
            pred_v = str(y_pred_chain[level][i])
            if true_v == missing_token:
                # this row has no ground truth at this level - skip
                continue
            applicable += 1
            if broken:
                # chain already broken at an earlier level - no credit
                continue
            if true_v == pred_v:
                correct += 1
            else:
                broken = True
        scores.append(correct / applicable if applicable > 0 else 0.0)
    return scores


def chained_accuracy(y_true_chain, y_pred_chain,
                     missing_token=DATA.missing_label_token):
    """Mean chained accuracy across all rows."""
    per_row = chained_accuracy_per_row(y_true_chain, y_pred_chain, missing_token)
    return float(np.mean(per_row)) if per_row else 0.0


def chained_accuracy_breakdown(y_true_chain, y_pred_chain, level_names,
                               missing_token=DATA.missing_label_token):
    """
    Break the chained accuracy down by level so we can see exactly *where*
    the chain breaks for each row. Useful for explaining results in the report.
    """
    n_levels = len(y_true_chain)
    n_samples = len(y_true_chain[0])
    rows = []

    # track per-row state across levels
    state = ["pending"] * n_samples

    for level in range(n_levels):
        counts = {"correct": 0, "broken_here": 0,
                  "broken_earlier": 0, "not_applicable": 0}
        for i in range(n_samples):
            true_v = str(y_true_chain[level][i])
            pred_v = str(y_pred_chain[level][i])
            if true_v == missing_token:
                counts["not_applicable"] += 1
                continue
            if state[i] in ("broken_here", "broken_earlier"):
                state[i] = "broken_earlier"
                counts["broken_earlier"] += 1
                continue
            if true_v == pred_v:
                counts["correct"] += 1
            else:
                state[i] = "broken_here"
                counts["broken_here"] += 1

        applicable = counts["correct"] + counts["broken_here"] + counts["broken_earlier"]
        rows.append({
            "level": level_names[level],
            "correct": counts["correct"],
            "broken_at_this_level": counts["broken_here"],
            "broken_at_earlier_level": counts["broken_earlier"],
            "not_applicable": counts["not_applicable"],
            "applicable_total": applicable,
            "level_accuracy": counts["correct"] / applicable if applicable else 0.0,
        })
    return pd.DataFrame(rows)


# --- Standard per-label metrics --------------------------------------------
def per_label_metrics(y_true, y_pred, label_name):
    """Accuracy, precision, recall, macro F1 for one level."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    mask = yt != DATA.missing_label_token
    yt, yp = yt[mask], yp[mask]

    if len(yt) == 0:
        return {"label": label_name, "accuracy": 0.0, "precision_macro": 0.0,
                "recall_macro": 0.0, "f1_macro": 0.0, "n_samples": 0}

    return {
        "label": label_name,
        "accuracy": float(accuracy_score(yt, yp)),
        "precision_macro": float(precision_score(yt, yp, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "n_samples": int(len(yt)),
    }


def build_classification_reports(y_true_chain, y_pred_chain, level_names):
    """Tidy long-form DataFrame of per-class metrics across all levels."""
    frames = []
    for level, name in enumerate(level_names):
        yt = np.asarray(y_true_chain[level])
        yp = np.asarray(y_pred_chain[level])
        mask = yt != DATA.missing_label_token
        if mask.sum() == 0:
            continue
        rep = classification_report(yt[mask], yp[mask],
                                    zero_division=0, output_dict=True)
        df = pd.DataFrame(rep).transpose()
        df["level"] = name
        df["class"] = df.index
        frames.append(df.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[["level", "class", "precision", "recall", "f1-score", "support"]]


# --- Confusion matrices ----------------------------------------------------
def save_confusion_matrices(y_true_chain, y_pred_chain, level_names,
                            model_name, out_dir=None):
    """Write a PNG confusion matrix per label level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or (OUTPUTS_DIR / ARTIFACTS.confusion_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for level, name in enumerate(level_names):
        yt = np.asarray(y_true_chain[level])
        yp = np.asarray(y_pred_chain[level])
        mask = yt != DATA.missing_label_token
        yt, yp = yt[mask], yp[mask]
        if len(yt) == 0:
            continue

        labels = sorted(set(yt.tolist()) | set(yp.tolist()))
        cm = confusion_matrix(yt, yp, labels=labels)

        fig, ax = plt.subplots(
            figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.5))
        )
        im = ax.imshow(cm, cmap="Greys")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name} - {name}")

        # annotate cells with counts
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=8, color=color)

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        path = out_dir / f"{model_name}_{name}_confusion.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


# --- Top-level eval function used by train.py ------------------------------
def evaluate_model(y_true_chain, y_pred_chain, level_names, model_name):
    """Return everything we need to write into metrics.json and the CSVs."""
    per_level = [
        per_label_metrics(yt, yp, name)
        for yt, yp, name in zip(y_true_chain, y_pred_chain, level_names)
    ]
    chain_score = chained_accuracy(y_true_chain, y_pred_chain)
    breakdown = chained_accuracy_breakdown(y_true_chain, y_pred_chain, level_names)
    per_class = build_classification_reports(y_true_chain, y_pred_chain, level_names)

    summary = {
        "model_name": model_name,
        "chained_accuracy": chain_score,
        "per_level_metrics": per_level,
    }
    return summary, per_class, breakdown
