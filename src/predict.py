"""
Batch inference. Takes a CSV of new messages, returns a CSV of predictions
with per-level confidence scores, model version and a timestamp.

Run with:
    python -m src.predict --input data/new_messages.csv \
                          --output outputs/predictions.csv \
                          --model rf
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS, ARTIFACTS_DIR, COLUMNS, LOGS_DIR
from src.features import load_vectorizer, transform_texts
from src.logging_utils import get_logger
from src.models import ChainedHierarchicalClassifier
from src.preprocessing import preprocess_message

log = get_logger(__name__, log_file=LOGS_DIR / "prediction.log")

REQUIRED_INPUT_COLS = [COLUMNS.ticket_summary, COLUMNS.interaction_content]


def _find_id_column(df):
    """Pick whichever id-like column exists, or invent a row_id."""
    for c in ("message_id", "row_id", COLUMNS.ticket_id, COLUMNS.interaction_id):
        if c in df.columns:
            return c
    df["row_id"] = range(1, len(df) + 1)
    return "row_id"


def predict_batch(input_path, output_path, model_name="rf"):
    log.info("=" * 70)
    log.info("Batch inference | input=%s | model=%s", input_path, model_name)
    log.info("=" * 70)

    df = pd.read_csv(input_path)

    # validate the input has the columns we need
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{input_path.name} is missing columns: {missing}")
    if df.empty:
        raise ValueError(f"{input_path.name} has no rows")

    id_col = _find_id_column(df)

    # Preprocess every message using the SAME function the trainer used.
    # This is what guarantees training-serving consistency.
    texts = [
        preprocess_message(s, c)
        for s, c in zip(df[COLUMNS.ticket_summary], df[COLUMNS.interaction_content])
    ]

    # load the saved artefacts
    vec_path = ARTIFACTS_DIR / ARTIFACTS.vectorizer
    clf_path = ARTIFACTS_DIR / ARTIFACTS.classifier_template.format(model_name=model_name)
    vec = load_vectorizer(vec_path)
    clf = ChainedHierarchicalClassifier.load(clf_path)

    # vectorise + predict
    X = transform_texts(vec, texts)
    y_pred, confs = clf.predict_with_confidence(X)

    # build the output table
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = pd.DataFrame({
        "message_id": df[id_col].values,
        "input_summary": df[COLUMNS.ticket_summary].astype(str).str[:200].values,
        "y2_predicted": y_pred[:, 0], "y2_confidence": confs[:, 0].round(4),
        "y3_predicted": y_pred[:, 1], "y3_confidence": confs[:, 1].round(4),
        "y4_predicted": y_pred[:, 2], "y4_confidence": confs[:, 2].round(4),
        "model_name": clf.name,
        "model_version": model_name,
        "prediction_timestamp_utc": now,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info("Wrote %d predictions to %s", len(out), output_path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch prediction over new messages.")
    ap.add_argument("--input", type=Path, required=True, help="Input CSV path")
    ap.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"),
                    help="Where to write the predictions CSV")
    ap.add_argument("--model", default="rf", choices=["rf", "lr"])
    args = ap.parse_args()
    predict_batch(args.input, args.output, model_name=args.model)
