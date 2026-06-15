"""
Gradio app for the customer-support ticket classifier.

Two tabs:
  - Single message: type a ticket summary + content, get the three predicted
    labels with confidence scores.
  - Batch CSV: upload a CSV (same format as new_messages.csv), download a
    predictions CSV.

The app reuses the trained model that was saved by src.train, so the same
chained accuracy reported in the report applies here.
"""
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

from src.config import ARTIFACTS, ARTIFACTS_DIR, COLUMNS
from src.features import load_vectorizer, transform_texts
from src.models import ChainedHierarchicalClassifier
from src.preprocessing import preprocess_message


# Load model artefacts once at startup
VECTORIZER = load_vectorizer(ARTIFACTS_DIR / ARTIFACTS.vectorizer)
CLASSIFIER_RF = ChainedHierarchicalClassifier.load(
    ARTIFACTS_DIR / ARTIFACTS.classifier_template.format(model_name="rf")
)
CLASSIFIER_LR = ChainedHierarchicalClassifier.load(
    ARTIFACTS_DIR / ARTIFACTS.classifier_template.format(model_name="lr")
)
CLASSIFIERS = {"Random Forest": CLASSIFIER_RF, "Logistic Regression": CLASSIFIER_LR}


def classify_single(summary, content, model_choice):
    if not summary and not content:
        return "Please provide a ticket summary or content.", "", "", ""

    text = preprocess_message(summary or "", content or "")
    X = transform_texts(VECTORIZER, [text])
    clf = CLASSIFIERS[model_choice]
    y_pred, confs = clf.predict_with_confidence(X)

    y2, y3, y4 = y_pred[0]
    c2, c3, c4 = confs[0]

    def fmt(label, conf, level):
        return f"### {level}\n**{label}**\n\nConfidence: `{conf:.2%}`"

    return (
        fmt(y2, c2, "Type 2 (top level)"),
        fmt(y3, c3, "Type 3 (mid level)"),
        fmt(y4, c4, "Type 4 (leaf level)"),
        f"Model: {model_choice}  |  Predicted at {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    )


REQUIRED_COLS = [COLUMNS.ticket_summary, COLUMNS.interaction_content]


def classify_batch(csv_file, model_choice):
    if csv_file is None:
        return None, "Please upload a CSV file."

    df = pd.read_csv(csv_file.name)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, (
            f"CSV is missing required columns: {missing}.\n"
            f"Expected columns: {REQUIRED_COLS}.\n"
            f"Found columns: {df.columns.tolist()}"
        )
    if df.empty:
        return None, "CSV has no rows."

    id_col = None
    for c in ("message_id", "row_id", COLUMNS.ticket_id, COLUMNS.interaction_id):
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        df["row_id"] = range(1, len(df) + 1)
        id_col = "row_id"

    texts = [
        preprocess_message(s, c)
        for s, c in zip(df[COLUMNS.ticket_summary], df[COLUMNS.interaction_content])
    ]

    X = transform_texts(VECTORIZER, texts)
    clf = CLASSIFIERS[model_choice]
    y_pred, confs = clf.predict_with_confidence(X)

    out = pd.DataFrame({
        "message_id": df[id_col].values,
        "input_summary": df[COLUMNS.ticket_summary].astype(str).str[:200].values,
        "y2_predicted": y_pred[:, 0],
        "y2_confidence": confs[:, 0].round(4),
        "y3_predicted": y_pred[:, 1],
        "y3_confidence": confs[:, 1].round(4),
        "y4_predicted": y_pred[:, 2],
        "y4_confidence": confs[:, 2].round(4),
        "model_name": clf.name,
        "model_version": "rf" if model_choice == "Random Forest" else "lr",
        "prediction_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    out_path = Path("/tmp/predictions.csv")
    out.to_csv(out_path, index=False)
    return str(out_path), f"Predicted {len(out)} rows with {model_choice}. Download the CSV below."


INTRO = """
# Customer Support Ticket Classifier

Hierarchical multi-label classifier that predicts three label levels for customer-support tickets:

- **Type 2**: top-level category (Suggestion / Problem/Fault / Others)
- **Type 3**: middle-level category (e.g. Payment, Refund, AppGallery-Install)
- **Type 4**: leaf-level category (e.g. Subscription cancellation)

> **Note on accuracy.** Trained on 206 customer-support tickets from the AppGallery and
> In-App Purchase domains. Chained accuracy on the held-out test set is 0.69.
> Predictions on text from very different domains will be unreliable.
"""

with gr.Blocks(title="Customer Support Ticket Classifier") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Single message"):
        with gr.Row():
            with gr.Column():
                summary_in = gr.Textbox(label="Ticket Summary",
                                        placeholder="e.g. AppGallery update problem", lines=1)
                content_in = gr.Textbox(label="Interaction content",
                                        placeholder="e.g. The app store update keeps failing.",
                                        lines=5)
                model_single = gr.Radio(choices=list(CLASSIFIERS.keys()),
                                        value="Random Forest", label="Model")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                y2_out = gr.Markdown()
                y3_out = gr.Markdown()
                y4_out = gr.Markdown()
                meta_out = gr.Markdown()

        btn.click(classify_single,
                  inputs=[summary_in, content_in, model_single],
                  outputs=[y2_out, y3_out, y4_out, meta_out])

        gr.Examples(
            examples=[
                ["Refund request", "I paid for an app but it did not work. Please help me get a refund."],
                ["Subscription cancellation", "I want to cancel my subscription and stop future payments."],
                ["Login issue", "I cannot sign in to my account after changing my password."],
            ],
            inputs=[summary_in, content_in],
        )

    with gr.Tab("Batch CSV"):
        gr.Markdown(
            "Upload a CSV with `Ticket Summary` and `Interaction content` columns "
            "(plus an optional `message_id`). Get a predictions CSV back with "
            "confidence scores per level."
        )
        csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
        model_batch = gr.Radio(choices=list(CLASSIFIERS.keys()),
                               value="Random Forest", label="Model")
        run_btn = gr.Button("Run predictions", variant="primary")
        status = gr.Markdown()
        csv_out = gr.File(label="Predictions CSV")

        run_btn.click(classify_batch,
                      inputs=[csv_in, model_batch],
                      outputs=[csv_out, status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)