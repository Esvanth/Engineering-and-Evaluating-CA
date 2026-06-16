"""
Gradio app for the customer-support ticket classifier
"""
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

from src.config import ARTIFACTS, ARTIFACTS_DIR, COLUMNS
from src.features import load_vectorizer, transform_texts
from src.models import ChainedHierarchicalClassifier
from src.preprocessing import preprocess_message


VECTORIZER = load_vectorizer(ARTIFACTS_DIR / ARTIFACTS.vectorizer)
CLASSIFIER_RF = ChainedHierarchicalClassifier.load(
    ARTIFACTS_DIR / ARTIFACTS.classifier_template.format(model_name="rf")
)
CLASSIFIER_LR = ChainedHierarchicalClassifier.load(
    ARTIFACTS_DIR / ARTIFACTS.classifier_template.format(model_name="lr")
)
CLASSIFIERS = {"Random Forest": CLASSIFIER_RF, "Logistic Regression": CLASSIFIER_LR}


def _conf_color(conf):
    if conf >= 0.70:
        return "#16a34a"
    if conf >= 0.40:
        return "#d97706"
    return "#dc2626"


def _result_card(level, sublabel, label, conf):
    color = _conf_color(conf)
    pct = f"{conf * 100:.0f}%"
    return (
        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid {color};'
        f'border-radius:10px;padding:16px 20px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,.06);">'
        f'<div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;'
        f'letter-spacing:.08em;margin-bottom:4px;">{level}</div>'
        f'<div style="font-size:19px;font-weight:800;color:#1e293b;margin-bottom:2px;">{label}</div>'
        f'<div style="font-size:12px;color:#64748b;margin-bottom:10px;">{sublabel}</div>'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<div style="flex:1;background:#f1f5f9;border-radius:99px;height:7px;overflow:hidden;">'
        f'<div style="width:{pct};background:{color};height:7px;border-radius:99px;'
        f'transition:width .4s ease;"></div></div>'
        f'<span style="font-size:13px;font-weight:700;color:{color};min-width:42px;text-align:right;">'
        f'{conf:.1%}</span></div></div>'
    )


EMPTY_CARD = (
    '<div style="background:#f8fafc;border:1px dashed #cbd5e1;border-radius:10px;'
    'padding:40px 20px;text-align:center;color:#94a3b8;font-size:14px;">'
    'Predictions will appear here after you classify a ticket.</div>'
)


def classify_single(summary, content, model_choice):
    if not summary.strip() and not content.strip():
        return EMPTY_CARD, "", "", ""

    text = preprocess_message(summary or "", content or "")
    X = transform_texts(VECTORIZER, [text])
    clf = CLASSIFIERS[model_choice]
    y_pred, confs = clf.predict_with_confidence(X)

    y2, y3, y4 = y_pred[0]
    c2, c3, c4 = confs[0]

    c2_html = _result_card("Type 2 — Top-level category", "Overall request type", y2, c2)
    c3_html = _result_card("Type 3 — Mid-level category", "Topic area", y3, c3)
    c4_html = _result_card("Type 4 — Leaf-level category", "Specific issue", y4, c4)
    stamp = (
        f'<div style="font-size:11px;color:#94a3b8;padding:4px 2px;">'
        f'Model: <b>{model_choice}</b> &nbsp;·&nbsp; '
        f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC</div>'
    )
    return c2_html, c3_html, c4_html, stamp


REQUIRED_COLS = [COLUMNS.ticket_summary, COLUMNS.interaction_content]


def classify_batch(csv_file, model_choice):
    if csv_file is None:
        return None, "Upload a CSV and click **Run Predictions**."

    path = getattr(csv_file, "name", csv_file)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, (
            f"Missing required columns: `{missing}`.\n\n"
            f"Expected: `{REQUIRED_COLS}`\n\nFound: `{df.columns.tolist()}`"
        )
    if df.empty:
        return None, "The uploaded CSV has no data rows."

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
    return str(out_path), f"Done — **{len(out)} rows** predicted with **{model_choice}**. Download below."


CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto; font-family: 'Inter', sans-serif; }
footer { display: none !important; }
#page-header { text-align: center; padding: 24px 0 8px; }
#classify-btn, #run-btn {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 0 !important;
    transition: opacity .15s;
}
#classify-btn:hover, #run-btn:hover { opacity: .88 !important; }
.metric-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
}
"""

with gr.Blocks(
    title="Customer Support Ticket Classifier",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    gr.HTML("""
    <div id="page-header">
      <h1 style="font-size:2rem;font-weight:900;margin:0;background:linear-gradient(90deg,#6366f1,#8b5cf6);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Customer Support Ticket Classifier
      </h1>
      <p style="color:#64748b;margin:8px 0 0;font-size:15px;">
        Hierarchical prediction across three label levels using ML models trained on AppGallery support tickets
      </p>
    </div>
    """)

    with gr.Accordion("Model performance on held-out test set", open=False):
        with gr.Row():
            gr.HTML('<div class="metric-pill"><div style="font-size:22px;font-weight:800;color:#6366f1;">69.4%</div><div style="font-size:12px;color:#64748b;margin-top:2px;">Chained Accuracy (LR)</div></div>')
            gr.HTML('<div class="metric-pill"><div style="font-size:22px;font-weight:800;color:#6366f1;">68.3%</div><div style="font-size:12px;color:#64748b;margin-top:2px;">Chained Accuracy (RF)</div></div>')
            gr.HTML('<div class="metric-pill"><div style="font-size:22px;font-weight:800;color:#6366f1;">83.3%</div><div style="font-size:12px;color:#64748b;margin-top:2px;">Type 2 Macro F1 (RF)</div></div>')
            gr.HTML('<div class="metric-pill"><div style="font-size:22px;font-weight:800;color:#6366f1;">206</div><div style="font-size:12px;color:#64748b;margin-top:2px;">Training samples</div></div>')
        gr.Markdown(
            "> **Heads up:** This is a research prototype trained on a small dataset. "
            "Confidence scores below 50% indicate the model is uncertain — treat those predictions with caution.",
            elem_id="disclaimer",
        )

    gr.HTML("<div style='height:8px'></div>")

    with gr.Tab("Single Message"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                summary_in = gr.Textbox(
                    label="Ticket Summary",
                    placeholder="e.g. AppGallery update keeps failing",
                    lines=1,
                )
                content_in = gr.Textbox(
                    label="Interaction Content",
                    placeholder="Describe the customer's issue in detail…",
                    lines=7,
                )
                with gr.Row():
                    model_single = gr.Radio(
                        choices=list(CLASSIFIERS.keys()),
                        value="Random Forest",
                        label="Model",
                        scale=2,
                    )
                    btn = gr.Button(
                        "Classify Ticket →",
                        variant="primary",
                        elem_id="classify-btn",
                        scale=1,
                    )

                gr.Examples(
                    examples=[
                        ["Refund request", "I paid for an app but it did not work. Please help me get a refund."],
                        ["Subscription cancellation", "I want to cancel my subscription and stop future payments."],
                        ["Login issue", "I cannot sign in to my account after changing my password."],
                    ],
                    inputs=[summary_in, content_in],
                    label="Examples — click to load",
                )

            with gr.Column(scale=5):
                gr.HTML('<div style="font-size:13px;font-weight:600;color:#475569;margin-bottom:8px;">PREDICTION RESULTS</div>')
                y2_out = gr.HTML(value=EMPTY_CARD)
                y3_out = gr.HTML()
                y4_out = gr.HTML()
                meta_out = gr.HTML()

        btn.click(
            classify_single,
            inputs=[summary_in, content_in, model_single],
            outputs=[y2_out, y3_out, y4_out, meta_out],
        )

    with gr.Tab("Batch CSV"):
        gr.Markdown(
            "Upload a CSV containing **`Ticket Summary`** and **`Interaction content`** columns. "
            "An optional `message_id` column is preserved in the output. "
            "Results include confidence scores for all three prediction levels."
        )
        gr.HTML("<div style='height:4px'></div>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                model_batch = gr.Radio(
                    choices=list(CLASSIFIERS.keys()),
                    value="Random Forest",
                    label="Model",
                )
                run_btn = gr.Button("Run Predictions →", variant="primary", elem_id="run-btn")
            with gr.Column(scale=6):
                status = gr.Markdown(value="Upload a CSV and click **Run Predictions**.")
                csv_out = gr.File(label="Download Predictions CSV")

        run_btn.click(
            classify_batch,
            inputs=[csv_in, model_batch],
            outputs=[csv_out, status],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False,
        ssr_mode=False,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,         share=False,
        ssr_mode=False,

    )
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False,
        ssr_mode=False,
    )