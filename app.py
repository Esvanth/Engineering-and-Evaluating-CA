"""
Gradio app for the customer-support ticket classifier.
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


def _tier(conf):
    if conf >= 0.70:
        return "#15803d", "#dcfce7", "#16a34a"
    if conf >= 0.40:
        return "#b45309", "#fef3c7", "#d97706"
    return "#b91c1c", "#fee2e2", "#dc2626"


def _result_card(level, subtitle, label, conf):
    fg, badge_bg, bar = _tier(conf)
    pct = f"{conf * 100:.1f}"
    word = "High" if conf >= 0.70 else "Medium" if conf >= 0.40 else "Low"
    return f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-left:3px solid {bar};
            border-radius:10px;padding:16px 18px;margin-bottom:10px;
            box-shadow:0 1px 4px rgba(0,0,0,.05);">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
    <div>
      <div style="font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:3px;">{level} — {subtitle}</div>
      <div style="font-size:16px;font-weight:700;color:#111827;">{label}</div>
    </div>
    <span style="background:{badge_bg};color:{fg};font-size:12px;font-weight:700;
                 padding:3px 10px;border-radius:99px;white-space:nowrap;margin-left:10px;">
      {pct}%
    </span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="flex:1;height:5px;background:#f3f4f6;border-radius:99px;overflow:hidden;">
      <div style="width:{pct}%;height:5px;background:{bar};border-radius:99px;"></div>
    </div>
    <span style="font-size:10px;font-weight:600;color:{fg};">{word}</span>
  </div>
</div>"""


EMPTY_STATE = """
<div style="background:#f9fafb;border:1.5px dashed #d1d5db;border-radius:10px;
            padding:44px 20px;text-align:center;">
  <div style="font-size:32px;margin-bottom:10px;">📋</div>
  <div style="font-size:14px;font-weight:600;color:#374151;margin-bottom:4px;">No prediction yet</div>
  <div style="font-size:13px;color:#9ca3af;">Enter a ticket and click <b>Analyse</b>.</div>
</div>"""


def classify_single(summary, content, model_choice):
    if not summary.strip() and not content.strip():
        return EMPTY_STATE, "", "", ""

    text = preprocess_message(summary or "", content or "")
    X = transform_texts(VECTORIZER, [text])
    clf = CLASSIFIERS[model_choice]
    y_pred, confs = clf.predict_with_confidence(X)

    y2, y3, y4 = y_pred[0]
    c2, c3, c4 = confs[0]

    stamp = (
        f'<div style="font-size:11px;color:#9ca3af;margin-top:4px;display:flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:#22c55e;display:inline-block;"></span>'
        f'{model_choice} &nbsp;·&nbsp; {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")} UTC'
        f'</div>'
    )
    return (
        _result_card("Type 2", "Top-level category", y2, c2),
        _result_card("Type 3", "Mid-level topic", y3, c3),
        _result_card("Type 4", "Specific issue", y4, c4),
        stamp,
    )


REQUIRED_COLS = [COLUMNS.ticket_summary, COLUMNS.interaction_content]


def classify_batch(csv_file, model_choice):
    if csv_file is None:
        return None, "Upload a CSV file to get started."

    path = getattr(csv_file, "name", csv_file)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, f"Missing columns: `{missing}`"
    if df.empty:
        return None, "CSV has no data rows."

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
    return str(out_path), f"Done — **{len(out)} rows** with **{model_choice}**. Download below."


CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    background: #f9fafb !important;
    padding-bottom: 40px !important;
}
footer { display: none !important; }

/* inputs — force white bg and dark text always */
.gradio-container input,
.gradio-container textarea {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: #6b7280 !important;
    box-shadow: 0 0 0 2px rgba(107,114,128,.15) !important;
    outline: none !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #9ca3af !important;
}

/* labels */
.gradio-container label > span:first-child,
.gradio-container .label-wrap span {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* primary button */
.gradio-container button.primary {
    background: #111827 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: opacity .15s !important;
}
.gradio-container button.primary:hover { opacity: .85 !important; }

/* tabs */
.gradio-container .tab-nav button {
    font-weight: 500 !important;
    font-size: 14px !important;
    color: #6b7280 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
.gradio-container .tab-nav button.selected {
    color: #111827 !important;
    border-bottom-color: #111827 !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(
    title="Support Ticket Classifier",
    theme=gr.themes.Default(),
    css=CSS,
) as demo:

    # ── header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="padding:32px 0 24px;border-bottom:1px solid #e5e7eb;margin-bottom:24px;">
      <h1 style="font-size:26px;font-weight:800;color:#111827;margin:0 0 6px;letter-spacing:-.02em;">
        Support Ticket Classifier
      </h1>
      <p style="font-size:14px;color:#6b7280;margin:0;line-height:1.6;">
        Hierarchical prediction across three label levels · Random Forest &amp; Logistic Regression
      </p>
    </div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px;">
      <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;
                  padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04);">
        <div style="font-size:22px;font-weight:800;color:#111827;">69.4%</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">Chained Acc · LR</div>
      </div>
      <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;
                  padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04);">
        <div style="font-size:22px;font-weight:800;color:#111827;">68.3%</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">Chained Acc · RF</div>
      </div>
      <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;
                  padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04);">
        <div style="font-size:22px;font-weight:800;color:#16a34a;">83.3%</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">Type 2 F1 · RF</div>
      </div>
      <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;
                  padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04);">
        <div style="font-size:22px;font-weight:800;color:#d97706;">206</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">Training samples</div>
      </div>
    </div>
    """)

    with gr.Tabs():

        # ── Single ticket ──────────────────────────────────────────────────────
        with gr.TabItem("Single ticket"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=5):
                    summary_in = gr.Textbox(
                        label="Ticket Summary",
                        placeholder="e.g. AppGallery update keeps failing",
                        lines=1,
                    )
                    content_in = gr.Textbox(
                        label="Ticket Description",
                        placeholder="Paste the full customer message here…",
                        lines=7,
                    )
                    model_single = gr.Radio(
                        choices=list(CLASSIFIERS.keys()),
                        value="Random Forest",
                        label="Model",
                    )
                    btn = gr.Button("Analyse ticket", variant="primary")

                    gr.Examples(
                        examples=[
                            ["Refund request", "I paid for an app but it did not work. Please help me get a refund."],
                            ["Subscription cancellation", "I want to cancel my subscription and stop future payments."],
                            ["Login issue", "I cannot sign in to my account after changing my password."],
                        ],
                        inputs=[summary_in, content_in],
                        label="Examples",
                    )

                with gr.Column(scale=5):
                    gr.HTML('<div style="font-size:12px;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;">Predictions</div>')
                    y2_out = gr.HTML(value=EMPTY_STATE)
                    y3_out = gr.HTML()
                    y4_out = gr.HTML()
                    meta_out = gr.HTML()
                    gr.HTML("""
                    <div style="display:flex;gap:14px;padding:10px 14px;background:#f9fafb;
                                border-radius:8px;border:1px solid #f3f4f6;margin-top:6px;">
                      <span style="font-size:11px;color:#6b7280;display:flex;align-items:center;gap:5px;">
                        <span style="width:8px;height:8px;border-radius:50%;background:#16a34a;display:inline-block;"></span>≥70% High
                      </span>
                      <span style="font-size:11px;color:#6b7280;display:flex;align-items:center;gap:5px;">
                        <span style="width:8px;height:8px;border-radius:50%;background:#d97706;display:inline-block;"></span>40–70% Medium
                      </span>
                      <span style="font-size:11px;color:#6b7280;display:flex;align-items:center;gap:5px;">
                        <span style="width:8px;height:8px;border-radius:50%;background:#dc2626;display:inline-block;"></span>&lt;40% Low
                      </span>
                    </div>
                    """)

            btn.click(
                classify_single,
                inputs=[summary_in, content_in, model_single],
                outputs=[y2_out, y3_out, y4_out, meta_out],
            )

        # ── Batch CSV ──────────────────────────────────────────────────────────
        with gr.TabItem("Batch CSV"):
            gr.Markdown(
                "Upload a CSV with **Ticket Summary** and **Interaction content** columns. "
                "Results include confidence scores for all three levels."
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                    model_batch = gr.Radio(
                        choices=list(CLASSIFIERS.keys()),
                        value="Random Forest",
                        label="Model",
                    )
                    run_btn = gr.Button("Run predictions", variant="primary")
                with gr.Column(scale=6):
                    status = gr.Markdown(value="Upload a CSV and click **Run predictions**.")
                    csv_out = gr.File(label="Download predictions", interactive=False)

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
