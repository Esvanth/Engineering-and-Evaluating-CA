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
<div style="background:#fff;border:1px solid rgba(26,24,22,.10);border-left:3px solid {bar};
            border-radius:12px;padding:16px 18px;margin-bottom:10px;
            box-shadow:0 1px 3px rgba(26,24,22,.06),0 4px 14px rgba(26,24,22,.04);">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
    <div>
      <div style="font-size:10px;font-weight:600;color:#9c9590;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:3px;">{level} — {subtitle}</div>
      <div style="font-size:16px;font-weight:700;color:#1a1816;">{label}</div>
    </div>
    <span style="background:{badge_bg};color:{fg};font-size:12px;font-weight:700;
                 padding:3px 11px;border-radius:99px;white-space:nowrap;margin-left:10px;">{pct}%</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="flex:1;height:5px;background:#efece8;border-radius:99px;overflow:hidden;">
      <div style="width:{pct}%;height:5px;background:{bar};border-radius:99px;"></div>
    </div>
    <span style="font-size:10px;font-weight:600;color:{fg};">{word}</span>
  </div>
</div>"""


EMPTY_STATE = """
<div style="background:#f7f5f0;border:1.5px dashed #c8c4be;border-radius:12px;
            padding:48px 20px;text-align:center;">
  <div style="font-size:30px;margin-bottom:12px;">📋</div>
  <div style="font-size:15px;font-weight:600;color:#1a1816;margin-bottom:5px;">No prediction yet</div>
  <div style="font-size:13px;color:#9c9590;line-height:1.5;">
    Enter a ticket on the left and click <strong>Analyse</strong>.
  </div>
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
        f'<div style="font-size:11px;color:#9c9590;margin-top:4px;display:flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:#22c55e;display:inline-block;flex-shrink:0;"></span>'
        f'{model_choice} · {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")} UTC</div>'
    )
    return (
        _result_card("Type 2", "Top-level category", y2, c2),
        _result_card("Type 3", "Mid-level topic",    y3, c3),
        _result_card("Type 4", "Specific issue",     y4, c4),
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
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700;800&display=swap');

.gradio-container {
    max-width: 1040px !important;
    margin: 0 auto !important;
    background: #f7f5f0 !important;
    padding-bottom: 48px !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
footer { display: none !important; }

/* force white readable inputs always */
.gradio-container input,
.gradio-container textarea,
.gradio-container .block input,
.gradio-container .block textarea {
    background-color: #ffffff !important;
    color: #1a1816 !important;
    border: 1px solid #d6d2cc !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: #5c5750 !important;
    box-shadow: 0 0 0 2px rgba(92,87,80,.12) !important;
    outline: none !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder { color: #b8b3ac !important; }

/* labels */
.gradio-container label > span,
.gradio-container .label-wrap span {
    color: #5c5750 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}

/* primary button */
.gradio-container button.primary {
    background: #1a1816 !important;
    color: #f7f5f0 !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity .15s !important;
}
.gradio-container button.primary:hover { opacity: .85 !important; }
.gradio-container button.primary:active { transform: scale(.99) !important; }

/* tab strip */
.gradio-container .tab-nav {
    background: transparent !important;
    border-bottom: 1.5px solid rgba(26,24,22,.10) !important;
}
.gradio-container .tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #9c9590 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 11px 18px !important;
    margin-bottom: -1.5px !important;
}
.gradio-container .tab-nav button:hover { color: #1a1816 !important; }
.gradio-container .tab-nav button.selected {
    color: #1a1816 !important;
    border-bottom-color: #1a1816 !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* surface card helper */
.surface-card {
    background: #ffffff;
    border: 1px solid rgba(26,24,22,.09);
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(26,24,22,.05), 0 4px 16px rgba(26,24,22,.03);
}
.card-label {
    font-size: 10px;
    font-weight: 700;
    color: #9c9590;
    text-transform: uppercase;
    letter-spacing: .09em;
    margin-bottom: 14px;
    font-family: 'Inter', sans-serif;
}
.divider { height: 1px; background: rgba(26,24,22,.07); margin: 14px 0; }
"""

with gr.Blocks(
    title="Support Ticket Classifier",
    theme=gr.themes.Default(),
    css=CSS,
) as demo:

    # ── hero ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="padding:36px 0 24px;border-bottom:1.5px solid rgba(26,24,22,.08);margin-bottom:24px;">
      <h1 style="font-family:'Instrument Serif',serif;font-size:clamp(26px,3.2vw,38px);
                 font-weight:400;color:#1a1816;margin:0 0 8px;letter-spacing:-.02em;line-height:1.2;">
        Customer Support <em>Ticket Classifier</em>
      </h1>
      <p style="font-size:14px;color:#5c5750;margin:0;line-height:1.65;font-weight:400;max-width:500px;">
        Hierarchical multi-label classification across three label levels using
        Random Forest &amp; Logistic Regression.
      </p>
    </div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px;">
      <div style="background:#fff;border:1px solid rgba(26,24,22,.09);border-radius:12px;padding:16px;
                  text-align:center;box-shadow:0 1px 3px rgba(26,24,22,.05);">
        <div style="font-size:22px;font-weight:800;color:#1a1816;letter-spacing:-.02em;">69.4%</div>
        <div style="font-size:10px;color:#9c9590;margin-top:3px;text-transform:uppercase;letter-spacing:.06em;">Chained Acc · LR</div>
      </div>
      <div style="background:#fff;border:1px solid rgba(26,24,22,.09);border-radius:12px;padding:16px;
                  text-align:center;box-shadow:0 1px 3px rgba(26,24,22,.05);">
        <div style="font-size:22px;font-weight:800;color:#1a1816;letter-spacing:-.02em;">68.3%</div>
        <div style="font-size:10px;color:#9c9590;margin-top:3px;text-transform:uppercase;letter-spacing:.06em;">Chained Acc · RF</div>
      </div>
      <div style="background:#fff;border:1px solid rgba(26,24,22,.09);border-radius:12px;padding:16px;
                  text-align:center;box-shadow:0 1px 3px rgba(26,24,22,.05);">
        <div style="font-size:22px;font-weight:800;color:#15803d;letter-spacing:-.02em;">83.3%</div>
        <div style="font-size:10px;color:#9c9590;margin-top:3px;text-transform:uppercase;letter-spacing:.06em;">Type 2 F1 · RF</div>
      </div>
      <div style="background:#fff;border:1px solid rgba(26,24,22,.09);border-radius:12px;padding:16px;
                  text-align:center;box-shadow:0 1px 3px rgba(26,24,22,.05);">
        <div style="font-size:22px;font-weight:800;color:#b45309;letter-spacing:-.02em;">206</div>
        <div style="font-size:10px;color:#9c9590;margin-top:3px;text-transform:uppercase;letter-spacing:.06em;">Training samples</div>
      </div>
    </div>
    """)

    with gr.Tabs():

        # ── Single ticket ──────────────────────────────────────────────────────
        with gr.TabItem("Single ticket"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=5):
                    gr.HTML('<div class="surface-card">')
                    gr.HTML('<div class="card-label">Ticket details</div>')
                    summary_in = gr.Textbox(
                        label="Summary",
                        placeholder="Brief one-line description…",
                        lines=1,
                    )
                    content_in = gr.Textbox(
                        label="Description",
                        placeholder="Paste the full customer message here…",
                        lines=7,
                    )
                    gr.HTML('<div class="divider"></div>')
                    model_single = gr.Radio(
                        choices=list(CLASSIFIERS.keys()),
                        value="Random Forest",
                        label="Model",
                    )
                    btn = gr.Button("Analyse ticket", variant="primary")
                    gr.HTML('</div>')

                    gr.HTML('<div style="height:10px;"></div>')
                    gr.HTML('<div class="surface-card" style="padding:16px 20px;">')
                    gr.HTML('<div class="card-label" style="margin-bottom:8px;">Quick examples</div>')
                    gr.Examples(
                        examples=[
                            ["Refund request", "I paid for an app but it did not work. Please help me get a refund."],
                            ["Subscription cancellation", "I want to cancel my subscription and stop future payments."],
                            ["Login issue", "I cannot sign in to my account after changing my password."],
                        ],
                        inputs=[summary_in, content_in],
                        label=None,
                    )
                    gr.HTML('</div>')

                with gr.Column(scale=5):
                    gr.HTML('<div class="surface-card" style="min-height:320px;">')
                    gr.HTML('<div class="card-label">Predictions</div>')
                    y2_out = gr.HTML(value=EMPTY_STATE)
                    y3_out = gr.HTML()
                    y4_out = gr.HTML()
                    meta_out = gr.HTML()
                    gr.HTML("""
                    <div style="display:flex;gap:14px;flex-wrap:wrap;padding:10px 12px;
                                background:#f7f5f0;border-radius:8px;margin-top:8px;">
                      <span style="font-size:11px;color:#5c5750;display:flex;align-items:center;gap:5px;">
                        <span style="width:8px;height:8px;border-radius:50%;background:#16a34a;display:inline-block;"></span>≥ 70% High
                      </span>
                      <span style="font-size:11px;color:#5c5750;display:flex;align-items:center;gap:5px;">
                        <span style="width:8px;height:8px;border-radius:50%;background:#d97706;display:inline-block;"></span>40–70% Medium
                      </span>
                      <span style="font-size:11px;color:#5c5750;display:flex;align-items:center;gap:5px;">
                        <span style="width:8px;height:8px;border-radius:50%;background:#dc2626;display:inline-block;"></span>&lt; 40% Low
                      </span>
                    </div>
                    """)
                    gr.HTML('</div>')

            btn.click(
                classify_single,
                inputs=[summary_in, content_in, model_single],
                outputs=[y2_out, y3_out, y4_out, meta_out],
            )

        # ── Batch CSV ──────────────────────────────────────────────────────────
        with gr.TabItem("Batch CSV"):
            gr.HTML("""
            <div style="font-size:14px;color:#5c5750;line-height:1.65;margin-bottom:20px;">
              Upload a CSV with <strong style="color:#1a1816;">Ticket Summary</strong> and
              <strong style="color:#1a1816;">Interaction content</strong> columns.
              An optional <code style="background:#efece8;padding:1px 6px;border-radius:4px;
              font-size:12px;">message_id</code> column is preserved in output.
            </div>
            """)
            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    gr.HTML('<div class="surface-card">')
                    gr.HTML('<div class="card-label">Upload</div>')
                    csv_in = gr.File(label="CSV file", file_types=[".csv"])
                    gr.HTML('<div class="divider"></div>')
                    model_batch = gr.Radio(
                        choices=list(CLASSIFIERS.keys()),
                        value="Random Forest",
                        label="Model",
                    )
                    run_btn = gr.Button("Run batch prediction", variant="primary")
                    gr.HTML('</div>')

                with gr.Column(scale=6):
                    gr.HTML('<div class="surface-card">')
                    gr.HTML('<div class="card-label">Results</div>')
                    status = gr.Markdown(value="Upload a CSV and click **Run batch prediction**.")
                    csv_out = gr.File(label="Download predictions", interactive=False)
                    gr.HTML('</div>')

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
