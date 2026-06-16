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


# ── visual helpers ───────────────────────────────────────────────────────────

PALETTE = {
    "high": {"fg": "#15803d", "bg": "#f0fdf4", "bar": "#22c55e", "badge": "#dcfce7", "border": "rgba(21,128,61,.25)"},
    "mid":  {"fg": "#b45309", "bg": "#fffbeb", "bar": "#f59e0b", "badge": "#fef3c7", "border": "rgba(180,83,9,.25)"},
    "low":  {"fg": "#b91c1c", "bg": "#fef2f2", "bar": "#ef4444", "badge": "#fee2e2", "border": "rgba(185,28,28,.25)"},
}

LEVEL_META = {
    "Type 2": {"label": "Type 2", "subtitle": "Top-level category"},
    "Type 3": {"label": "Type 3", "subtitle": "Mid-level topic"},
    "Type 4": {"label": "Type 4", "subtitle": "Specific issue type"},
}


def _tier(conf):
    if conf >= 0.70:
        return PALETTE["high"], "High"
    if conf >= 0.40:
        return PALETTE["mid"], "Medium"
    return PALETTE["low"], "Low"


def _result_card(level_key, label, conf):
    p, word = _tier(conf)
    meta = LEVEL_META[level_key]
    pct = f"{conf * 100:.1f}"
    return f"""
<div style="background:#ffffff;border:1px solid {p['border']};border-left:3px solid {p['bar']};
     border-radius:12px;padding:18px 20px;margin-bottom:10px;
     box-shadow:0 1px 3px rgba(26,24,22,.06),0 4px 16px rgba(26,24,22,.04);">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:12px;">
    <div>
      <div style="font-size:10px;font-weight:600;color:#9c9590;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:4px;font-family:'DM Mono',monospace;">
        {meta['label']} — {meta['subtitle']}
      </div>
      <div style="font-size:17px;font-weight:700;color:#1a1816;letter-spacing:-.01em;">{label}</div>
    </div>
    <div style="background:{p['badge']};color:{p['fg']};font-size:12px;font-weight:700;
                padding:4px 12px;border-radius:99px;white-space:nowrap;margin-left:12px;margin-top:2px;
                font-family:'DM Mono',monospace;">{pct}%</div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="flex:1;background:#efece8;border-radius:99px;height:5px;overflow:hidden;">
      <div style="width:{pct}%;background:{p['bar']};height:5px;border-radius:99px;
                  transition:width .6s cubic-bezier(.4,0,.2,1);"></div>
    </div>
    <span style="font-size:10px;font-weight:600;color:{p['fg']};white-space:nowrap;
                 font-family:'DM Mono',monospace;">{word} confidence</span>
  </div>
</div>"""


EMPTY_CARD = """
<div style="background:#f7f5f0;border:1.5px dashed #c8c4be;border-radius:14px;
            padding:52px 24px;text-align:center;">
  <div style="width:44px;height:44px;background:#efece8;border-radius:12px;margin:0 auto 16px;
              display:flex;align-items:center;justify-content:center;font-size:22px;">📋</div>
  <div style="font-size:15px;font-weight:600;color:#1a1816;margin-bottom:6px;
              font-family:'Instrument Serif',serif;font-style:italic;">
    No ticket analysed yet
  </div>
  <div style="font-size:13px;color:#9c9590;max-width:240px;margin:0 auto;line-height:1.6;">
    Fill in the ticket details and press <strong style="color:#5c5750;">Analyse</strong>.
  </div>
</div>"""


def classify_single(summary, content, model_choice):
    if not summary.strip() and not content.strip():
        return EMPTY_CARD, "", "", ""

    text = preprocess_message(summary or "", content or "")
    X = transform_texts(VECTORIZER, [text])
    clf = CLASSIFIERS[model_choice]
    y_pred, confs = clf.predict_with_confidence(X)

    y2, y3, y4 = y_pred[0]
    c2, c3, c4 = confs[0]

    stamp = (
        f'<div style="font-size:11px;color:#9c9590;padding:4px 2px;display:flex;'
        f'align-items:center;gap:6px;font-family:\'DM Mono\',monospace;">'
        f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
        f'background:#22c55e;flex-shrink:0;"></span>'
        f'{model_choice} &nbsp;·&nbsp; '
        f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC</div>'
    )
    return _result_card("Type 2", y2, c2), _result_card("Type 3", y3, c3), _result_card("Type 4", y4, c4), stamp


# ── batch ────────────────────────────────────────────────────────────────────

REQUIRED_COLS = [COLUMNS.ticket_summary, COLUMNS.interaction_content]


def classify_batch(csv_file, model_choice):
    if csv_file is None:
        return None, "Upload a CSV file to get started."

    path = getattr(csv_file, "name", csv_file)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, (
            f"Missing columns: `{missing}`.\n\nExpected: `{REQUIRED_COLS}`\n\nFound: `{df.columns.tolist()}`"
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
    return str(out_path), f"Done — **{len(out)} rows** classified with **{model_choice}**. Download below."


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    max-width: 1040px !important;
    margin: 0 auto !important;
    font-family: 'Geist', system-ui, sans-serif !important;
    background: #f7f5f0 !important;
    padding: 0 24px 48px !important;
}
footer { display: none !important; }

/* ── nav tabs ── */
.tab-nav {
    background: transparent !important;
    border-bottom: 1.5px solid rgba(26,24,22,.1) !important;
    padding: 0 !important;
    gap: 0 !important;
}
.tab-nav button {
    font-family: 'Geist', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    color: #9c9590 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 18px !important;
    margin-bottom: -1.5px !important;
    border-radius: 0 !important;
    transition: color .15s !important;
}
.tab-nav button:hover { color: #1a1816 !important; }
.tab-nav button.selected {
    color: #1a1816 !important;
    border-bottom-color: #1a1816 !important;
    font-weight: 600 !important;
}
.tabitem { padding-top: 24px !important; border: none !important; }

/* ── surface cards ── */
.surface {
    background: #ffffff;
    border: 1px solid rgba(26,24,22,.09);
    border-radius: 14px;
    padding: 24px 26px;
    box-shadow: 0 1px 3px rgba(26,24,22,.06), 0 4px 16px rgba(26,24,22,.04);
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    color: #9c9590;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: 14px;
}
.divider {
    height: 1px;
    background: rgba(26,24,22,.07);
    margin: 16px 0;
}

/* ── primary button ── */
button.primary, .run-btn button, #analyse-btn {
    background: #1a1816 !important;
    color: #f7f5f0 !important;
    font-family: 'Geist', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 11px 24px !important;
    width: 100% !important;
    transition: opacity .15s !important;
    cursor: pointer !important;
    letter-spacing: .01em !important;
}
button.primary:hover, .run-btn button:hover, #analyse-btn:hover {
    opacity: .87 !important;
}
button.primary:active, #analyse-btn:active {
    transform: scale(.99) !important;
}

/* ── inputs ── */
label span, .label-wrap span {
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #5c5750 !important;
}
textarea, input[type=text] {
    font-family: 'Geist', sans-serif !important;
    font-size: 14px !important;
    color: #1a1816 !important;
    background: #faf9f7 !important;
    border: 1px solid rgba(26,24,22,.12) !important;
    border-radius: 8px !important;
    transition: border-color .15s, box-shadow .15s !important;
}
textarea:focus, input[type=text]:focus {
    border-color: rgba(26,24,22,.35) !important;
    box-shadow: 0 0 0 3px rgba(26,24,22,.06) !important;
    outline: none !important;
}
textarea::placeholder, input::placeholder { color: #c8c4be !important; }

/* ── radio ── */
.radio-group { gap: 8px !important; }
.radio-group label {
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #5c5750 !important;
    background: #f7f5f0 !important;
    border: 1px solid rgba(26,24,22,.12) !important;
    border-radius: 7px !important;
    padding: 7px 14px !important;
    cursor: pointer !important;
    transition: all .15s !important;
}
.radio-group label:hover { border-color: rgba(26,24,22,.25) !important; }
.radio-group input:checked + label, .radio-group label.selected {
    background: #1a1816 !important;
    color: #f7f5f0 !important;
    border-color: #1a1816 !important;
}

/* ── examples ── */
.examples-table { border-radius: 8px !important; overflow: hidden !important; }
.examples-table td {
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    color: #1a1816 !important;
    padding: 8px 14px !important;
    background: #faf9f7 !important;
    border-color: rgba(26,24,22,.08) !important;
    cursor: pointer !important;
    transition: background .1s !important;
}
.examples-table tr:hover td { background: #efece8 !important; }

/* ── stat grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.stat-box {
    background: #ffffff;
    border: 1px solid rgba(26,24,22,.09);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(26,24,22,.04);
}
.stat-num {
    font-family: 'Instrument Serif', serif;
    font-size: 26px;
    color: #1a1816;
    margin-bottom: 2px;
}
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #9c9590;
    text-transform: uppercase;
    letter-spacing: .07em;
}

/* ── legend ── */
.conf-legend {
    display: flex;
    gap: 16px;
    padding: 10px 14px;
    background: #f7f5f0;
    border-radius: 8px;
    border: 1px solid rgba(26,24,22,.07);
    margin-top: 8px;
    flex-wrap: wrap;
}
.conf-legend span {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #5c5750;
    display: flex;
    align-items: center;
    gap: 6px;
}
.conf-legend i {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

@media (max-width: 640px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
}
"""

# ── layout ───────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Support Ticket Classifier",
    theme=gr.themes.Base(
        font=gr.themes.GoogleFont("Geist"),
        font_mono=gr.themes.GoogleFont("DM Mono"),
    ),
    css=CSS,
) as demo:

    # ── hero ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="padding: 40px 0 28px; border-bottom: 1.5px solid rgba(26,24,22,.08); margin-bottom: 28px;">
      <div style="font-family:'DM Mono',monospace;font-size:11px;color:#9c9590;
                  text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;">
        MSc AI Engineering &amp; Evaluating AI Systems
      </div>
      <h1 style="font-family:'Instrument Serif',serif;font-size:clamp(28px,3.6vw,42px);
                 font-weight:400;color:#1a1816;margin:0 0 10px;letter-spacing:-.02em;line-height:1.15;">
        Customer Support<br><em>Ticket Classifier</em>
      </h1>
      <p style="font-family:'Geist',sans-serif;font-size:15px;color:#5c5750;
                margin:0;max-width:520px;line-height:1.65;font-weight:300;">
        Hierarchical multi-label classification across three label levels —
        using Random Forest and Logistic Regression trained on AppGallery support tickets.
      </p>
    </div>

    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-num">69.4%</div>
        <div class="stat-lbl">Chained Acc · LR</div>
      </div>
      <div class="stat-box">
        <div class="stat-num">68.3%</div>
        <div class="stat-lbl">Chained Acc · RF</div>
      </div>
      <div class="stat-box">
        <div class="stat-num" style="color:#15803d;">83.3%</div>
        <div class="stat-lbl">Type 2 F1 · RF</div>
      </div>
      <div class="stat-box">
        <div class="stat-num" style="color:#b45309;">206</div>
        <div class="stat-lbl">Training samples</div>
      </div>
    </div>
    """)

    with gr.Tabs():

        # ── Single ticket tab ─────────────────────────────────────────────────
        with gr.TabItem("Single ticket"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=5):
                    gr.HTML('<div class="surface">')
                    gr.HTML('<div class="section-label">Ticket details</div>')
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
                    btn = gr.Button("Analyse ticket", variant="primary", elem_id="analyse-btn")
                    gr.HTML('</div>')

                    gr.HTML('<div style="height:12px;"></div>')

                    gr.HTML('<div class="surface" style="padding:16px 20px;">')
                    gr.HTML('<div class="section-label" style="margin-bottom:10px;">Quick examples</div>')
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
                    gr.HTML('<div class="surface" style="min-height:340px;">')
                    gr.HTML('<div class="section-label">Predictions</div>')
                    y2_out = gr.HTML(value=EMPTY_CARD)
                    y3_out = gr.HTML()
                    y4_out = gr.HTML()
                    meta_out = gr.HTML()
                    gr.HTML("""
                    <div class="conf-legend">
                      <span><i style="background:#22c55e;"></i>≥ 70% High</span>
                      <span><i style="background:#f59e0b;"></i>40–70% Medium</span>
                      <span><i style="background:#ef4444;"></i>&lt; 40% Low</span>
                    </div>
                    """)
                    gr.HTML('</div>')

            btn.click(
                classify_single,
                inputs=[summary_in, content_in, model_single],
                outputs=[y2_out, y3_out, y4_out, meta_out],
            )

        # ── Batch tab ─────────────────────────────────────────────────────────
        with gr.TabItem("Batch CSV"):
            gr.HTML("""
            <div style="font-family:'Geist',sans-serif;font-size:14px;color:#5c5750;
                        line-height:1.65;margin-bottom:20px;font-weight:300;">
              Upload a CSV with <strong style="color:#1a1816;font-weight:600;">Ticket Summary</strong>
              and <strong style="color:#1a1816;font-weight:600;">Interaction content</strong> columns.
              An optional <code style="font-family:'DM Mono',monospace;font-size:12px;
              background:#efece8;padding:1px 6px;border-radius:4px;">message_id</code> column is preserved.
            </div>
            """)

            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    gr.HTML('<div class="surface">')
                    gr.HTML('<div class="section-label">Upload</div>')
                    csv_in = gr.File(label="CSV file", file_types=[".csv"])
                    gr.HTML('<div class="divider"></div>')
                    model_batch = gr.Radio(
                        choices=list(CLASSIFIERS.keys()),
                        value="Random Forest",
                        label="Model",
                    )
                    run_btn = gr.Button("Run batch prediction", variant="primary", elem_classes="run-btn")
                    gr.HTML('</div>')

                with gr.Column(scale=6):
                    gr.HTML('<div class="surface">')
                    gr.HTML('<div class="section-label">Results</div>')
                    status = gr.Markdown(value="Upload a CSV and click **Run batch prediction**.")
                    csv_out = gr.File(label="Download predictions", interactive=False)
                    gr.HTML('</div>')

            run_btn.click(
                classify_batch,
                inputs=[csv_in, model_batch],
                outputs=[csv_out, status],
            )

    gr.HTML("""
    <div style="text-align:center;padding:32px 0 0;font-family:'DM Mono',monospace;
                font-size:11px;color:#c8c4be;letter-spacing:.06em;
                border-top:1px solid rgba(26,24,22,.07);margin-top:32px;">
      HIERARCHICAL CLASSIFIER · SCIKIT-LEARN · NCI DUBLIN
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False,
        ssr_mode=False,
    )
