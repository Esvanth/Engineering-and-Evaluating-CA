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


# ── visual helpers ──────────────────────────────────────────────────────────

PALETTE = {
    "high": {"fg": "#065f46", "bg": "#ecfdf5", "bar": "#059669", "badge": "#d1fae5"},
    "mid": {"fg": "#92400e", "bg": "#fffbeb", "bar": "#d97706", "badge": "#fef3c7"},
    "low": {"fg": "#991b1b", "bg": "#fef2f2", "bar": "#dc2626", "badge": "#fee2e2"},
}

LEVEL_META = {
    "Type 2": {"icon": "📁", "subtitle": "Category"},
    "Type 3": {"icon": "📌", "subtitle": "Topic"},
    "Type 4": {"icon": "🏷️", "subtitle": "Issue"},
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
    <div style="
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid {p['bar']};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
        transition: box-shadow .2s;
    ">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:18px;">{meta['icon']}</span>
          <div>
            <div style="font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;
                        letter-spacing:.04em;">{meta['subtitle']}</div>
            <div style="font-size:15px;font-weight:700;color:#1f2937;margin-top:1px;">{label}</div>
          </div>
        </div>
        <div style="background:{p['badge']};color:{p['fg']};font-size:12px;font-weight:700;
                    padding:3px 12px;border-radius:99px;white-space:nowrap;">{pct}%</div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <div style="flex:1;background:#f3f4f6;border-radius:99px;height:4px;overflow:hidden;">
          <div style="width:{pct}%;background:{p['bar']};height:4px;border-radius:99px;
                      transition:width .4s ease;"></div>
        </div>
        <span style="font-size:9px;font-weight:600;color:{p['fg']};white-space:nowrap;">{word}</span>
      </div>
    </div>
    """


EMPTY_CARD = """
<div style="background:#fafafa;border:2px dashed #d1d5db;border-radius:12px;
            padding:48px 20px;text-align:center;">
  <div style="font-size:36px;margin-bottom:12px;">📬</div>
  <div style="font-size:15px;font-weight:600;color:#4b5563;margin-bottom:4px;">
    No ticket analysed yet
  </div>
  <div style="font-size:12px;color:#9ca3af;max-width:260px;margin:0 auto;line-height:1.5;">
    Enter a summary and description, then click <strong>Analyse</strong>.
  </div>
</div>
"""


def classify_single(summary, content, model_choice):
    if not summary.strip() and not content.strip():
        return EMPTY_CARD, "", "", ""

    text = preprocess_message(summary or "", content or "")
    X = transform_texts(VECTORIZER, [text])
    clf = CLASSIFIERS[model_choice]
    y_pred, confs = clf.predict_with_confidence(X)

    y2, y3, y4 = y_pred[0]
    c2, c3, c4 = confs[0]

    cards = (
        _result_card("Type 2", y2, c2),
        _result_card("Type 3", y3, c3),
        _result_card("Type 4", y4, c4),
    )
    stamp = (
        f'<div style="font-size:11px;color:#9ca3af;padding:4px 2px;display:flex;'
        f'align-items:center;gap:6px;">'
        f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
        f'background:#059669;"></span>'
        f'<span style="font-weight:500;">{model_choice}</span>'
        f'&nbsp;·&nbsp; '
        f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC</div>'
    )
    return *cards, stamp


# ── batch mode ──────────────────────────────────────────────────────────────

REQUIRED_COLS = [COLUMNS.ticket_summary, COLUMNS.interaction_content]


def classify_batch(csv_file, model_choice):
    if csv_file is None:
        return None, "Upload a CSV file to get started."

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
    return str(out_path), f"✅ **{len(out)} rows** processed with **{model_choice}**. Download the results below."


# ── UI ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    background: #f8fafc !important;
}
.dark .gradio-container { background: #0f172a !important; }
footer { display: none !important; }

/* ── header ── */
.app-header {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.app-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.app-brand .logo {
    width: 40px;
    height: 40px;
    background: #4f46e5;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 18px;
    font-weight: 700;
}
.app-brand h1 {
    font-size: 20px;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    letter-spacing: -.02em;
}
.app-brand .tagline {
    font-size: 12px;
    color: #64748b;
    font-weight: 400;
    margin-left: 4px;
}
.app-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #f1f5f9;
    padding: 6px 16px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
    color: #334155;
}
.app-badge .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
}

/* ── tabs ── */
.tabs {
    border: none !important;
    gap: 0 !important;
}
.tabs > .tab-nav {
    border-bottom: 2px solid #e5e7eb !important;
    background: transparent !important;
    padding: 0 !important;
}
.tabs > .tab-nav button {
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: .01em !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    transition: all .15s !important;
}
.tabs > .tab-nav button:hover {
    color: #1f2937 !important;
    background: transparent !important;
}
.tabs > .tab-nav button.selected {
    color: #4f46e5 !important;
    border-bottom: 2px solid #4f46e5 !important;
    background: transparent !important;
}
.tabs > .tab-panel {
    padding: 20px 0 0 0 !important;
    border: none !important;
}

/* ── cards ── */
.card {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    padding: 20px 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,.03);
}
.card-title {
    font-size: 13px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .04em;
    margin-bottom: 14px;
}

/* ── buttons ── */
.btn-primary {
    background: #4f46e5 !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all .15s !important;
    min-height: 44px !important;
    box-shadow: 0 1px 2px rgba(79,70,229,.15) !important;
}
.btn-primary:hover {
    background: #4338ca !important;
    box-shadow: 0 4px 12px rgba(79,70,229,.25) !important;
    transform: translateY(-1px);
}
.btn-primary:active {
    transform: translateY(0);
}

/* ── inputs ── */
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="file"] {
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color .15s, box-shadow .15s !important;
    background: #ffffff !important;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,.08) !important;
    outline: none !important;
}
.gradio-container textarea::placeholder,
.gradio-container input[type="text"]::placeholder {
    color: #9ca3af !important;
}

/* ── radio ── */
.gradio-container .radio-group {
    background: #f8fafc !important;
    border-radius: 8px !important;
    padding: 4px !important;
    border: 1px solid #e5e7eb !important;
}
.gradio-container .radio-group label {
    padding: 6px 14px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    transition: all .15s !important;
    cursor: pointer !important;
}
.gradio-container .radio-group label.selected {
    background: #ffffff !important;
    color: #1f2937 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.06) !important;
}

/* ── examples ── */
.examples-wrap {
    margin-top: 12px;
}
.examples-wrap .examples-table {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
.examples-wrap .examples-table td {
    padding: 8px 14px !important;
    font-size: 13px !important;
    color: #1f2937 !important;
    cursor: pointer !important;
    transition: background .1s !important;
}
.examples-wrap .examples-table td:hover {
    background: #f1f5f9 !important;
}
.examples-wrap .examples-table .example-label {
    color: #64748b !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: .04em !important;
    background: #f8fafc !important;
}

/* ── results panel ── */
.results-panel {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    padding: 20px 22px;
    min-height: 280px;
}
.results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}
.results-header .title {
    font-size: 13px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .04em;
}
.results-header .badge {
    font-size: 11px;
    color: #9ca3af;
}

/* ── legend ── */
.legend {
    display: flex;
    gap: 14px;
    padding: 10px 16px;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid #f1f5f9;
    margin-top: 10px;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #64748b;
}
.legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

/* ── batch status ── */
.batch-status {
    background: #f8fafc;
    border-radius: 8px;
    padding: 14px 18px;
    border: 1px solid #e5e7eb;
    font-size: 14px;
    color: #1f2937;
    min-height: 52px;
}

/* ── responsive ── */
@media (max-width: 640px) {
    .app-header {
        flex-direction: column;
        align-items: stretch;
        text-align: center;
    }
    .app-brand {
        justify-content: center;
    }
    .app-badge {
        justify-content: center;
    }
}
"""

with gr.Blocks(
    title="Support Ticket Classifier",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    # ── header ──
    gr.HTML("""
    <div class="app-header">
      <div class="app-brand">
        <div class="logo">✦</div>
        <div>
          <h1>Ticket Classifier</h1>
          <span class="tagline">Hierarchical support ticket analysis</span>
        </div>
      </div>
      <div class="app-badge">
        <span class="dot"></span>
        <span>Random Forest · Logistic Regression</span>
      </div>
    </div>
    """)

    # ── tabs ──
    with gr.Tabs(elem_classes="tabs"):

        # ──── SINGLE TAB ────
        with gr.TabItem("Single Ticket", id="single"):
            with gr.Row(equal_height=False):

                # left — inputs
                with gr.Column(scale=5):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title">Ticket details</div>')
                        summary_in = gr.Textbox(
                            label="Summary",
                            placeholder="Brief description of the issue…",
                            lines=1,
                            container=False,
                        )
                        content_in = gr.Textbox(
                            label="Description",
                            placeholder="Detailed explanation of the customer's problem…",
                            lines=7,
                            container=False,
                        )
                        gr.HTML('<div style="height:6px;"></div>')
                        with gr.Row():
                            model_single = gr.Radio(
                                choices=list(CLASSIFIERS.keys()),
                                value="Random Forest",
                                label="Model",
                                container=False,
                                scale=2,
                            )
                            btn = gr.Button(
                                "Analyse ticket",
                                variant="primary",
                                elem_classes="btn-primary",
                                scale=1,
                            )
                        gr.HTML('<div style="height:6px;"></div>')
                        gr.Examples(
                            examples=[
                                ["Refund request",
                                 "I paid for an app but it did not work. Please help me get a refund."],
                                ["Subscription cancellation",
                                 "I want to cancel my subscription and stop future payments."],
                                ["Login issue",
                                 "I cannot sign in to my account after changing my password."],
                            ],
                            inputs=[summary_in, content_in],
                            label="Try an example",
                            elem_classes="examples-wrap",
                        )

                # right — results
                with gr.Column(scale=5):
                    with gr.Group(elem_classes="results-panel"):
                        gr.HTML("""
                        <div class="results-header">
                          <span class="title">Predictions</span>
                          <span class="badge">confidence</span>
                        </div>
                        """)
                        y2_out = gr.HTML(value=EMPTY_CARD)
                        y3_out = gr.HTML()
                        y4_out = gr.HTML()
                        meta_out = gr.HTML()
                        gr.HTML("""
                        <div class="legend">
                          <span class="legend-item"><span class="legend-dot" style="background:#059669;"></span>≥ 70% High</span>
                          <span class="legend-item"><span class="legend-dot" style="background:#d97706;"></span>40–70% Medium</span>
                          <span class="legend-item"><span class="legend-dot" style="background:#dc2626;"></span>&lt; 40% Low</span>
                        </div>
                        """)

            btn.click(
                classify_single,
                inputs=[summary_in, content_in, model_single],
                outputs=[y2_out, y3_out, y4_out, meta_out],
            )

        # ──── BATCH TAB ────
        with gr.TabItem("Batch CSV", id="batch"):
            gr.HTML("""
            <div style="background:#f8fafc;border-radius:8px;padding:14px 18px;margin-bottom:16px;
                        border:1px solid #e5e7eb;font-size:13px;color:#4b5563;line-height:1.6;">
              Upload a CSV with columns <strong>Ticket Summary</strong> and <strong>Interaction content</strong>.
              Results include confidence scores for all three prediction levels.
              <span style="color:#9ca3af;font-size:12px;margin-left:8px;">
                (optional: <code>message_id</code> is preserved)
              </span>
            </div>
            """)
            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title">Upload data</div>')
                        csv_in = gr.File(
                            label="CSV file",
                            file_types=[".csv"],
                            container=False,
                        )
                        gr.HTML('<div style="height:8px;"></div>')
                        model_batch = gr.Radio(
                            choices=list(CLASSIFIERS.keys()),
                            value="Random Forest",
                            label="Model",
                            container=False,
                        )
                        gr.HTML('<div style="height:10px;"></div>')
                        run_btn = gr.Button(
                            "Run batch prediction",
                            variant="primary",
                            elem_classes="btn-primary",
                        )
                with gr.Column(scale=6):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title">Results</div>')
                        status = gr.Markdown(
                            value="Upload a CSV and click **Run batch prediction**.",
                            elem_classes="batch-status",
                        )
                        csv_out = gr.File(label="Download predictions", interactive=False)

            run_btn.click(
                classify_batch,
                inputs=[csv_in, model_batch],
                outputs=[csv_out, status],
            )

    # ── footer ──
    gr.HTML("""
    <div style="text-align:center;padding:28px 0 8px;font-size:12px;color:#9ca3af;
                border-top:1px solid #f1f5f9;margin-top:16px;letter-spacing:.02em;">
      Hierarchical classifier · powered by scikit-learn
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