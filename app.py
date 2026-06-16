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
    "high":   {"fg": "#15803d", "bg": "#f0fdf4", "bar": "#22c55e", "badge": "#dcfce7"},
    "mid":    {"fg": "#b45309", "bg": "#fffbeb", "bar": "#f59e0b", "badge": "#fef3c7"},
    "low":    {"fg": "#b91c1c", "bg": "#fef2f2", "bar": "#ef4444", "badge": "#fee2e2"},
}

LEVEL_META = {
    "Type 2": {"icon": "📂", "subtitle": "Top-level category"},
    "Type 3": {"icon": "📌", "subtitle": "Mid-level topic"},
    "Type 4": {"icon": "🏷️", "subtitle": "Specific issue"},
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
        border: 1px solid #e2e8f0;
        border-left: 4px solid {p['bar']};
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
        transition: box-shadow .2s;
    " onmouseenter="this.style.boxShadow='0 4px 16px rgba(0,0,0,.07)'"
       onmouseleave="this.style.boxShadow='none'">

      <!-- header row -->
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:20px;">{meta['icon']}</span>
          <div>
            <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:.06em;">{level_key} — {meta['subtitle']}</div>
            <div style="font-size:17px;font-weight:800;color:#1e293b;margin-top:2px;">{label}</div>
          </div>
        </div>
        <div style="background:{p['badge']};color:{p['fg']};font-size:13px;font-weight:700;
                    padding:4px 12px;border-radius:99px;white-space:nowrap;">{pct}%</div>
      </div>

      <!-- progress bar -->
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="flex:1;background:#f1f5f9;border-radius:99px;height:5px;overflow:hidden;">
          <div style="width:{pct}%;background:{p['bar']};height:5px;border-radius:99px;
                      transition:width .4s ease;"></div>
        </div>
        <span style="font-size:10px;font-weight:600;color:{p['fg']};white-space:nowrap;">{word}</span>
      </div>
    </div>
    """


EMPTY_CARD = """
<div style="background:#f8fafc;border:2px dashed #cbd5e1;border-radius:14px;
            padding:56px 24px;text-align:center;">
  <div style="font-size:40px;margin-bottom:14px;">🎯</div>
  <div style="font-size:16px;font-weight:700;color:#475569;margin-bottom:6px;">
    Awaiting your ticket
  </div>
  <div style="font-size:13px;color:#94a3b8;max-width:280px;margin:0 auto;line-height:1.5;">
    Enter a summary and description on the left, then press <b>Classify</b>.
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
        f'<div style="font-size:11px;color:#94a3b8;padding:6px 4px;display:flex;'
        f'align-items:center;gap:6px;">'
        f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
        f'background:#22c55e;"></span>'
        f'Model: <b>{model_choice}</b> &nbsp;·&nbsp; '
        f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC</div>'
    )
    return *cards, stamp


# ── batch mode ──────────────────────────────────────────────────────────────

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


# ── UI ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500&display=swap');

/* ── globals ── */
.gradio-container {
    max-width: 1020px !important;
    margin: 0 auto !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #f8fafc !important;
}
.dark .gradio-container { background: #0f172a !important; }
footer { display: none !important; }

/* ── tab strip ── */
.tab-nav { border-bottom: 2px solid #e2e8f0 !important; }
.tab-nav button {
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: .02em !important;
    padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important;
}
.tab-nav button.selected {
    background: #fff !important;
    border-bottom: 2px solid #6366f1 !important;
    color: #6366f1 !important;
}

/* ── primary buttons ── */
#classify-btn, #run-btn {
    background: #6366f1 !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    box-shadow: 0 1px 3px rgba(99,102,241,.25) !important;
    transition: background .15s, box-shadow .15s !important;
    letter-spacing: .01em !important;
}
#classify-btn:hover, #run-btn:hover {
    background: #4f46e5 !important;
    box-shadow: 0 4px 14px rgba(99,102,241,.35) !important;
}

/* ── inputs ── */
.gradio-container textarea, .gradio-container input[type="text"] {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 14px !important;
    transition: border-color .15s !important;
}
.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.12) !important;
}

/* ── metric cards ── */
.stat-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 14px;
    text-align: center;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -.5px;
}
.stat-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    margin-top: 4px;
}
.stat-sub {
    font-size: 10px;
    color: #94a3b8;
    margin-top: 2px;
}

/* ── legend chips ── */
.legend-row {
    display: flex; gap: 16px; margin-top: 6px;
    padding: 8px 14px; background: #f8fafc;
    border-radius: 8px; border: 1px solid #f1f5f9;
}
.legend-chip {
    font-size: 11px; color: #64748b;
    display: flex; align-items: center; gap: 5px;
}
.legend-dot {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block;
}

/* ── accordion ── */
.gradio-accordion { border-radius: 10px !important; border: 1px solid #e2e8f0 !important; }
"""

with gr.Blocks(
    title="Customer Support Ticket Classifier",
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
    <div style="text-align:center;padding:32px 0 8px;">

      <div style="display:inline-flex;align-items:center;gap:6px;
                  background:#eef2ff;border:1px solid #c7d2fe;border-radius:99px;
                  padding:5px 14px;font-size:11px;font-weight:600;color:#4f46e5;
                  letter-spacing:.03em;margin-bottom:16px;">
        <span style="font-size:14px;">🎓</span>
        MSc AI · Engineering &amp; Evaluating AI Systems
      </div>

      <h1 style="font-size:2rem;font-weight:900;margin:0 0 8px;color:#0f172a;
                 line-height:1.15;letter-spacing:-.02em;">
        Support Ticket Classifier
      </h1>

      <p style="color:#64748b;margin:0 auto;font-size:14px;max-width:500px;line-height:1.6;">
        Hierarchical model that predicts <strong style="color:#4f46e5;">three category levels</strong>
        for customer-support tickets — powered by Random Forest &amp; Logistic Regression.
      </p>

      <div style="height:1px;background:linear-gradient(90deg,transparent 0%,#e2e8f0 30%,#e2e8f0 70%,transparent 100%);
                  margin:22px auto 0;max-width:600px;"></div>
    </div>
    """)

    # ── metrics accordion ──
    with gr.Accordion("Model performance on held-out test set (42 rows)", open=False):
        with gr.Row():
            gr.HTML(
                '<div class="stat-card">'
                '<div class="stat-val" style="color:#6366f1;">69.4%</div>'
                '<div class="stat-label">Chained Accuracy</div>'
                '<div class="stat-sub">Logistic Regression</div></div>'
            )
            gr.HTML(
                '<div class="stat-card">'
                '<div class="stat-val" style="color:#6366f1;">68.3%</div>'
                '<div class="stat-label">Chained Accuracy</div>'
                '<div class="stat-sub">Random Forest</div></div>'
            )
            gr.HTML(
                '<div class="stat-card">'
                '<div class="stat-val" style="color:#15803d;">83.3%</div>'
                '<div class="stat-label">Type 2 Macro F1</div>'
                '<div class="stat-sub">Random Forest</div></div>'
            )
            gr.HTML(
                '<div class="stat-card">'
                '<div class="stat-val" style="color:#b45309;">206</div>'
                '<div class="stat-label">Training Samples</div>'
                '<div class="stat-sub">AppGallery domain</div></div>'
            )
        gr.HTML(
            '<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;'
            'padding:10px 14px;font-size:12px;color:#92400e;margin-top:6px;line-height:1.5;">'
            '⚠️ <strong>Research prototype</strong> — trained on a small dataset. '
            'Predictions with confidence below 50 % should be treated with caution.'
            '</div>'
        )

    gr.HTML("<div style='height:6px'></div>")

    # ── single ticket ──
    with gr.Tab("Single Ticket"):
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
                        "Classify →",
                        variant="primary",
                        elem_id="classify-btn",
                        scale=1,
                    )
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
                )

            with gr.Column(scale=5):
                gr.HTML(
                    '<div style="font-size:11px;font-weight:700;color:#94a3b8;'
                    'text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;'
                    'display:flex;align-items:center;gap:8px;">'
                    'Prediction Results'
                    '<span style="flex:1;height:1px;background:#e2e8f0;"></span></div>'
                )
                y2_out = gr.HTML(value=EMPTY_CARD)
                y3_out = gr.HTML()
                y4_out = gr.HTML()
                meta_out = gr.HTML()
                gr.HTML(
                    '<div class="legend-row">'
                    '<span class="legend-chip"><span class="legend-dot" style="background:#22c55e;"></span>≥ 70 % High</span>'
                    '<span class="legend-chip"><span class="legend-dot" style="background:#f59e0b;"></span>40–70 % Medium</span>'
                    '<span class="legend-chip"><span class="legend-dot" style="background:#ef4444;"></span>&lt; 40 % Low</span>'
                    '</div>'
                )

        btn.click(
            classify_single,
            inputs=[summary_in, content_in, model_single],
            outputs=[y2_out, y3_out, y4_out, meta_out],
        )

    # ── batch csv ──
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
                run_btn = gr.Button(
                    "Run Predictions →", variant="primary", elem_id="run-btn",
                )
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