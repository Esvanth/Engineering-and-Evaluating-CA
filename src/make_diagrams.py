"""One-off script to generate the two architecture diagrams used in the report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent.parent / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _box(ax, xy, w, h, text, fc="white", ec="black", fontsize=9, bold=False):
    """Draw a rounded box with text inside it."""
    rect = patches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal", wrap=True,
    )


def _arrow(ax, start, end, text=None, color="black", connectionstyle="arc3,rad=0"):
    ax.annotate(
        "", xy=end, xycoords="data", xytext=start, textcoords="data",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                        connectionstyle=connectionstyle),
    )
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.05, text, fontsize=7.5, ha="center", style="italic")


def training_pipeline_diagram(path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(6, 7.6, "Training Pipeline Architecture",
            fontsize=14, fontweight="bold", ha="center")

    # raw data sources (left)
    _box(ax, (0.3, 6.0), 2.0, 0.7, "AppGallery.csv", fc="#f0f0f0")
    _box(ax, (0.3, 5.1), 2.0, 0.7, "Purchasing.csv", fc="#f0f0f0")
    ax.text(1.3, 4.85, "Raw data sources", fontsize=8, ha="center", style="italic")

    # main horizontal flow
    _box(ax, (3.0, 5.45), 2.0, 0.9, "Data Loader\n+ Schema Validation", bold=True)
    _arrow(ax, (2.3, 6.35), (3.0, 5.9))
    _arrow(ax, (2.3, 5.45), (3.0, 5.7))

    _box(ax, (5.7, 5.45), 2.0, 0.9, "Preprocessing\n(dedup + clean)", bold=True)
    _arrow(ax, (5.0, 5.9), (5.7, 5.9))

    _box(ax, (8.4, 5.45), 2.0, 0.9, "Train/Test Split\n(stratified 80/20)", bold=True)
    _arrow(ax, (7.7, 5.9), (8.4, 5.9))

    # ML stack - second row
    _box(ax, (8.4, 3.8), 2.0, 0.9, "TF-IDF Vectorizer\n(fit on TRAIN only)",
         bold=True, fc="#e8f0e8")
    _arrow(ax, (9.4, 5.45), (9.4, 4.7))

    _box(ax, (5.7, 3.8), 2.0, 0.9, "Chained Classifier\ny2 -> y3 -> y4",
         bold=True, fc="#e8f0e8")
    _arrow(ax, (8.4, 4.25), (7.7, 4.25))

    # two underlying models hang off the cascade box
    _box(ax, (5.7, 2.6), 0.95, 0.65, "RF", fontsize=8, fc="#f5f5f5")
    _box(ax, (6.75, 2.6), 0.95, 0.65, "LR", fontsize=8, fc="#f5f5f5")
    _arrow(ax, (6.7, 3.8), (6.17, 3.25))
    _arrow(ax, (6.7, 3.8), (7.22, 3.25))

    _box(ax, (3.0, 3.8), 2.0, 0.9, "Evaluation\n(chained acc + F1)",
         bold=True, fc="#e8f0e8")
    _arrow(ax, (5.7, 4.25), (5.0, 4.25))

    # persisted outputs (bottom row)
    _box(ax, (0.3, 1.4), 2.4, 0.85,
         "Artefacts\nvectorizer.joblib\nclassifier_*.joblib", fc="#fffacd")
    _box(ax, (3.1, 1.4), 2.4, 0.85,
         "Outputs\nmetrics.json\nclassification_reports.csv", fc="#fffacd")
    _box(ax, (5.9, 1.4), 2.4, 0.85,
         "Outputs\ntest_predictions.csv\nerror_analysis.csv", fc="#fffacd")
    _box(ax, (8.7, 1.4), 2.4, 0.85,
         "Outputs\nconfusion_matrices/\nlogs/training.log", fc="#fffacd")

    _arrow(ax, (4.0, 3.8), (4.0, 2.3))
    _arrow(ax, (9.4, 3.8), (9.4, 2.3))
    _arrow(ax, (6.7, 3.8), (6.7, 2.3))
    _arrow(ax, (6.7, 2.6), (1.5, 2.3), connectionstyle="arc3,rad=-0.2")

    ax.text(6, 0.7,
            "Legend:  white = stage,  green = ML core,  yellow = persisted artefacts",
            fontsize=8, ha="center", style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def inference_monitoring_diagram(path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(7, 9.5, "Inference and Monitoring Architecture",
            fontsize=14, fontweight="bold", ha="center")

    # ---- Lane 1: Inference path ----
    ax.text(0.4, 8.7, "1. Inference path", fontsize=10, fontweight="bold")
    ax.axhline(8.5, xmin=0.02, xmax=0.98, color="grey", linestyle=":", linewidth=0.6)

    _box(ax, (0.3, 7.3), 2.1, 0.8, "New messages\nCSV (batch)", fc="#f0f0f0")
    _box(ax, (3.0, 7.3), 2.3, 0.8, "Predict API\n(predict.py)", bold=True)
    _arrow(ax, (2.4, 7.7), (3.0, 7.7))

    _box(ax, (5.9, 7.3), 2.4, 0.8, "preprocess_message()\nshared with train",
         bold=True, fc="#e8f0e8")
    _arrow(ax, (5.3, 7.7), (5.9, 7.7))

    _box(ax, (8.9, 7.3), 2.2, 0.8, "Load vectorizer\n+ classifier", fc="#fffacd")
    _arrow(ax, (8.3, 7.7), (8.9, 7.7))

    _box(ax, (11.7, 7.3), 2.0, 0.8, "Chained\npredict", bold=True, fc="#e8f0e8")
    _arrow(ax, (11.1, 7.7), (11.7, 7.7))

    _box(ax, (11.7, 5.9), 2.0, 0.8, "Predictions\nCSV", fc="#f0f0f0")
    _arrow(ax, (12.7, 7.3), (12.7, 6.7))

    # ---- Lane 2: Logging + observability ----
    ax.axhline(5.5, xmin=0.02, xmax=0.98, color="grey", linestyle=":", linewidth=0.6)
    ax.text(0.4, 5.2, "2. Logging and observability", fontsize=10, fontweight="bold")

    _box(ax, (0.3, 3.9), 2.6, 0.9,
         "Prediction Logs\nmsg_id, pred, conf,\nmodel_version, ts", fc="#fffacd")
    _arrow(ax, (12.7, 5.9), (1.6, 4.8), connectionstyle="arc3,rad=-0.15")

    _box(ax, (3.4, 3.9), 2.6, 0.9,
         "Drift Monitor\ninput stats vs.\ntraining baseline", bold=True)
    _arrow(ax, (2.9, 4.35), (3.4, 4.35))

    _box(ax, (6.5, 3.9), 2.6, 0.9,
         "Performance\nMonitor\non labelled subset", bold=True)
    _arrow(ax, (6.0, 4.35), (6.5, 4.35))

    _box(ax, (9.6, 3.9), 2.0, 0.9, "Alerting\n(thresholds)", bold=True, fc="#ffe0e0")
    _arrow(ax, (9.1, 4.35), (9.6, 4.35))

    _box(ax, (12.0, 3.9), 1.7, 0.9, "Human\nReview Queue", fc="#ffe0e0")
    _arrow(ax, (11.6, 4.35), (12.0, 4.35))

    # ---- Lane 3: Retraining + registry ----
    ax.axhline(3.5, xmin=0.02, xmax=0.98, color="grey", linestyle=":", linewidth=0.6)
    ax.text(0.4, 3.2, "3. Retraining and model registry",
            fontsize=10, fontweight="bold")

    _box(ax, (0.3, 1.8), 2.6, 1.0, "Labelled\nFeedback Store", fc="#fffacd")
    _arrow(ax, (12.8, 3.9), (1.6, 2.8), connectionstyle="arc3,rad=-0.25",
           text="approved labels")

    _box(ax, (3.4, 1.8), 3.4, 1.0,
         "Retraining Trigger\nscheduled  |  drift  |\nperformance drop",
         bold=True, fc="#e8f0e8")
    _arrow(ax, (1.6, 1.8), (3.4, 2.3))
    _arrow(ax, (4.7, 3.9), (4.7, 2.8))
    _arrow(ax, (7.8, 3.9), (5.5, 2.8))

    _box(ax, (7.3, 1.8), 2.8, 1.0, "Training Pipeline\nrun end-to-end", bold=True)
    _arrow(ax, (6.8, 2.3), (7.3, 2.3))

    _box(ax, (10.5, 1.8), 3.2, 1.0,
         "Model Registry\nversioned artefacts", fc="#fffacd")
    _arrow(ax, (10.1, 2.3), (10.5, 2.3))

    # registry back to "Load vectorizer + classifier"
    _arrow(ax, (12.1, 2.8), (10.0, 7.3), connectionstyle="arc3,rad=0.3",
           text="serves new\nversion")

    ax.text(7, 0.6,
            "Legend:  white = service/process,  green = ML core,  "
            "yellow = persisted data,  red = human or alert",
            fontsize=8, ha="center", style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    training_pipeline_diagram(OUT_DIR / "training_pipeline.png")
    inference_monitoring_diagram(OUT_DIR / "inference_monitoring.png")
    print("Wrote diagrams to", OUT_DIR)
