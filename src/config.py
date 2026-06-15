"""Project configuration. Paths, column names, model hyperparameters."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Make sure the output dirs exist on import
for d in (ARTIFACTS_DIR, OUTPUTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


class COLUMNS:
    # raw column names in the CSVs
    ticket_id = "Ticket id"
    interaction_id = "Interaction id"
    ticket_summary = "Ticket Summary"
    interaction_content = "Interaction content"
    # after rename
    group_col = "y1"
    label_cols = ("y2", "y3", "y4")
    text_cols = [ticket_summary, interaction_content]


class DATA:
    csv_files = ("AppGallery.csv", "Purchasing.csv")
    test_size = 0.20
    random_state = 42
    min_samples_per_class = 3
    min_samples_y1_group = 10
    # placeholder for rows where y3/y4 are blank
    missing_label_token = "__MISSING__"


class FEATURES:
    max_features = 2000
    min_df = 4
    max_df = 0.90
    ngram_range = (1, 2)
    sublinear_tf = True


class MODELS:
    random_state = 42
    # Random Forest
    rf_n_estimators = 300
    rf_max_depth = 30
    rf_class_weight = "balanced_subsample"
    # Logistic Regression
    lr_C = 1.0
    lr_max_iter = 1000
    lr_class_weight = "balanced"


class ARTIFACTS:
    vectorizer = "vectorizer.joblib"
    classifier_template = "classifier_{model_name}.joblib"
    metrics_json = "metrics.json"
    classification_reports_csv = "classification_reports.csv"
    chained_accuracy_csv = "chained_accuracy.csv"
    error_analysis_csv = "error_analysis.csv"
    predictions_csv = "predictions.csv"
    confusion_dir = "confusion_matrices"
