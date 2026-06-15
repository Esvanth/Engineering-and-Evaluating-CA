"""Load the two CSVs, validate columns, tidy up labels."""
import pandas as pd

from src.config import COLUMNS, DATA, DATA_DIR
from src.logging_utils import get_logger

log = get_logger(__name__)

REQUIRED_COLS = [
    "Ticket id", "Interaction id", "Ticket Summary", "Interaction content",
    "Type 1", "Type 2", "Type 3", "Type 4",
]


def load_raw_dataset(data_dir=DATA_DIR, files=DATA.csv_files):
    """Read each CSV in `files`, concat them, and tidy up label column names."""
    frames = []
    for fname in files:
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

        df = pd.read_csv(path, skipinitialspace=True)

        # quick schema check - fail loud if the file isn't what we expect
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{fname} is missing columns: {missing}")
        if df.empty:
            raise ValueError(f"{fname} has no rows")

        # AppGallery.csv has trailing commas that produce phantom 'Unnamed' columns
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)

        # rename Type 1..4 to y1..y4 - shorter and consistent across the project
        df = df.rename(columns={"Type 1": "y1", "Type 2": "y2",
                                "Type 3": "y3", "Type 4": "y4"})
        df["source_file"] = fname
        log.info("Loaded %s: %d rows", fname, len(df))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined dataset shape: %s", combined.shape)
    return combined


def filter_for_training(df):
    """
    Drop rows with no y2 (root label is required).
    Replace missing y3/y4 with a sentinel so they can still flow through
    the cascade but won't count against accuracy at that level.
    """
    n_before = len(df)
    df = df.copy()

    # y2 is the root of the hierarchy - if it's blank we can't use the row at all
    df = df.loc[df["y2"].notna() & (df["y2"].astype(str).str.strip() != "")]
    dropped = n_before - len(df)
    if dropped:
        log.warning("Dropped %d rows with missing y2 (%.1f%%)",
                    dropped, 100 * dropped / n_before)

    # make text columns strings, never NaN
    for col in COLUMNS.text_cols:
        df[col] = df[col].fillna("").astype(str)

    # tokenise missing y3/y4 instead of dropping the row
    for col in ("y3", "y4"):
        df[col] = df[col].fillna(DATA.missing_label_token).astype(str).str.strip()
        df.loc[df[col] == "", col] = DATA.missing_label_token

    df["y2"] = df["y2"].astype(str).str.strip()
    df["y1"] = df["y1"].astype(str).str.strip()
    return df.reset_index(drop=True)


def data_summary(df):
    """Quick stats for logging / sanity checking."""
    return {
        "n_rows": int(len(df)),
        "n_y1_classes": int(df["y1"].nunique()),
        "y2_distribution": df["y2"].value_counts().to_dict(),
        "y3_distribution": df["y3"].value_counts().head(10).to_dict(),
        "y4_distribution": df["y4"].value_counts().head(10).to_dict(),
        "missing_text_summary": int(df[COLUMNS.ticket_summary].eq("").sum()),
        "missing_text_content": int(df[COLUMNS.interaction_content].eq("").sum()),
    }
