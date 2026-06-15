"""TF-IDF features. Fit once on train texts, save to disk, reload at predict time."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import ARTIFACTS, ARTIFACTS_DIR, FEATURES
from src.logging_utils import get_logger

log = get_logger(__name__)


def build_vectorizer():
    return TfidfVectorizer(
        max_features=FEATURES.max_features,
        min_df=FEATURES.min_df,
        max_df=FEATURES.max_df,
        ngram_range=FEATURES.ngram_range,
        sublinear_tf=FEATURES.sublinear_tf,
        strip_accents="unicode",
        lowercase=False,  # we already lowercase in preprocessing
    )


def fit_vectorizer(texts):
    vec = build_vectorizer()
    vec.fit(list(texts))
    log.info("Fitted vectorizer | vocab=%d | ngram=%s",
             len(vec.vocabulary_), FEATURES.ngram_range)
    return vec


def save_vectorizer(vec, path=None):
    path = path or ARTIFACTS_DIR / ARTIFACTS.vectorizer
    joblib.dump(vec, path)
    log.info("Saved vectorizer to %s", path)
    return path


def load_vectorizer(path=None):
    path = path or ARTIFACTS_DIR / ARTIFACTS.vectorizer
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {path}. Run train.py first.")
    vec = joblib.load(path)
    log.info("Loaded vectorizer from %s (vocab=%d)", path, len(vec.vocabulary_))
    return vec


def transform_texts(vec, texts):
    # dense output is fine here - the corpus is tiny (~200 rows).
    # for a real production corpus we'd keep this sparse.
    return vec.transform(list(texts)).toarray()
