"""
Chained classifier for the y2 -> y3 -> y4 hierarchy.

Idea: train three single-label models. Each level gets the original text
features PLUS the one-hot encoded label(s) of the previous level(s).

  level 0 (y2):  features = X
  level 1 (y3):  features = [X | onehot(y2)]
  level 2 (y4):  features = [X | onehot(y2) | onehot(y3)]

At training time we use the TRUE parent labels to teach each level. At
inference time we use the model's OWN predicted parent labels, which is
exactly what the chained accuracy metric measures.

The whole cascade looks like a single estimator from the outside: one
fit(X, Y) call, one predict(X) call. Y is shape (n_samples, 3).
"""
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import DATA
from src.logging_utils import get_logger
from src.models.base_model import BaseClassifier

log = get_logger(__name__)


def _one_hot(values, encoder):
    """One-hot encode `values` using the given fitted LabelEncoder.

    Anything not in encoder.classes_ becomes the all-zeros row, which is what
    we want when the cascade predicts something rare that the next level
    has never seen alongside in training.
    """
    n = len(values)
    n_classes = len(encoder.classes_)
    out = np.zeros((n, n_classes), dtype=np.float32)
    idx_map = {c: i for i, c in enumerate(encoder.classes_)}
    for r, v in enumerate(values):
        col = idx_map.get(v)
        if col is not None:
            out[r, col] = 1.0
    return out


class ChainedHierarchicalClassifier:
    def __init__(self, level_estimators,
                 level_names=("y2", "y3", "y4"),
                 missing_token=DATA.missing_label_token):
        if len(level_estimators) != len(level_names):
            raise ValueError("Need one estimator per level")
        self.level_estimators = list(level_estimators)
        self.level_names = list(level_names)
        self.missing_token = missing_token
        self.encoders = []
        self._fitted = False

    @property
    def name(self):
        # eg "random_forest+random_forest+random_forest"
        return "+".join(e.name for e in self.level_estimators)

    # -----------------------------------------------------------------
    def fit(self, X, Y):
        """
        Fit the cascade. Y must be shape (n_samples, n_levels).

        Rows whose label at the current level is the missing-token are
        skipped when fitting THAT level only.
        """
        if Y.shape[1] != len(self.level_estimators):
            raise ValueError(
                f"Y must have {len(self.level_estimators)} columns, got {Y.shape[1]}"
            )

        # Fit label encoders for each level - skipping missing values when
        # collecting the class list.
        self.encoders = []
        for level in range(Y.shape[1]):
            enc = LabelEncoder()
            vals = Y[:, level]
            mask = vals != self.missing_token
            unique = np.unique(vals[mask]) if mask.any() else np.array([self.missing_token])
            enc.fit(unique)
            self.encoders.append(enc)

        # Train each level. After each level we append the TRUE one-hot
        # to the feature matrix for the next level.
        feats = X.astype(np.float32)
        for level, est in enumerate(self.level_estimators):
            y_level = Y[:, level]
            mask = y_level != self.missing_token

            if mask.sum() < 2 or len(np.unique(y_level[mask])) < 2:
                # not enough data to learn at this level - fit on whatever
                # we have so the estimator at least exposes a classes_ list
                log.warning(
                    "Level %s has insufficient data (%d rows, %d classes)",
                    self.level_names[level], int(mask.sum()),
                    len(np.unique(y_level[mask])),
                )
                X_tr = feats[mask] if mask.any() else feats[:1]
                y_tr = y_level[mask] if mask.any() else np.array([self.missing_token])
            else:
                X_tr = feats[mask]
                y_tr = y_level[mask]

            est.fit(X_tr, y_tr)
            log.info("Fitted %s | n_train=%d | n_classes=%d | %s",
                     self.level_names[level], len(y_tr),
                     len(np.unique(y_tr)), est.name)

            # extend the feature matrix for the next level (if any)
            if level + 1 < len(self.level_estimators):
                feats = np.hstack([feats, _one_hot(y_level, self.encoders[level])])

        self._fitted = True
        return self

    # -----------------------------------------------------------------
    def predict(self, X):
        preds, _ = self.predict_with_confidence(X)
        return preds

    def predict_with_confidence(self, X):
        """Cascade predict. Returns (labels, confidences), each (n_samples, n_levels)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        n = X.shape[0]
        n_levels = len(self.level_estimators)
        preds = np.empty((n, n_levels), dtype=object)
        confs = np.zeros((n, n_levels), dtype=np.float32)

        feats = X.astype(np.float32)
        for level, est in enumerate(self.level_estimators):
            y_hat = est.predict(feats)
            proba = est.predict_proba(feats)

            if proba is not None:
                # confidence = probability of the predicted class
                col_map = {c: i for i, c in enumerate(est.classes_)}
                cols = np.array([col_map[c] for c in y_hat])
                confs[:, level] = proba[np.arange(n), cols]
            else:
                confs[:, level] = np.nan

            preds[:, level] = y_hat

            # feed predicted label into the next level's feature matrix
            if level + 1 < n_levels:
                feats = np.hstack([feats, _one_hot(y_hat, self.encoders[level])])

        return preds, confs

    # -----------------------------------------------------------------
    def save(self, path):
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted classifier")
        joblib.dump(self, path)
        log.info("Saved chained classifier to %s", path)

    @staticmethod
    def load(path):
        return joblib.load(path)
