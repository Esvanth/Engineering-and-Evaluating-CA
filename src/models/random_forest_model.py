"""Random Forest wrapper that matches the BaseClassifier interface."""
from sklearn.ensemble import RandomForestClassifier

from src.config import MODELS
from src.models.base_model import BaseClassifier


class RandomForestModel(BaseClassifier):
    name = "random_forest"

    def __init__(self, n_estimators=MODELS.rf_n_estimators,
                 max_depth=MODELS.rf_max_depth,
                 class_weight=MODELS.rf_class_weight,
                 random_state=MODELS.random_state):
        self._estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self._estimator.fit(X, y)
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)

    @property
    def classes_(self):
        return self._estimator.classes_
