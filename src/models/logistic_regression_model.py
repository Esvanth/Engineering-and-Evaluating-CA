"""Logistic Regression wrapper for the cascade."""
from sklearn.linear_model import LogisticRegression

from src.config import MODELS
from src.models.base_model import BaseClassifier


class LogisticRegressionModel(BaseClassifier):
    name = "logistic_regression"

    def __init__(self, C=MODELS.lr_C, max_iter=MODELS.lr_max_iter,
                 class_weight=MODELS.lr_class_weight,
                 random_state=MODELS.random_state):
        self._estimator = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            solver="lbfgs",
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
