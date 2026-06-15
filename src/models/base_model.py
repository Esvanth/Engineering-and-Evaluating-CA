"""Abstract base for any single-label classifier we plug into the cascade."""
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """
    Minimal interface our chained classifier needs from each level estimator.
    Anything implementing fit, predict, predict_proba and exposing classes_
    can be slotted in.
    """
    name = "base"

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    def predict_proba(self, X):
        # default: model doesn't expose calibrated probabilities
        return None

    @property
    @abstractmethod
    def classes_(self):
        ...
