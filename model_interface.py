"""
Universal model interface for construction stage prediction.

All model implementations must implement this interface so that
app.py, services, GPS module and UI do not depend on a concrete model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL.Image import Image as PILImage

from schemas.prediction import PredictionResult, StageLabel


class ConstructionModel(ABC):
    """Abstract interface for a construction-stage prediction model."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Short identifier of the model backend, e.g. 'clip'."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name used in reports."""

    @abstractmethod
    def predict(self, image: PILImage) -> PredictionResult:
        """Return a prediction for the given PIL image.

        Implementations must never raise unhandled exceptions for valid
        images; errors should be wrapped into a PredictionResult with
        manual_review_required=True when possible.
        """


def create_model(config: Any) -> ConstructionModel:
    """Factory function used by the application to build a model instance.

    Swap this function (or the config.MODEL_TYPE value) when you add a
    custom model. The default implementation uses CLIP.
    """
    from models.model_factory import build_model

    return build_model(config)
