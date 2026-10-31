"""
Tests for the model interface and a mock model used to avoid loading CLIP.
"""

from __future__ import annotations

import pathlib

import pytest

from model_interface import ConstructionModel
from schemas.prediction import PredictionResult, StageLabel


class MockModel(ConstructionModel):
    """Test double that returns deterministic predictions."""

    def __init__(
        self,
        model_type: str = "mock",
        model_name: str = "mock-model",
        predicted_stage: StageLabel = StageLabel.STAGE_50,
        scores: dict[str, float] | None = None,
        gap: float = 0.5,
        confidence: float = 0.85,
        manual_review: bool = False,
        reasoning: str = "",
    ) -> None:
        self._model_type = model_type
        self._model_name = model_name
        self._predicted_stage = predicted_stage
        self._scores = scores or {
            StageLabel.STAGE_0.value: 0.1,
            StageLabel.STAGE_50.value: 0.8,
            StageLabel.STAGE_100.value: 0.1,
        }
        self._gap = gap
        self._confidence = confidence
        self._manual_review = manual_review
        self._reasoning = reasoning
        self.predict_called = False

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def model_name(self) -> str:
        return self._model_name

    def predict(self, image) -> PredictionResult:
        self.predict_called = True
        return PredictionResult(
            predicted_stage=self._predicted_stage,
            scores=self._scores,
            score_gap=self._gap,
            confidence=self._confidence,
            manual_review_required=self._manual_review,
            model_type=self._model_type,
            model_name=self._model_name,
            reasoning=self._reasoning,
        )


def test_interface_compliance():
    """A mock model must satisfy the ConstructionModel contract."""
    model = MockModel()
    image = pathlib.Path(__file__).parent / "dummy.png"
    result = model.predict(image)
    assert isinstance(result, PredictionResult)
    assert result.predicted_stage == StageLabel.STAGE_50
    assert result.model_type == "mock"
    assert model.predict_called is True


def test_prediction_scores_are_round_trip_safe():
    """PredictionResult should serialize and deserialize without losing data."""
    result = PredictionResult(
        predicted_stage=StageLabel.STAGE_100,
        scores={
            StageLabel.STAGE_0.value: 0.05,
            StageLabel.STAGE_50.value: 0.15,
            StageLabel.STAGE_100.value: 0.80,
        },
        score_gap=0.65,
        confidence=0.80,
        manual_review_required=False,
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        reasoning="ok",
    )
    payload = result.model_dump()
    restored = PredictionResult(**payload)
    assert restored.predicted_stage == StageLabel.STAGE_100
    assert restored.scores[StageLabel.STAGE_100.value] == 0.80


def test_manual_review_flag_propagates():
    """When the mock model sets manual_review=True, the result carries it."""
    model = MockModel(manual_review=True)
    result = model.predict(None)
    assert result.manual_review_required is True


def test_empty_scores_default():
    """PredictionResult should work with empty scores."""
    result = PredictionResult(
        predicted_stage=StageLabel.STAGE_0,
        scores={},
        manual_review_required=True,
        reasoning="no data",
    )
    assert result.scores == {}
    assert result.manual_review_required is True
