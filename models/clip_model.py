"""
CLIP-based construction stage predictor.

Uses openai/clip-vit-base-patch32 in a zero-shot classification setup.
Compares the image against three text descriptions of construction stages.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from PIL import Image
from transformers import CLIPModel as HFCLIPModel, CLIPProcessor

from model_interface import ConstructionModel
from schemas.prediction import (
    GPSStatus,
    PredictionResult,
    StageLabel,
)
from utils.device import get_device
import config as app_config


class CLIPModel(ConstructionModel):
    """Wrap the Hugging Face CLIP model for zero-shot stage classification."""

    # Stage descriptions used as zero-shot classes.
    STAGE_DESCRIPTIONS: ClassVar[dict[StageLabel, list[str]]] = {
        StageLabel.STAGE_0: [
            "empty construction plot",
            "land with only foundation",
            "no completed residential house",
        ],
        StageLabel.STAGE_50: [
            "unfinished residential house",
            "house under construction",
            "incomplete walls or structural frame",
            "unfinished roof or exterior",
        ],
        StageLabel.STAGE_100: [
            "completed residential house",
            "finished roof, windows and doors",
            "completed exterior of a residential building",
        ],
    }

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self._model_name = model_name
        self._device = get_device()
        self._model: HFCLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._loaded = False

    @property
    def model_type(self) -> str:
        return "clip"

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        """Load the CLIP model and processor. Safe to call multiple times."""
        if self._loaded:
            return
        self._model = HFCLIPModel.from_pretrained(self._model_name)
        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()
        self._loaded = True

    def predict(self, image: Image.Image) -> PredictionResult:
        """Classify the image into one of the construction stages."""
        if not self._loaded:
            self.load()

        if self._model is None or self._processor is None:
            return PredictionResult(
                predicted_stage=StageLabel.STAGE_0,
                scores={},
                manual_review_required=True,
                reasoning="Model not loaded",
            )

        try:
            inputs = self._processor(
                text=[desc for descs in self.STAGE_DESCRIPTIONS.values() for desc in descs],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self._model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.softmax(logits_per_image, dim=1)

            # Aggregate scores per stage (max similarity across its descriptions).
            stage_scores: dict[StageLabel, float] = {}
            text_ids = list(self.STAGE_DESCRIPTIONS.keys())
            offset = 0
            for stage in text_ids:
                count = len(self.STAGE_DESCRIPTIONS[stage])
                stage_probs = probs[0, offset : offset + count]
                stage_scores[stage] = float(stage_probs.max().item())
                offset += count

            sorted_stages = sorted(stage_scores.items(), key=lambda x: x[1], reverse=True)
            top_stage, top_score = sorted_stages[0]
            second_stage, second_score = sorted_stages[1] if len(sorted_stages) > 1 else (None, 0.0)

            gap = top_score - second_score if second_stage is not None else 0.0
            confidence = top_score
            manual_review = gap < app_config.MANUAL_REVIEW_SCORE_GAP_THRESHOLD

            return PredictionResult(
                predicted_stage=top_stage,
                scores={stage.value: round(score, 4) for stage, score in stage_scores.items()},
                score_gap=round(gap, 4),
                confidence=round(confidence, 4),
                manual_review_required=manual_review,
                model_type=self.model_type,
                model_name=self.model_name,
                reasoning=(
                    "Low distinction between top classes; recommend human review."
                    if manual_review
                    else "Model prediction is reasonably distinct."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return PredictionResult(
                predicted_stage=StageLabel.STAGE_0,
                scores={},
                manual_review_required=True,
                reasoning=f"Prediction error: {exc}",
            )
