"""
Analysis service: orchestrates image validation, GPS check and model prediction.
"""

from __future__ import annotations

import pathlib
from typing import Optional

from PIL import Image

from config import GPS_DISTANCE_THRESHOLD_KM, MANUAL_REVIEW_SCORE_GAP_THRESHOLD
from model_interface import ConstructionModel
from schemas.prediction import (
    GPSStatus,
    ImageAnalysisResult,
    ObjectSummary,
    PredictionResult,
    RiskFlag,
    StageLabel,
)
from services.image_service import ImageLoadError, load_image
from services.gps_service import check_gps


def _filename(path: str) -> str:
    return pathlib.PurePath(path).name


def analyze_single_image(
    file_path: str,
    model: ConstructionModel,
    expected_stage: StageLabel,
    object_latitude: Optional[float],
    object_longitude: Optional[float],
) -> ImageAnalysisResult:
    """Run the full pipeline for one image."""
    error: Optional[str] = None
    image: Optional[Image.Image] = None

    try:
        image = load_image(file_path)
    except ImageLoadError as exc:
        error = str(exc)

    if image is None:
        # Create a dummy image for scoring failures so the pipeline still returns a result.
        from PIL import Image as _Image

        image = _Image.new("RGB", (1, 1))
        error = error or "Image could not be loaded"

    gps_result = check_gps(image, object_latitude, object_longitude)

    prediction = model.predict(image)

    manual_review = (
        prediction.manual_review_required
        or gps_result["gps_status"] == GPSStatus.GPS_MISMATCH
        or (prediction.predicted_stage != expected_stage and not error)
    )

    return ImageAnalysisResult(
        filename=_filename(file_path),
        predicted_stage=prediction.predicted_stage,
        scores=prediction.scores,
        score_gap=prediction.score_gap,
        confidence=prediction.confidence,
        manual_review_required=manual_review,
        gps_status=gps_result["gps_status"],
        gps_latitude=gps_result.get("image_latitude"),
        gps_longitude=gps_result.get("image_longitude"),
        object_latitude=gps_result.get("object_latitude"),
        object_longitude=gps_result.get("object_longitude"),
        distance_km=gps_result.get("distance_km"),
        error=error,
    )


def summarize(
    results: list[ImageAnalysisResult],
    expected_stage: StageLabel,
) -> ObjectSummary:
    """Build an object-level summary from per-image results."""
    if not results:
        return ObjectSummary(
            expected_stage=expected_stage,
            risk_status=RiskFlag.INSUFFICIENT_DATA,
            manual_review_required=True,
            explanation="No images were processed.",
        )

    processed = len(results)
    with_gps = sum(1 for r in results if r.gps_status == GPSStatus.GPS_FOUND)
    needing_review = sum(1 for r in results if r.manual_review_required)

    stages = [r.predicted_stage for r in results]
    unique_stages = set(stages)
    contradictory = len(unique_stages) > 1

    most_common_stage = max(unique_stages, key=stages.count)

    if contradictory:
        risk = RiskFlag.MANUAL_REVIEW
        manual_review_flag = True
        explanation = (
            "Photos show different construction stages. "
            "Manual inspection is required before any conclusion."
        )
    elif most_common_stage == expected_stage and needing_review == 0:
        risk = RiskFlag.APPROVE_FOR_INSPECTOR_REVIEW
        manual_review_flag = False
        explanation = (
            "The majority of photos match the expected stage and no flags were raised. "
            "Still subject to human review."
        )
    elif most_common_stage != expected_stage:
        risk = RiskFlag.RISK_DETECTED
        manual_review_flag = True
        explanation = (
            f"The predicted stage ({most_common_stage.value}) differs from the "
            f"expected stage ({expected_stage.value}). Manual review recommended."
        )
    else:
        risk = RiskFlag.MANUAL_REVIEW
        manual_review_flag = True
        explanation = (
            "Some images require manual review. Final conclusion is not automatic."
        )

    return ObjectSummary(
        expected_stage=expected_stage,
        detected_stage=most_common_stage,
        risk_status=risk,
        manual_review_required=needing_review > 0 or contradictory or manual_review_flag,
        processed_images=processed,
        images_with_gps=with_gps,
        images_needing_review=needing_review,
        contradictory_predictions=contradictory,
        explanation=explanation,
    )
