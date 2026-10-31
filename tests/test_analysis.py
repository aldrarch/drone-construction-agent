"""
Tests for the analysis pipeline using a mock model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from PIL import Image
from schemas.prediction import (
    GPSStatus,
    ImageAnalysisResult,
    ObjectSummary,
    StageLabel,
    RiskFlag,
)
from services.analysis_service import analyze_single_image, summarize
from tests.test_model_interface import MockModel


def _create_test_image(width: int = 64, height: int = 64) -> Path:
    """Create a small on-disk RGB image for test scenarios."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp, format="PNG")
    tmp.close()
    return Path(tmp.name)


def test_image_load_valid_png():
    path = _create_test_image()
    try:
        img = Image.open(path)
        assert img.size == (64, 64)
    finally:
        path.unlink(missing_ok=True)


def test_invalid_file_rejected():
    from services.image_service import ImageLoadError, validate_image_path

    with pytest.raises(ImageLoadError):
        _ = validate_image_path(_create_test_image().parent / "nonexistent.jpg")


def test_analyze_valid_image_with_mock_model():
    path = _create_test_image()
    model = MockModel(
        predicted_stage=StageLabel.STAGE_50,
        scores={StageLabel.STAGE_0.value: 0.2, StageLabel.STAGE_50.value: 0.7, StageLabel.STAGE_100.value: 0.1},
        gap=0.5,
        confidence=0.7,
        manual_review=False,
    )
    try:
        result = analyze_single_image(
            str(path),
            model=model,
            expected_stage=StageLabel.STAGE_50,
            object_latitude=55.0,
            object_longitude=37.0,
        )
        assert result.filename == path.name
        assert result.predicted_stage == StageLabel.STAGE_50
        assert result.scores[StageLabel.STAGE_50.value] == 0.7
        assert result.gps_status == GPSStatus.GPS_NOT_AVAILABLE
        assert result.manual_review_required is False
    finally:
        path.unlink(missing_ok=True)


def test_summarize_no_images():
    summary = summarize([], expected_stage=StageLabel.STAGE_50)
    assert summary.processed_images == 0
    assert summary.risk_status == RiskFlag.INSUFFICIENT_DATA
    assert summary.manual_review_required is True


def test_summarize_contradictory_predictions():
    results = [
        ImageAnalysisResult(
            filename="a.jpg",
            predicted_stage=StageLabel.STAGE_0,
            scores={StageLabel.STAGE_0.value: 0.9, StageLabel.STAGE_50.value: 0.05, StageLabel.STAGE_100.value: 0.05},
            score_gap=0.85,
            confidence=0.9,
            manual_review_required=True,
            gps_status=GPSStatus.GPS_NOT_AVAILABLE,
        ),
        ImageAnalysisResult(
            filename="b.jpg",
            predicted_stage=StageLabel.STAGE_100,
            scores={StageLabel.STAGE_0.value: 0.05, StageLabel.STAGE_50.value: 0.05, StageLabel.STAGE_100.value: 0.9},
            score_gap=0.85,
            confidence=0.9,
            manual_review_required=True,
            gps_status=GPSStatus.GPS_NOT_AVAILABLE,
        ),
    ]
    summary = summarize(results, expected_stage=StageLabel.STAGE_50)
    assert summary.contradictory_predictions is True
    assert summary.risk_status == RiskFlag.MANUAL_REVIEW


def test_summarize_agreeing_predictions_match_expected():
    results = [
        ImageAnalysisResult(
            filename="a.jpg",
            predicted_stage=StageLabel.STAGE_50,
            scores={StageLabel.STAGE_0.value: 0.1, StageLabel.STAGE_50.value: 0.8, StageLabel.STAGE_100.value: 0.1},
            score_gap=0.7,
            confidence=0.8,
            manual_review_required=False,
            gps_status=GPSStatus.GPS_NOT_AVAILABLE,
        ),
        ImageAnalysisResult(
            filename="b.jpg",
            predicted_stage=StageLabel.STAGE_50,
            scores={StageLabel.STAGE_0.value: 0.1, StageLabel.STAGE_50.value: 0.8, StageLabel.STAGE_100.value: 0.1},
            score_gap=0.7,
            confidence=0.8,
            manual_review_required=False,
            gps_status=GPSStatus.GPS_NOT_AVAILABLE,
        ),
    ]
    summary = summarize(results, expected_stage=StageLabel.STAGE_50)
    assert summary.detected_stage == StageLabel.STAGE_50
    assert summary.risk_status == RiskFlag.APPROVE_FOR_INSPECTOR_REVIEW
    assert summary.manual_review_required is False


def test_summarize_mismatch_with_expected():
    results = [
        ImageAnalysisResult(
            filename="a.jpg",
            predicted_stage=StageLabel.STAGE_0,
            scores={StageLabel.STAGE_0.value: 0.9, StageLabel.STAGE_50.value: 0.05, StageLabel.STAGE_100.value: 0.05},
            score_gap=0.85,
            confidence=0.9,
            manual_review_required=False,
            gps_status=GPSStatus.GPS_NOT_AVAILABLE,
        ),
    ]
    summary = summarize(results, expected_stage=StageLabel.STAGE_50)
    assert summary.detected_stage == StageLabel.STAGE_0
    assert summary.risk_status == RiskFlag.RISK_DETECTED
    assert summary.manual_review_required is True
