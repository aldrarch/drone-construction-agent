"""
Tests for report generation and JSON serialization.
"""

from __future__ import annotations

import json

from schemas.prediction import (
    AnalysisReport,
    GPSStatus,
    ImageAnalysisResult,
    ObjectSummary,
    PredictionResult,
    RiskFlag,
    StageLabel,
)
from services.report_service import build_report, report_to_json, save_report


def test_report_has_required_fields():
    report = build_report(
        project_id="OBJECT_001",
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        summary=ObjectSummary(
            expected_stage=StageLabel.STAGE_50,
            detected_stage=StageLabel.STAGE_50,
            risk_status=RiskFlag.MANUAL_REVIEW,
            manual_review_required=True,
            processed_images=1,
            images_with_gps=1,
            images_needing_review=1,
            contradictory_predictions=False,
            explanation="test",
        ),
        images=[
            ImageAnalysisResult(
                filename="drone_001.jpg",
                predicted_stage=StageLabel.STAGE_50,
                scores={
                    StageLabel.STAGE_0.value: 0.21,
                    StageLabel.STAGE_50.value: 0.63,
                    StageLabel.STAGE_100.value: 0.16,
                },
                score_gap=0.42,
                confidence=0.63,
                manual_review_required=True,
                gps_status=GPSStatus.GPS_FOUND,
            )
        ],
    )

    assert report.project_id == "OBJECT_001"
    assert report.analysis_status == "COMPLETED"
    assert report.model["type"] == "clip"
    assert report.model["name"] == "openai/clip-vit-base-patch32"
    assert report.summary.expected_stage == StageLabel.STAGE_50
    assert report.summary.risk_status == RiskFlag.MANUAL_REVIEW
    assert report.summary.manual_review_required is True
    assert len(report.images) == 1
    assert report.images[0].filename == "drone_001.jpg"
    assert report.images[0].scores[StageLabel.STAGE_50.value] == 0.63
    assert report.images[0].gps_status == GPSStatus.GPS_FOUND
    assert len(report.limitations) >= 2


def test_report_serializes_to_valid_json():
    report = build_report(
        project_id="OBJ",
        model_type="clip",
        model_name="clip",
        summary=ObjectSummary(expected_stage=StageLabel.STAGE_0),
        images=[],
    )
    text = report_to_json(report)
    payload = json.loads(text)
    assert payload["project_id"] == "OBJ"
    assert "generated_at" in payload
    assert "limitations" in payload


def test_save_report_writes_file(tmp_path):
    report = build_report(
        project_id="SAVE_TEST",
        model_type="clip",
        model_name="clip",
        summary=ObjectSummary(expected_stage=StageLabel.STAGE_100),
        images=[],
    )
    out_dir = tmp_path / "reports"
    path = save_report(report, out_dir)
    assert path.is_file()
    assert path.suffix == ".json"
    content = path.read_text(encoding="utf-8")
    payload = json.loads(content)
    assert payload["project_id"] == "SAVE_TEST"


def test_limitation_text_includes_clip_warning():
    report = build_report(
        project_id="L",
        model_type="clip",
        model_name="clip",
        summary=ObjectSummary(expected_stage=StageLabel.STAGE_0),
        images=[],
    )
    text = report_to_json(report)
    assert "general-purpose" in text.lower() or "CLIP" in text
