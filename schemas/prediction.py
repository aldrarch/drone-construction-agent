"""
Pydantic schemas used across the project, especially prediction results.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StageLabel(str, Enum):
    STAGE_0 = "STAGE_0"
    STAGE_50 = "STAGE_50"
    STAGE_100 = "STAGE_100"


class GPSStatus(str, Enum):
    GPS_FOUND = "GPS_FOUND"
    GPS_NOT_AVAILABLE = "GPS_NOT_AVAILABLE"
    GPS_MATCH = "GPS_MATCH"
    GPS_MISMATCH = "GPS_MISMATCH"


class RiskFlag(str, Enum):
    APPROVE_FOR_INSPECTOR_REVIEW = "APPROVE_FOR_INSPECTOR_REVIEW"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    RISK_DETECTED = "RISK_DETECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class PredictionResult(BaseModel):
    predicted_stage: StageLabel
    scores: dict[str, float] = Field(default_factory=dict)
    score_gap: float = 0.0
    confidence: float = 0.0
    manual_review_required: bool = False
    model_type: str = "unknown"
    model_name: str = "unknown"
    reasoning: str = ""


class ImageAnalysisResult(BaseModel):
    filename: str
    predicted_stage: StageLabel
    scores: dict[str, float]
    score_gap: float
    confidence: float
    manual_review_required: bool
    gps_status: GPSStatus
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    object_latitude: Optional[float] = None
    object_longitude: Optional[float] = None
    distance_km: Optional[float] = None
    error: Optional[str] = None


class ObjectSummary(BaseModel):
    expected_stage: StageLabel
    detected_stage: Optional[StageLabel] = None
    risk_status: RiskFlag = RiskFlag.INSUFFICIENT_DATA
    manual_review_required: bool = True
    processed_images: int = 0
    images_with_gps: int = 0
    images_needing_review: int = 0
    contradictory_predictions: bool = False
    explanation: str = ""


class AnalysisReport(BaseModel):
    project_id: str
    analysis_status: str = "COMPLETED"
    model: dict[str, str] = Field(default_factory=dict)
    summary: ObjectSummary
    images: list[ImageAnalysisResult] = []
    limitations: list[str] = []
    generated_at: str = ""
