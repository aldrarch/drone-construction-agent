"""
Report generation service: builds the JSON report from analysis results.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from schemas.prediction import (
    AnalysisReport,
    ImageAnalysisResult,
    ObjectSummary,
)


LIMITATIONS = [
    "CLIP is a general-purpose model and not a specialized construction assessment model.",
    "Similarity scores are not calibrated probabilities.",
    "Results require human verification and should not be treated as a final decision.",
]


def build_report(
    project_id: str,
    model_type: str,
    model_name: str,
    summary: ObjectSummary,
    images: list[ImageAnalysisResult],
    status: str = "COMPLETED",
) -> AnalysisReport:
    """Create an AnalysisReport from the given results."""
    return AnalysisReport(
        project_id=project_id,
        analysis_status=status,
        model={"type": model_type, "name": model_name},
        summary=summary,
        images=images,
        limitations=LIMITATIONS,
        generated_at=datetime.now(UTC).isoformat(),
    )


def report_to_json(report: AnalysisReport, indent: int = 2) -> str:
    """Serialize an AnalysisReport to a JSON string."""
    return json.dumps(report.model_dump(exclude_none=False), indent=indent, ensure_ascii=False)


def save_report(report: AnalysisReport, directory: str | Path = "reports") -> Path:
    """Save the report as a JSON file in the given directory."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    filename = f"{report.project_id}_report_{report.generated_at[:10]}.json"
    path = target / filename
    path.write_text(report_to_json(report), encoding="utf-8")
    return path
