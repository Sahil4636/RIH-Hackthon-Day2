"""
ml/scorer.py - Shelf health score calculator.

This version is tuned for a simple hackathon prototype that focuses on:
- visible product/facing counts
- low-stock gaps
- optional layout mismatches
"""
from dataclasses import dataclass

from config import settings
from ml.detector import DetectionResult
from ml.planogram import PlanogramResult


@dataclass
class ShelfScore:
    """Composite shelf health score with component breakdown."""
    overall: float
    occupancy_score: float
    compliance_score: float
    visibility_score: float
    oos_count: int
    violation_count: int
    total_detections: int
    severity: str
    summary: str


@dataclass
class AlertItem:
    alert_type: str
    severity: str
    message: str
    zone_info: dict


class ShelfScorer:
    """Computes a simple shelf score and practical alerts."""

    def score(
        self,
        shelf_id: str,
        detection: DetectionResult,
        planogram: PlanogramResult,
    ) -> tuple[ShelfScore, list[AlertItem]]:
        occupancy_score = detection.occupancy_ratio * 100
        compliance_score = planogram.compliance_score
        visibility_score = self._estimate_visibility(detection)

        overall = round(
            occupancy_score * 0.45 +
            compliance_score * 0.35 +
            visibility_score * 0.20,
            1,
        )

        oos_count = len(detection.empty_zones)
        violation_count = len(planogram.violations)

        if overall >= settings.SHELF_HEALTH_WARN:
            severity = "ok"
        elif overall >= settings.SHELF_HEALTH_CRITICAL:
            severity = "warning"
        else:
            severity = "critical"

        summary = self._build_summary(
            overall=overall,
            visible_products=len(detection.detections),
            gaps=oos_count,
            violations=violation_count,
        )

        alerts = self._generate_alerts(
            shelf_id=shelf_id,
            overall=overall,
            detection=detection,
            planogram=planogram,
        )

        return ShelfScore(
            overall=overall,
            occupancy_score=round(occupancy_score, 1),
            compliance_score=compliance_score,
            visibility_score=round(visibility_score, 1),
            oos_count=oos_count,
            violation_count=violation_count,
            total_detections=len(detection.detections),
            severity=severity,
            summary=summary,
        ), alerts

    def _estimate_visibility(self, detection: DetectionResult) -> float:
        """Use confidence as a proxy for how clearly products are visible."""
        if not detection.detections:
            return 40.0
        high_conf = sum(1 for det in detection.detections if det.confidence >= 0.65)
        return round((high_conf / len(detection.detections)) * 100, 1)

    def _build_summary(
        self,
        overall: float,
        visible_products: int,
        gaps: int,
        violations: int,
    ) -> str:
        parts = [f"{visible_products} visible product facings"]
        if gaps:
            parts.append(f"{gaps} low-stock gap{'s' if gaps != 1 else ''}")
        if violations:
            parts.append(f"{violations} layout mismatch{'es' if violations != 1 else ''}")
        return f"Shelf health {overall}/100 with " + " and ".join(parts) + "."

    def _generate_alerts(
        self,
        shelf_id: str,
        overall: float,
        detection: DetectionResult,
        planogram: PlanogramResult,
    ) -> list[AlertItem]:
        alerts: list[AlertItem] = []

        for index, zone in enumerate(detection.empty_zones, start=1):
            pct = round(zone["area_ratio"] * 100, 1)
            severity = "critical" if overall < settings.SHELF_HEALTH_CRITICAL else "warning"
            alerts.append(AlertItem(
                alert_type="gap",
                severity=severity,
                message=f"Low-stock gap on {shelf_id} in zone {index} ({pct}% of image area).",
                zone_info=zone,
            ))

        for zone in planogram.violations:
            severity = "critical" if zone.priority == "high" else "warning"
            alerts.append(AlertItem(
                alert_type="layout",
                severity=severity,
                message=(
                    f"Layout mismatch on {shelf_id} at row {zone.row}, col {zone.col}: "
                    f"expected '{zone.expected_label}' but found '{zone.found_label}'."
                ),
                zone_info={"row": zone.row, "col": zone.col, "priority": zone.priority},
            ))

        if overall < settings.SHELF_HEALTH_CRITICAL:
            alerts.append(AlertItem(
                alert_type="health",
                severity="critical",
                message=f"Shelf {shelf_id} is in critical condition with score {overall}/100.",
                zone_info={},
            ))

        return alerts


scorer = ShelfScorer()
