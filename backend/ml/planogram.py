"""
ml/planogram.py — Planogram compliance checker

A planogram is a JSON file that describes the expected product layout
for a shelf section as a grid of zones.

Schema  (data/planograms/<shelf_id>.json)
──────────────────────────────────────────
{
  "shelf_id": "shelf_A1",
  "grid_cols": 5,
  "grid_rows": 3,
  "zones": [
    {
      "row": 0, "col": 0,
      "expected_sku": "SKU-001",
      "expected_label": "Pepsi 500ml",
      "priority": "high"
    },
    ...
  ]
}
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings
from ml.detector import DetectionResult

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class ZoneResult:
    row: int
    col: int
    expected_sku: str
    expected_label: str
    status: str           # compliant | violation | empty | unknown
    found_label: str
    confidence: float
    priority: str


@dataclass
class PlanogramResult:
    shelf_id: str
    zone_results: list[ZoneResult]
    compliance_score: float    # 0-100
    violations: list[ZoneResult]
    empty_zones: list[ZoneResult]
    compliant_zones: list[ZoneResult]
    has_planogram: bool


# ── Checker ────────────────────────────────────────────────────────────────

class PlanogramChecker:
    """
    Compares a DetectionResult against the stored planogram for a shelf.

    Approach
    ────────
    1. Divide the image into a G×G grid (matching the planogram).
    2. For each grid cell, find which detected product overlaps it most.
    3. Compare detected product label vs expected label.
    4. Score compliance per zone.
    """

    def __init__(self):
        self.planogram_dir = Path(settings.DATA_DIR) / "planograms"

    # ── Public API ─────────────────────────────────────────────────────────

    def check(self, shelf_id: str, detection: DetectionResult) -> PlanogramResult:
        """
        Run compliance check.

        If no planogram exists for the shelf, returns a result with
        has_planogram=False and a neutral compliance score.
        """
        planogram = self._load_planogram(shelf_id)

        if planogram is None:
            logger.warning(f"No planogram found for shelf '{shelf_id}' — skipping compliance check.")
            return PlanogramResult(
                shelf_id=shelf_id,
                zone_results=[],
                compliance_score=100.0,   # neutral — no reference to compare
                violations=[],
                empty_zones=[],
                compliant_zones=[],
                has_planogram=False,
            )

        zone_results = self._evaluate_zones(planogram, detection)
        return self._build_result(shelf_id, zone_results)

    def load_or_create_default(self, shelf_id: str) -> dict:
        """Return existing planogram or create a blank template."""
        planogram = self._load_planogram(shelf_id)
        if planogram:
            return planogram
        return self._create_default_planogram(shelf_id)

    def save_planogram(self, shelf_id: str, data: dict) -> Path:
        path = self.planogram_dir / f"{shelf_id}.json"
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Planogram saved: {path}")
        return path

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_planogram(self, shelf_id: str) -> Optional[dict]:
        path = self.planogram_dir / f"{shelf_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _evaluate_zones(self, planogram: dict, detection: DetectionResult) -> list[ZoneResult]:
        """Map detections onto the planogram grid and score each zone."""
        cols = planogram.get("grid_cols", 5)
        rows = planogram.get("grid_rows", 3)
        zones_cfg = planogram.get("zones", [])

        # Build a grid: each cell gets the best-matching detection label
        cell_w = 1.0 / cols
        cell_h = 1.0 / rows

        # Pre-compute cell centres from detections
        det_centres = []
        for det in detection.detections:
            cx = (det.box[0] + det.box[2]) / 2
            cy = (det.box[1] + det.box[3]) / 2
            det_centres.append((cx, cy, det.label, det.confidence))

        results = []
        for zone in zones_cfg:
            r, c = zone["row"], zone["col"]
            expected_sku   = zone.get("expected_sku", "")
            expected_label = zone.get("expected_label", "").lower()
            priority       = zone.get("priority", "normal")

            # Cell bounding box (normalised)
            cell_x1 = c * cell_w
            cell_y1 = r * cell_h
            cell_x2 = cell_x1 + cell_w
            cell_y2 = cell_y1 + cell_h

            # Find detection centres inside this cell
            inside = [
                (lbl, conf)
                for cx, cy, lbl, conf in det_centres
                if cell_x1 <= cx <= cell_x2 and cell_y1 <= cy <= cell_y2
            ]

            if not inside:
                # Check if it's an expected-empty zone
                if expected_sku == "" or expected_sku == "EMPTY":
                    status       = "compliant"
                    found_label  = ""
                    conf         = 1.0
                else:
                    status       = "empty"
                    found_label  = ""
                    conf         = 0.0
            else:
                # Pick highest-confidence detection in the cell
                best_label, best_conf = max(inside, key=lambda x: x[1])
                found_label = best_label
                conf        = best_conf

                if expected_sku == "" or expected_sku == "ANY":
                    status = "compliant"
                elif best_label.lower() == expected_label:
                    status = "compliant"
                else:
                    status = "violation"

            results.append(ZoneResult(
                row=r, col=c,
                expected_sku=expected_sku,
                expected_label=expected_label,
                status=status,
                found_label=found_label,
                confidence=conf,
                priority=priority,
            ))

        return results

    def _build_result(self, shelf_id: str, zone_results: list[ZoneResult]) -> PlanogramResult:
        violations     = [z for z in zone_results if z.status == "violation"]
        empty_zones    = [z for z in zone_results if z.status == "empty"]
        compliant      = [z for z in zone_results if z.status == "compliant"]

        total = len(zone_results) or 1
        # Weight high-priority violations more heavily
        penalty = sum(2 if z.priority == "high" else 1 for z in violations + empty_zones)
        max_penalty = sum(2 if z.priority == "high" else 1 for z in zone_results)
        compliance_score = max(0.0, 100.0 * (1 - penalty / (max_penalty or 1)))

        return PlanogramResult(
            shelf_id=shelf_id,
            zone_results=zone_results,
            compliance_score=round(compliance_score, 1),
            violations=violations,
            empty_zones=empty_zones,
            compliant_zones=compliant,
            has_planogram=True,
        )

    def _create_default_planogram(self, shelf_id: str) -> dict:
        """Generate a blank 3×5 planogram template."""
        zones = [
            {"row": r, "col": c, "expected_sku": "", "expected_label": "", "priority": "normal"}
            for r in range(3)
            for c in range(5)
        ]
        return {
            "shelf_id": shelf_id,
            "grid_cols": 5,
            "grid_rows": 3,
            "zones": zones,
        }


# Singleton
planogram_checker = PlanogramChecker()
