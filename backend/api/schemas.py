"""
api/schemas.py — Pydantic v2 request & response models
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Detection schemas ──────────────────────────────────────────────────────

class DetectionBox(BaseModel):
    box: list[float]       # [x1, y1, x2, y2] normalised 0-1
    label: str
    confidence: float
    area_ratio: float


class EmptyZone(BaseModel):
    box: list[float]
    area_ratio: float
    pixel_area: int


class ZoneComplianceResult(BaseModel):
    row: int
    col: int
    expected_sku: str
    expected_label: str
    status: str            # compliant | violation | empty | unknown
    found_label: str
    confidence: float
    priority: str


# ── Shelf analysis response ────────────────────────────────────────────────

class ShelfScoreOut(BaseModel):
    overall: float
    occupancy_score: float
    compliance_score: float
    visibility_score: float
    severity: str
    summary: str


class AnalysisResponse(BaseModel):
    scan_id: int
    shelf_id: str
    health_score: ShelfScoreOut
    detections: list[DetectionBox]
    empty_zones: list[EmptyZone]
    oos_count: int
    violation_count: int
    total_detections: int
    planogram_available: bool
    zone_results: list[ZoneComplianceResult]
    annotated_image_url: Optional[str] = None
    scanned_at: datetime


# ── Alert schemas ──────────────────────────────────────────────────────────

class AlertOut(BaseModel):
    id: int
    shelf_id: str
    scan_id: Optional[int]
    severity: str
    alert_type: str
    message: str
    resolved: bool
    created_at: datetime
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True


# ── Product schemas ────────────────────────────────────────────────────────

class ProductCreate(BaseModel):
    sku: str
    name: str
    category: Optional[str] = None


class ProductOut(ProductCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ── Planogram schemas ──────────────────────────────────────────────────────

class PlanogramZone(BaseModel):
    row: int
    col: int
    expected_sku: str = ""
    expected_label: str = ""
    priority: str = "normal"


class PlanogramIn(BaseModel):
    shelf_id: str
    grid_cols: int = Field(default=5, ge=1, le=20)
    grid_rows: int = Field(default=3, ge=1, le=10)
    zones: list[PlanogramZone]


class PlanogramOut(PlanogramIn):
    pass


# ── History / stats ────────────────────────────────────────────────────────

class ScanHistoryItem(BaseModel):
    id: int
    shelf_id: str
    health_score: float
    oos_count: int
    violation_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class ShelfStats(BaseModel):
    shelf_id: str
    total_scans: int
    avg_health: float
    min_health: float
    max_health: float
    total_oos_events: int
    total_violations: int
