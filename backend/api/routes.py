"""
api/routes.py — All FastAPI route handlers
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import aiofiles
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db, ScanResult, Alert, Product
from ml.detector import detector
from ml.planogram import planogram_checker
from ml.scorer import scorer
from api.schemas import (
    AnalysisResponse, AlertOut, ProductCreate, ProductOut,
    PlanogramIn, PlanogramOut, ScanHistoryItem, ShelfStats,
    DetectionBox, EmptyZone, ZoneComplianceResult, ShelfScoreOut,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── POST /api/analyze ──────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse, summary="Analyse a shelf image")
async def analyze_shelf(
    file: UploadFile = File(..., description="Shelf image (JPEG/PNG)"),
    shelf_id: str = Form(default="shelf_default", description="Unique shelf identifier"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a shelf image and get:
    - Detected products + empty zones (with annotated image)
    - Planogram compliance results
    - Shelf health score (0-100)
    - Generated alerts
    """
    # ── 1. Save uploaded image ─────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "Only JPEG, PNG, and WebP images are supported.")

    upload_dir = Path(settings.UPLOAD_DIR)
    fname      = f"{uuid.uuid4().hex}_{file.filename}"
    image_path = upload_dir / fname

    async with aiofiles.open(image_path, "wb") as f:
        await f.write(await file.read())

    # ── 2. Load image ──────────────────────────────────────────────────────
    img_array = cv2.imread(str(image_path))
    if img_array is None:
        raise HTTPException(422, "Could not decode image. Please upload a valid image file.")

    # ── 3. Run detection ───────────────────────────────────────────────────
    detection_result = detector.detect(img_array, draw=True)

    # ── 4. Save annotated image ────────────────────────────────────────────
    annotated_path = None
    annotated_url  = None
    if detection_result.annotated_image is not None:
        ann_dir  = Path(settings.DATA_DIR) / "annotated"
        ann_name = f"annotated_{fname}"
        ann_path = ann_dir / ann_name
        cv2.imwrite(str(ann_path), detection_result.annotated_image)
        annotated_path = str(ann_path)
        annotated_url  = f"/api/images/{ann_name}"

    # ── 5. Planogram check ─────────────────────────────────────────────────
    planogram_result = planogram_checker.check(shelf_id, detection_result)

    # ── 6. Score & alerts ──────────────────────────────────────────────────
    shelf_score, alerts = scorer.score(shelf_id, detection_result, planogram_result)

    # ── 7. Persist scan result ─────────────────────────────────────────────
    meta = {
        "detections": len(detection_result.detections),
        "occupancy":  round(detection_result.occupancy_ratio, 3),
        "labels":     detection_result.raw_labels[:20],  # cap for storage
    }
    scan_row = ScanResult(
        shelf_id       = shelf_id,
        image_path     = str(image_path),
        annotated_path = annotated_path,
        health_score   = shelf_score.overall,
        oos_count      = shelf_score.oos_count,
        violation_count= shelf_score.violation_count,
        total_zones    = len(planogram_result.zone_results),
        scan_meta      = json.dumps(meta),
    )
    db.add(scan_row)
    await db.flush()  # get the id

    # ── 8. Persist alerts ──────────────────────────────────────────────────
    for a in alerts:
        db.add(Alert(
            shelf_id   = shelf_id,
            scan_id    = scan_row.id,
            severity   = a.severity,
            alert_type = a.alert_type,
            message    = a.message,
        ))

    await db.commit()
    await db.refresh(scan_row)

    # ── 9. Build response ──────────────────────────────────────────────────
    return AnalysisResponse(
        scan_id   = scan_row.id,
        shelf_id  = shelf_id,
        health_score = ShelfScoreOut(
            overall          = shelf_score.overall,
            occupancy_score  = shelf_score.occupancy_score,
            compliance_score = shelf_score.compliance_score,
            visibility_score = shelf_score.visibility_score,
            severity         = shelf_score.severity,
            summary          = shelf_score.summary,
        ),
        detections = [
            DetectionBox(
                box=d.box, label=d.label,
                confidence=d.confidence, area_ratio=d.area_ratio
            ) for d in detection_result.detections
        ],
        empty_zones = [
            EmptyZone(
                box=z["box"],
                area_ratio=z["area_ratio"],
                pixel_area=z["pixel_area"],
            ) for z in detection_result.empty_zones
        ],
        oos_count           = shelf_score.oos_count,
        violation_count     = shelf_score.violation_count,
        total_detections    = shelf_score.total_detections,
        planogram_available = planogram_result.has_planogram,
        zone_results = [
            ZoneComplianceResult(
                row=z.row, col=z.col,
                expected_sku=z.expected_sku, expected_label=z.expected_label,
                status=z.status, found_label=z.found_label,
                confidence=z.confidence, priority=z.priority,
            ) for z in planogram_result.zone_results
        ],
        annotated_image_url = annotated_url,
        scanned_at          = scan_row.created_at,
    )


# ── GET /api/images/{filename} ─────────────────────────────────────────────

@router.get("/images/{filename}", summary="Serve annotated shelf image")
async def get_image(filename: str):
    path = Path(settings.DATA_DIR) / "annotated" / filename
    if not path.exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(str(path), media_type="image/jpeg")


# ── GET /api/alerts ────────────────────────────────────────────────────────

@router.get("/alerts", response_model=list[AlertOut], summary="Get all active alerts")
async def get_alerts(
    shelf_id: str | None = None,
    resolved: bool = False,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    q = select(Alert).where(Alert.resolved == resolved)
    if shelf_id:
        q = q.where(Alert.shelf_id == shelf_id)
    q = q.order_by(Alert.created_at.desc()).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


# ── DELETE /api/alerts/{id} ────────────────────────────────────────────────

@router.delete("/alerts/{alert_id}", summary="Resolve / dismiss an alert")
async def resolve_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    alert = await db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found.")
    alert.resolved    = True
    alert.resolved_at = datetime.utcnow()
    await db.commit()
    return {"message": f"Alert {alert_id} resolved."}


# ── GET /api/history/{shelf_id} ────────────────────────────────────────────

@router.get("/history/{shelf_id}", response_model=list[ScanHistoryItem])
async def get_history(shelf_id: str, limit: int = 30, db: AsyncSession = Depends(get_db)):
    q = (
        select(ScanResult)
        .where(ScanResult.shelf_id == shelf_id)
        .order_by(ScanResult.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(q)
    return result.scalars().all()


# ── GET /api/stats/{shelf_id} ──────────────────────────────────────────────

@router.get("/stats/{shelf_id}", response_model=ShelfStats)
async def get_stats(shelf_id: str, db: AsyncSession = Depends(get_db)):
    q = select(
        func.count(ScanResult.id).label("total"),
        func.avg(ScanResult.health_score).label("avg"),
        func.min(ScanResult.health_score).label("min"),
        func.max(ScanResult.health_score).label("max"),
        func.sum(ScanResult.oos_count).label("oos"),
        func.sum(ScanResult.violation_count).label("viol"),
    ).where(ScanResult.shelf_id == shelf_id)

    res = (await db.execute(q)).one()
    if not res.total:
        raise HTTPException(404, f"No scan history found for shelf '{shelf_id}'.")

    return ShelfStats(
        shelf_id=shelf_id,
        total_scans=res.total,
        avg_health=round(res.avg or 0, 1),
        min_health=round(res.min or 0, 1),
        max_health=round(res.max or 0, 1),
        total_oos_events=res.oos or 0,
        total_violations=res.viol or 0,
    )


# ── POST /api/planogram ────────────────────────────────────────────────────

@router.post("/planogram", response_model=PlanogramOut, summary="Upload planogram for a shelf")
async def upload_planogram(payload: PlanogramIn):
    planogram_checker.save_planogram(payload.shelf_id, payload.model_dump())
    return payload


# ── GET /api/planogram/{shelf_id} ─────────────────────────────────────────

@router.get("/planogram/{shelf_id}", response_model=PlanogramOut)
async def get_planogram(shelf_id: str):
    p = planogram_checker.load_or_create_default(shelf_id)
    return p


# ── GET /api/products ──────────────────────────────────────────────────────

@router.get("/products", response_model=list[ProductOut])
async def list_products(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Product).order_by(Product.name))
    return result.scalars().all()


# ── POST /api/products ─────────────────────────────────────────────────────

@router.post("/products", response_model=ProductOut, status_code=201)
async def create_product(payload: ProductCreate, db: AsyncSession = Depends(get_db)):
    product = Product(**payload.model_dump())
    db.add(product)
    await db.commit()
    await db.refresh(product)
    return product


# ── GET /api/health ────────────────────────────────────────────────────────

@router.get("/health", summary="Service health check")
async def health_check():
    import torch
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": True,
    }
