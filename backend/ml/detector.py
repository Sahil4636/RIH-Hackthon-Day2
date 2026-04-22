"""
ml/detector.py - Hybrid shelf detector.

Uses YOLO when it works, but falls back to image-structure heuristics for
dense retail shelves where a generic model cannot identify individual SKUs.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detected product or shelf facing."""
    box: list[float]
    label: str
    confidence: float
    class_id: int
    area_ratio: float


@dataclass
class DetectionResult:
    """Full result for one shelf image."""
    detections: list[Detection]
    empty_zones: list[dict]
    image_shape: tuple[int, int, int]
    occupancy_ratio: float
    annotated_image: Optional[np.ndarray] = None
    raw_labels: list[str] = field(default_factory=list)


class ShelfDetector:
    """
    Hybrid detector for retail shelves.

    Strategy:
    1. Try YOLO for semantic object detections.
    2. If YOLO returns too little useful coverage, detect product facings from
       edges/contours so the demo still works on crowded shelf photos.
    3. Estimate empty zones from uncovered regions inside inferred shelf rows.
    """

    _COLOURS = {
        "high": (82, 212, 132),
        "medium": (240, 192, 64),
        "fallback": (76, 144, 255),
        "empty": (255, 107, 107),
    }

    def __init__(self):
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and settings.DEVICE == "cuda":
            logger.warning("CUDA not available, falling back to CPU.")

        custom = Path(settings.MODEL_DIR) / "shelf_model.pt"
        base = Path(settings.MODEL_DIR) / settings.YOLO_MODEL

        if custom.exists():
            logger.info("Loading custom model: %s", custom)
            self.model = YOLO(str(custom))
        elif base.exists():
            logger.info("Loading base model: %s", base)
            self.model = YOLO(str(base))
        else:
            logger.info("Downloading base model: %s", settings.YOLO_MODEL)
            self.model = YOLO(settings.YOLO_MODEL)

        self.model.to(self.device)
        logger.info("Detector ready on %s", self.device)

    def detect(self, image: np.ndarray, draw: bool = True) -> DetectionResult:
        """Run hybrid detection on a BGR image."""
        h, w = image.shape[:2]
        yolo_detections = self._run_yolo(image, w, h)
        shelf_rows = self._estimate_shelf_rows(image)

        use_fallback = self._should_use_fallback(yolo_detections, shelf_rows)
        detections = self._merge_detections(
            yolo_detections,
            self._detect_facings(image, shelf_rows) if use_fallback else [],
        )

        empty_zones = self._find_empty_zones(image, detections, shelf_rows)
        occupancy = self._compute_occupancy(detections, empty_zones)
        annotated = self._draw_annotations(image.copy(), detections, empty_zones, shelf_rows, w, h) if draw else None

        return DetectionResult(
            detections=detections,
            empty_zones=empty_zones,
            image_shape=image.shape,
            occupancy_ratio=occupancy,
            annotated_image=annotated,
            raw_labels=[d.label for d in detections],
        )

    def _run_yolo(self, image: np.ndarray, w: int, h: int) -> list[Detection]:
        results = self.model.predict(
            source=image,
            imgsz=settings.INPUT_SIZE,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = str(self.model.names[cls])
            area = ((x2 - x1) * (y2 - y1)) / (w * h)

            if area < 0.002 or area > 0.35:
                continue

            detections.append(Detection(
                box=[x1 / w, y1 / h, x2 / w, y2 / h],
                label=label,
                confidence=conf,
                class_id=cls,
                area_ratio=area,
            ))

        return detections

    def _estimate_shelf_rows(self, image: np.ndarray) -> list[tuple[int, int]]:
        """
        Estimate shelf bands from horizontal edge density.

        This is intentionally simple and robust for hackathon/demo use.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy = np.mean(np.abs(grad), axis=1)
        energy = cv2.GaussianBlur(energy.reshape(-1, 1), (1, 31), 0).reshape(-1)

        threshold = float(np.percentile(energy, 58))
        mask = energy > threshold
        bands: list[tuple[int, int]] = []
        start = None

        for idx, active in enumerate(mask):
            if active and start is None:
                start = idx
            elif not active and start is not None:
                if idx - start >= max(24, h // 16):
                    bands.append((start, idx))
                start = None
        if start is not None and h - start >= max(24, h // 16):
            bands.append((start, h))

        merged: list[tuple[int, int]] = []
        for band in bands:
            if not merged or band[0] - merged[-1][1] > 18:
                merged.append(list(band))
            else:
                merged[-1][1] = band[1]

        merged_rows = [(max(0, int(y1 - 8)), min(h, int(y2 + 8))) for y1, y2 in merged]
        if 2 <= len(merged_rows) <= 8:
            return merged_rows

        default_rows = 4
        row_h = h // default_rows
        return [(i * row_h, h if i == default_rows - 1 else (i + 1) * row_h) for i in range(default_rows)]

    def _should_use_fallback(self, detections: list[Detection], shelf_rows: list[tuple[int, int]]) -> bool:
        if len(detections) < max(8, len(shelf_rows) * 3):
            return True
        occupied = sum(d.area_ratio for d in detections)
        return occupied < 0.18

    def _detect_facings(self, image: np.ndarray, shelf_rows: list[tuple[int, int]]) -> list[Detection]:
        """
        Detect likely product facings from contours inside each shelf row.

        Labels are generic because this is geometry-based fallback, not SKU
        recognition.
        """
        h, w = image.shape[:2]
        detections: list[Detection] = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for row_index, (y1, y2) in enumerate(shelf_rows):
            roi = gray[y1:y2, :]
            if roi.size == 0:
                continue

            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            edges = cv2.Canny(blur, 60, 160)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11))
            merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            row_detections: list[Detection] = []
            row_area = max(1, w * (y2 - y1))
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                area = (bw * bh) / row_area
                aspect = bw / max(bh, 1)

                if bw < max(18, w // 55):
                    continue
                if bh < max(28, (y2 - y1) // 4):
                    continue
                if area < 0.01 or area > 0.32:
                    continue
                if aspect > 2.4:
                    continue

                confidence = min(0.79, 0.42 + area * 2.4)
                row_detections.append(Detection(
                    box=[x / w, (y + y1) / h, (x + bw) / w, (y + y1 + bh) / h],
                    label=f"product_facing_r{row_index + 1}",
                    confidence=float(confidence),
                    class_id=-1,
                    area_ratio=(bw * bh) / (w * h),
                ))

            detections.extend(self._dedupe_row_detections(sorted(row_detections, key=lambda d: d.box[0])))

        logger.info("Fallback shelf-facings detector produced %s detections", len(detections))
        return detections

    def _dedupe_row_detections(self, detections: list[Detection]) -> list[Detection]:
        deduped: list[Detection] = []
        for det in detections:
            if not deduped:
                deduped.append(det)
                continue
            prev = deduped[-1]
            if self._iou(prev.box, det.box) > 0.35:
                if det.confidence > prev.confidence:
                    deduped[-1] = det
            else:
                deduped.append(det)
        return deduped

    def _merge_detections(self, primary: list[Detection], fallback: list[Detection]) -> list[Detection]:
        merged = list(primary)
        for candidate in fallback:
            if all(self._iou(candidate.box, existing.box) < 0.25 for existing in merged):
                merged.append(candidate)
        return merged

    def _find_empty_zones(
        self,
        image: np.ndarray,
        detections: list[Detection],
        shelf_rows: list[tuple[int, int]],
    ) -> list[dict]:
        """
        Estimate empty zones as large uncovered gaps inside shelf rows.

        Restricting the search to row bands avoids labeling the background above
        and below the shelf as out-of-stock space.
        """
        h, w = image.shape[:2]
        min_row_gap = 0.07
        empty_zones: list[dict] = []

        for y1, y2 in shelf_rows:
            row_height = max(1, y2 - y1)
            row_mask = np.zeros((row_height, w), dtype=np.uint8)

            row_detections = [
                det for det in detections
                if not (int(det.box[3] * h) <= y1 or int(det.box[1] * h) >= y2)
            ]

            for det in row_detections:
                x1 = max(0, int(det.box[0] * w))
                x2 = min(w, int(det.box[2] * w))
                yy1 = max(0, int(det.box[1] * h) - y1)
                yy2 = min(row_height, int(det.box[3] * h) - y1)
                if x2 > x1 and yy2 > yy1:
                    cv2.rectangle(row_mask, (x1, yy1), (x2, yy2), 255, -1)

            coverage_profile = np.mean(row_mask > 0, axis=0).astype(np.float32)
            occupied = cv2.GaussianBlur(coverage_profile.reshape(1, -1), (1, 31), 0).reshape(-1) > 0.12

            start = None
            for x, is_occupied in enumerate(occupied):
                if not is_occupied and start is None:
                    start = x
                elif is_occupied and start is not None:
                    if (x - start) / w >= min_row_gap:
                        empty_zones.append({
                            "box": [start / w, y1 / h, x / w, y2 / h],
                            "area_ratio": ((x - start) * row_height) / (w * h),
                            "pixel_area": int((x - start) * row_height),
                        })
                    start = None
            if start is not None and (w - start) / w >= min_row_gap:
                empty_zones.append({
                    "box": [start / w, y1 / h, 1.0, y2 / h],
                    "area_ratio": ((w - start) * row_height) / (w * h),
                    "pixel_area": int((w - start) * row_height),
                })

        return [zone for zone in empty_zones if zone["area_ratio"] >= 0.015]

    def _compute_occupancy(self, detections: list[Detection], empty_zones: list[dict]) -> float:
        if detections:
            detection_area = sum(d.area_ratio for d in detections)
            capped_detection_occupancy = min(0.96, detection_area * 1.05)
        else:
            capped_detection_occupancy = 0.0

        empty_penalty = min(0.95, sum(zone["area_ratio"] for zone in empty_zones))
        combined = max(capped_detection_occupancy, 1.0 - empty_penalty)
        return max(0.0, min(1.0, combined))

    def _draw_annotations(
        self,
        image: np.ndarray,
        detections: list[Detection],
        empty_zones: list[dict],
        shelf_rows: list[tuple[int, int]],
        w: int,
        h: int,
    ) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 2

        for idx, (y1, y2) in enumerate(shelf_rows):
            cv2.line(image, (0, y1), (w, y1), (203, 160, 99), 1)
            cv2.putText(image, f"ROW {idx + 1}", (8, min(h - 8, y1 + 18)), font, 0.45, (120, 92, 50), 1)
            cv2.line(image, (0, y2), (w, y2), (203, 160, 99), 1)

        for det in detections:
            x1, y1, x2, y2 = (
                int(det.box[0] * w), int(det.box[1] * h),
                int(det.box[2] * w), int(det.box[3] * h),
            )
            if det.class_id == -1:
                colour = self._COLOURS["fallback"]
            else:
                colour = self._COLOURS["high"] if det.confidence >= 0.65 else self._COLOURS["medium"]

            cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
            label = det.label if det.class_id != -1 else "product"
            label_txt = f"{label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label_txt, font, font_scale, 1)
            top = max(th + 8, y1)
            cv2.rectangle(image, (x1, top - th - 6), (x1 + tw + 4, top), colour, -1)
            cv2.putText(image, label_txt, (x1 + 2, top - 4), font, font_scale, (0, 0, 0), 1)

        for zone in empty_zones:
            x1, y1, x2, y2 = (
                int(zone["box"][0] * w), int(zone["box"][1] * h),
                int(zone["box"][2] * w), int(zone["box"][3] * h),
            )
            colour = self._COLOURS["empty"]
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
            cv2.addWeighted(overlay, 0.18, image, 0.82, 0, image)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
            cv2.putText(image, "LOW STOCK / GAP", (x1 + 4, min(h - 8, y1 + 18)), font, 0.45, colour, 1)

        return image

    def _iou(self, box_a: list[float], box_b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


detector = ShelfDetector()
