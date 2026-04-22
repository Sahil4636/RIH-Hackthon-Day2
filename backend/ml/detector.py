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

        self.model = self._load_semantic_model()
        self.dense_model = self._load_dense_model()
        logger.info("Detector ready on %s", self.device)

    def _load_semantic_model(self) -> YOLO:
        custom = Path(settings.MODEL_DIR) / "shelf_model.pt"
        base = Path(settings.MODEL_DIR) / settings.YOLO_MODEL

        if custom.exists():
            logger.info("Loading custom model: %s", custom)
            model = YOLO(str(custom))
        elif base.exists():
            logger.info("Loading base model: %s", base)
            model = YOLO(str(base))
        else:
            logger.info("Downloading base model: %s", settings.YOLO_MODEL)
            model = YOLO(settings.YOLO_MODEL)

        model.to(self.device)
        return model

    def _load_dense_model(self) -> YOLO | None:
        dense_path = Path(settings.MODEL_DIR) / settings.SKU110K_MODEL
        if dense_path.exists():
            logger.info("Loading dense shelf model: %s", dense_path)
            model = YOLO(str(dense_path))
            model.to(self.device)
            return model

        logger.info("Dense shelf model not found at %s; tiled inference will reuse the semantic model.", dense_path)
        return None

    def detect(self, image: np.ndarray, draw: bool = True) -> DetectionResult:
        """Run hybrid detection on a BGR image."""
        h, w = image.shape[:2]
        global_detections = self._run_yolo(self.model, image, w, h)
        dense_detections = self._run_dense_tiled_inference(image, w, h)
        yolo_detections = self._dedupe_row_detections(
            sorted(global_detections + dense_detections, key=lambda det: (det.box[1], det.box[0]))
        )
        shelf_rows = self._estimate_shelf_rows(image)
        price_bands = self._estimate_price_bands(image, shelf_rows)

        use_fallback = self._should_use_fallback(yolo_detections, shelf_rows)
        detections = self._merge_detections(
            yolo_detections,
            self._detect_facings(image, shelf_rows) if use_fallback else [],
        )
        detections = self._filter_label_like_detections(detections, price_bands, shelf_rows, w, h)

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

    def _estimate_price_bands(
        self,
        image: np.ndarray,
        shelf_rows: list[tuple[int, int]],
    ) -> list[tuple[float, float, float, float]]:
        """
        Estimate shelf-edge price label bands.

        Price tags usually appear as bright, low-saturation horizontal strips near
        the lower part of a shelf row. We detect those strips and later suppress
        detections that mostly overlap them.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bands: list[tuple[float, float, float, float]] = []

        for y1, y2 in shelf_rows:
            roi = hsv[y1:y2, :]
            if roi.size == 0:
                continue

            sat = roi[:, :, 1]
            val = roi[:, :, 2]
            bright_low_sat = ((val > 155) & (sat < 95)).astype(np.uint8)
            profile = np.mean(bright_low_sat, axis=1)
            profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 11), 0).reshape(-1)

            threshold = max(0.34, float(np.percentile(profile, 75)))
            active_rows = np.where(profile >= threshold)[0]
            if active_rows.size == 0:
                continue

            lower_half = active_rows[active_rows >= int((y2 - y1) * 0.45)]
            if lower_half.size == 0:
                continue

            band_top = int(np.min(lower_half))
            band_bottom = int(np.max(lower_half))
            band_height = band_bottom - band_top + 1

            if band_height < max(8, (y2 - y1) // 12):
                continue

            bands.append((0.0, (y1 + band_top) / h, 1.0, (y1 + band_bottom + 1) / h))

        return bands

    def _filter_label_like_detections(
        self,
        detections: list[Detection],
        price_bands: list[tuple[float, float, float, float]],
        shelf_rows: list[tuple[int, int]],
        w: int,
        h: int,
    ) -> list[Detection]:
        filtered: list[Detection] = []

        for det in detections:
            box_h = (det.box[3] - det.box[1]) * h
            box_w = (det.box[2] - det.box[0]) * w
            aspect = box_w / max(box_h, 1)
            overlap_with_band = max((self._intersection_ratio(det.box, band) for band in price_bands), default=0.0)

            row_height = None
            for y1, y2 in shelf_rows:
                cy = ((det.box[1] + det.box[3]) / 2) * h
                if y1 <= cy <= y2:
                    row_height = max(1, y2 - y1)
                    break

            looks_like_label = False
            if row_height is not None:
                height_ratio = box_h / row_height
                near_bottom = False
                for y1, y2 in shelf_rows:
                    cy = ((det.box[1] + det.box[3]) / 2) * h
                    if y1 <= cy <= y2:
                        near_bottom = cy >= y1 + 0.62 * (y2 - y1)
                        break

                looks_like_label = (
                    overlap_with_band > 0.55 or
                    (near_bottom and height_ratio < 0.32 and aspect > 0.9)
                )

            if not looks_like_label:
                filtered.append(det)

        return filtered

    def _run_yolo(self, model: YOLO, image: np.ndarray, w: int, h: int) -> list[Detection]:
        results = model.predict(
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
            label = str(model.names[cls])
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

    def _run_dense_tiled_inference(self, image: np.ndarray, w: int, h: int) -> list[Detection]:
        if not settings.ENABLE_DENSE_TILING:
            return []

        model = self.dense_model or self.model
        tile_size = max(settings.INPUT_SIZE, settings.DENSE_TILE_SIZE)
        overlap = min(0.45, max(0.05, settings.DENSE_TILE_OVERLAP))
        stride = max(64, int(tile_size * (1.0 - overlap)))
        dense_detections: list[Detection] = []

        for top in range(0, max(1, h), stride):
            for left in range(0, max(1, w), stride):
                bottom = min(h, top + tile_size)
                right = min(w, left + tile_size)
                tile = image[top:bottom, left:right]
                if tile.size == 0:
                    continue

                tile_h, tile_w = tile.shape[:2]
                results = model.predict(
                    source=tile,
                    imgsz=max(settings.INPUT_SIZE, min(tile_size, 1280)),
                    conf=settings.DENSE_CONFIDENCE_THRESHOLD,
                    iou=settings.DENSE_IOU_THRESHOLD,
                    device=self.device,
                    verbose=False,
                )[0]

                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    gx1 = (x1 + left) / w
                    gy1 = (y1 + top) / h
                    gx2 = (x2 + left) / w
                    gy2 = (y2 + top) / h
                    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)

                    if area_ratio < 0.0006 or area_ratio > 0.10:
                        continue

                    label = "product" if self.dense_model is not None else str(model.names[cls])
                    dense_detections.append(Detection(
                        box=[gx1, gy1, gx2, gy2],
                        label=label,
                        confidence=conf,
                        class_id=-20 if self.dense_model is not None else cls,
                        area_ratio=area_ratio,
                    ))

                if right == w:
                    break
            if bottom == h:
                break

        deduped = self._nms_detections(dense_detections, iou_threshold=settings.DENSE_IOU_THRESHOLD)
        logger.info("Dense tiled inference produced %s detections", len(deduped))
        return deduped

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

            row_detections.extend(self._detect_facings_from_profile(image, row_index, y1, y2))
            detections.extend(self._dedupe_row_detections(sorted(row_detections, key=lambda d: d.box[0])))

        logger.info("Fallback shelf-facings detector produced %s detections", len(detections))
        return detections

    def _detect_facings_from_profile(
        self,
        image: np.ndarray,
        row_index: int,
        y1: int,
        y2: int,
    ) -> list[Detection]:
        """
        Detect facings from per-column visual activity.

        Dense shelves often fail contour detection because adjacent products touch
        each other. This profile-based fallback looks for vertical bands with
        texture/color variation and converts them into generic product facings.
        """
        h, w = image.shape[:2]
        roi = image[y1:y2, :]
        if roi.size == 0:
            return []

        row_height = max(1, y2 - y1)
        upper = int(row_height * 0.10)
        lower = int(row_height * 0.88)
        focus = roi[upper:lower, :]
        if focus.size == 0:
            focus = roi

        gray = cv2.cvtColor(focus, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        texture = np.mean(np.abs(grad_x), axis=0)

        hsv = cv2.cvtColor(focus, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1], axis=0)
        value = np.mean(hsv[:, :, 2], axis=0)

        activity = 0.50 * texture + 0.30 * saturation + 0.20 * value
        activity = cv2.GaussianBlur(activity.reshape(1, -1), (1, 15), 0).reshape(-1)

        threshold = float(np.percentile(activity, 40))
        active = activity >= threshold

        min_width = max(10, w // 90)
        target_width = max(18, w // 18)
        max_width = max(target_width * 3, w // 7)
        segments: list[tuple[int, int]] = []
        start = None

        for x, is_active in enumerate(active):
            if is_active and start is None:
                start = x
            elif not is_active and start is not None:
                if min_width <= (x - start) <= max_width:
                    segments.append((start, x))
                start = None
        if start is not None and min_width <= (w - start) <= max_width:
            segments.append((start, w))

        merged: list[list[int]] = []
        for left, right in segments:
            if not merged:
                merged.append([left, right])
                continue
            gap = left - merged[-1][1]
            if gap <= max(4, w // 220):
                merged[-1][1] = right
            else:
                merged.append([left, right])

        detections: list[Detection] = []
        for left, right in merged:
            detections.extend(
                self._split_profile_segment_into_facings(
                    left=left,
                    right=right,
                    activity=activity,
                    row_index=row_index,
                    y1=y1,
                    y2=y2,
                    image_width=w,
                    image_height=h,
                    min_width=min_width,
                    target_width=target_width,
                    max_width=max_width,
                )
            )

        return detections

    def _split_profile_segment_into_facings(
        self,
        left: int,
        right: int,
        activity: np.ndarray,
        row_index: int,
        y1: int,
        y2: int,
        image_width: int,
        image_height: int,
        min_width: int,
        target_width: int,
        max_width: int,
    ) -> list[Detection]:
        width = right - left
        if width < min_width:
            return []

        boundaries = [left]
        estimated_facings = max(1, round(width / max(target_width, 1)))
        if width > max_width:
            estimated_facings = max(estimated_facings, 2)

        if estimated_facings > 1:
            local = activity[left:right]
            search_radius = max(4, target_width // 3)
            for split_idx in range(1, estimated_facings):
                target = left + int(width * split_idx / estimated_facings)
                window_start = max(left + min_width, target - search_radius)
                window_end = min(right - min_width, target + search_radius)
                if window_end <= window_start:
                    continue
                local_window = activity[window_start:window_end]
                valley = int(np.argmin(local_window)) + window_start
                if valley - boundaries[-1] >= min_width and right - valley >= min_width:
                    boundaries.append(valley)

        boundaries.append(right)
        boundaries = sorted(set(boundaries))

        detections: list[Detection] = []
        for start, end in zip(boundaries, boundaries[1:]):
            facing_width = end - start
            if facing_width < min_width:
                continue

            confidence = float(
                min(
                    0.82,
                    0.38 + (facing_width / max(target_width, 1)) * 0.12
                )
            )
            detections.append(Detection(
                box=[start / image_width, y1 / image_height, end / image_width, y2 / image_height],
                label=f"product_facing_r{row_index + 1}",
                confidence=confidence,
                class_id=-2,
                area_ratio=(facing_width * (y2 - y1)) / (image_width * image_height),
            ))

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
            if all(self._iou(candidate.box, existing.box) < 0.45 for existing in merged):
                merged.append(candidate)
        return merged

    def _nms_detections(self, detections: list[Detection], iou_threshold: float) -> list[Detection]:
        kept: list[Detection] = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if all(self._iou(det.box, other.box) < iou_threshold for other in kept):
                kept.append(det)
        return kept

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
            occupied = cv2.GaussianBlur(coverage_profile.reshape(1, -1), (1, 31), 0).reshape(-1) > 0.06

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
            if det.class_id in (-1, -2):
                colour = self._COLOURS["fallback"]
            else:
                colour = self._COLOURS["high"] if det.confidence >= 0.65 else self._COLOURS["medium"]

            cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
            label = det.label if det.class_id not in (-1, -2) else "product"
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

    def _intersection_ratio(self, box_a: list[float], box_b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        return inter / area_a if area_a > 0 else 0.0


detector = ShelfDetector()
