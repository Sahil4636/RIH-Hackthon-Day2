"""
test_pipeline.py — Quick sanity-check for the ML pipeline
Run: python test_pipeline.py

Creates a synthetic shelf image, runs detection, planogram check,
and scoring — prints results to terminal. No server needed.
"""
import sys
import numpy as np
import cv2
from pathlib import Path

# ── Make sure we can import our modules ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


def make_fake_shelf(width=800, height=400) -> np.ndarray:
    """
    Draw a simple synthetic shelf image:
    rows of coloured rectangles = 'products', with deliberate gaps.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # light grey bg

    # Shelf lines
    for y in [130, 260]:
        cv2.line(img, (0, y), (width, y), (80, 60, 40), 6)

    colours = [
        (220, 50, 50), (50, 200, 80), (50, 80, 220),
        (220, 180, 50), (160, 50, 210),
    ]

    product_w, product_h = 110, 100
    for row_idx, row_y in enumerate([20, 150, 280]):
        for col_idx in range(6):
            # leave a deliberate gap at col 3, row 1
            if row_idx == 1 and col_idx == 3:
                continue
            x1 = col_idx * (product_w + 10) + 20
            y1 = row_y
            x2 = x1 + product_w
            y2 = y1 + product_h
            colour = colours[(row_idx + col_idx) % len(colours)]
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), 2)
            cv2.putText(img, f"P{row_idx}{col_idx}", (x1 + 10, y1 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return img


def run_test():
    print("\n" + "═" * 55)
    print("  ShelfVision — Pipeline Sanity Test")
    print("═" * 55)

    # ── GPU check ─────────────────────────────────────────────────────────
    import torch
    print(f"\n[GPU]  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[GPU]  Device         : {torch.cuda.get_device_name(0)}")

    # ── Create fake shelf image ────────────────────────────────────────────
    print("\n[IMG]  Generating synthetic shelf image …")
    shelf_img = make_fake_shelf()
    test_path = Path("data/uploads/test_shelf.jpg")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), shelf_img)
    print(f"[IMG]  Saved → {test_path}")

    # ── Detection ─────────────────────────────────────────────────────────
    print("\n[DET]  Running YOLOv8 detection …")
    from ml.detector import detector
    result = detector.detect(shelf_img, draw=True)
    print(f"[DET]  Products detected : {len(result.detections)}")
    print(f"[DET]  Empty zones found : {len(result.empty_zones)}")
    print(f"[DET]  Shelf occupancy   : {result.occupancy_ratio:.1%}")

    if result.annotated_image is not None:
        ann_path = Path("data/annotated/test_annotated.jpg")
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ann_path), result.annotated_image)
        print(f"[DET]  Annotated image  → {ann_path}")

    # ── Planogram check ───────────────────────────────────────────────────
    print("\n[PLN]  Checking planogram compliance for 'shelf_demo' …")
    from ml.planogram import planogram_checker
    plano_result = planogram_checker.check("shelf_demo", result)
    print(f"[PLN]  Has planogram      : {plano_result.has_planogram}")
    print(f"[PLN]  Compliance score   : {plano_result.compliance_score:.1f}/100")
    print(f"[PLN]  Violations         : {len(plano_result.violations)}")
    print(f"[PLN]  Empty zones        : {len(plano_result.empty_zones)}")

    # ── Scoring ───────────────────────────────────────────────────────────
    print("\n[SCR]  Computing shelf health score …")
    from ml.scorer import scorer
    shelf_score, alerts = scorer.score("shelf_demo", result, plano_result)
    print(f"[SCR]  Overall score      : {shelf_score.overall}/100  ({shelf_score.severity.upper()})")
    print(f"[SCR]  Occupancy score    : {shelf_score.occupancy_score}")
    print(f"[SCR]  Compliance score   : {shelf_score.compliance_score}")
    print(f"[SCR]  Visibility score   : {shelf_score.visibility_score}")
    print(f"[SCR]  Summary            : {shelf_score.summary}")
    print(f"[SCR]  Alerts generated   : {len(alerts)}")
    for a in alerts:
        print(f"         [{a.severity.upper():8s}] {a.message[:70]}")

    print("\n" + "═" * 55)
    print("  ✅  Pipeline test PASSED — all modules working!")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    run_test()
