"""
YOLO26m Training Script — Augmented (class balance)
Dataset: cameratrap_data (merged First + Second training, 4 classes)
Changes vs baseline:
  - copy_paste=0.3  : oversamples minority classes
  - mixup=0.1       : smooths class boundaries (Up vs Down confusion)
  - cls=1.0         : doubles classification loss weight
  - degrees=5.0     : small rotation for cross-camera robustness
  - scale=0.7       : wider scale range for small target variety
  - close_mosaic=50 : longer real-distribution convergence phase
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
YAML_PATH = BASE_DIR / "cameratrap_data" / "dataset.yaml"

# ─── Config ───────────────────────────────────────────────────────────────────
EPOCHS   = 300
BATCH    = 8
IMG_SIZE = 1280
DEVICE   = 0


def train():
    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found: {YAML_PATH}\n"
            "Run merge_first_second_data.py first to prepare the dataset."
        )

    model = YOLO("yolo26m.pt")

    results = model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=str(BASE_DIR / "runs" / "train"),
        name="yolo26m_augmented",
        # ── Standard augmentation ──
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        # ── Class balance augmentation ──
        copy_paste=0.3,
        mixup=0.1,
        # ── Higher classification loss weight ──
        cls=1.0,
        # ── Small object & cross-camera robustness ──
        degrees=5.0,
        scale=0.7,
        close_mosaic=50,
        # ── Training control ──
        patience=20,
        save=True,
        save_period=20,
        verbose=True,
    )

    print(f"\n[DONE] Training complete.")
    print(f"  Best weights: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    train()
