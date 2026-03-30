"""
YOLO26x Training Script — Baseline
Dataset: cameratrap_data (merged First + Second training, 4 classes)
Classes: Brent_Up, Brent_Down, Barnacle_Up, Barnacle_Down
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
DEVICE   = 0      # GPU index; set to "cpu" if no GPU


def train():
    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found: {YAML_PATH}\n"
            "Run merge_first_second_data.py first to prepare the dataset."
        )

    model = YOLO("yolo26x.pt")

    results = model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=str(BASE_DIR / "runs" / "train"),
        name="yolo26x_baseline",
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        # Training control
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
