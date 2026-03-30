"""
YOLO26n Test Script — Baseline
Evaluates the model trained with train_yolo26n_baseline.py.
Usage:
  python test_yolo26n_baseline.py eval
  python test_yolo26n_baseline.py visualize
  python test_yolo26n_baseline.py predict <source>
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
YAML_PATH  = BASE_DIR / "cameratrap_data" / "dataset.yaml"
SPLIT_DIR  = BASE_DIR / "cameratrap_data"

DEFAULT_WEIGHTS = BASE_DIR / "runs" / "train" / "yolo26n_baseline" / "weights" / "best.pt"

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE   = 1280
CONF_THRES = 0.25   # confidence threshold for predictions
IOU_THRES  = 0.45   # NMS IoU threshold
DEVICE     = 0      # GPU index; set to "cpu" if no GPU


def evaluate(weights: Path):
    """Run val on the held-out test split and print metrics."""
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Run train_yolo26n_baseline.py first, or pass --weights <path/to/best.pt>"
        )
    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found: {YAML_PATH}\n"
            "Run merge_first_second_data.py first to prepare the dataset."
        )

    print(f"[INFO] Loading model: {weights}")
    model = YOLO(str(weights))

    print("[INFO] Evaluating on test split...")
    metrics = model.val(
        data=str(YAML_PATH),
        split="test",
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        project=str(BASE_DIR / "runs" / "test"),
        name="yolo26n_baseline",
        save_json=True,
        plots=True,
        verbose=True,
    )

    print("\n─── Test Results (yolo26n_baseline) ────────────────────")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print("────────────────────────────────────────────────────────")

    names = ["Brent_Up", "Brent_Down", "Barnacle_Up", "Barnacle_Down"]
    print("\nPer-class AP50:")
    for cls_name, ap in zip(names, metrics.box.ap50):
        print(f"  {cls_name:<15}: {ap:.4f}")

    print(f"\n[INFO] Plots saved to: {metrics.save_dir}")
    return metrics


def predict(weights: Path, source: str, save_dir: Path | None = None):
    """Run inference on a folder / single image and save visualisations."""
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    out_dir = save_dir or (BASE_DIR / "runs" / "predict" / "yolo26n_baseline")

    print(f"[INFO] Running inference on: {source}")
    model = YOLO(str(weights))

    results = model.predict(
        source=source,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        verbose=True,
    )

    print(f"\n[DONE] Predictions saved to: {out_dir}")
    return results


def visualize_test(weights: Path, save_dir: Path | None = None):
    """Run inference on the test split and save images with prediction boxes."""
    test_images = SPLIT_DIR / "images" / "test"
    if not test_images.exists():
        raise FileNotFoundError(
            f"Test split not found: {test_images}\n"
            "Run merge_first_second_data.py first to prepare the dataset."
        )
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    out_dir = save_dir or (BASE_DIR / "runs" / "visualize" / "yolo26n_baseline")
    predict(weights, str(test_images), out_dir)


def main():
    parser = argparse.ArgumentParser(description="YOLO26n Camera Trap — Test & Inference")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    eval_p = subparsers.add_parser("eval", help="Evaluate on test split (metrics)")
    eval_p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Path to model weights (.pt)")

    pred_p = subparsers.add_parser("predict", help="Run inference on images/folder")
    pred_p.add_argument("source", type=str,
                        help="Image path, folder, or glob")
    pred_p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Path to model weights (.pt)")
    pred_p.add_argument("--save-dir", type=Path, default=None,
                        help="Output directory for predictions")

    vis_p = subparsers.add_parser("visualize", help="Save test split images with prediction boxes")
    vis_p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                       help="Path to model weights (.pt)")
    vis_p.add_argument("--save-dir", type=Path, default=None,
                       help="Output directory (default: runs/visualize/yolo26n_baseline)")

    args = parser.parse_args()

    if args.mode == "eval":
        evaluate(args.weights)
    elif args.mode == "predict":
        predict(args.weights, args.source, args.save_dir)
    elif args.mode == "visualize":
        visualize_test(args.weights, args.save_dir)


if __name__ == "__main__":
    main()
