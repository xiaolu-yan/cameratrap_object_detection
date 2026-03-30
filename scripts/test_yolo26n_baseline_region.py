"""
YOLO26n Region-Filtered Evaluation — Baseline
Evaluates only boxes whose center falls below a configurable top-crop threshold.

Usage:
  python test_yolo26n_baseline_region.py
  python test_yolo26n_baseline_region.py --top-crop 0.2
  python test_yolo26n_baseline_region.py --weights path/to/best.pt

--top-crop  : fraction of image height to ignore from the top (default 0.15)
              e.g. 0.15 means only evaluate boxes with y_center >= 0.15 * img_height

Outputs (all saved to --save-dir or runs/test/yolo26n_baseline_region/):
  results.json              — metrics summary
  confusion_matrix.png      — raw counts heatmap
  confusion_matrix_norm.png — row-normalised heatmap
  pr_curves.png             — per-class precision-recall curves
  f1_curves.png             — per-class F1 vs confidence curves
  vis/                      — top-30 worst images with prediction boxes + crop line
"""

import sys
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
SPLIT_DIR       = BASE_DIR / "cameratrap_data"
DEFAULT_WEIGHTS = BASE_DIR / "runs" / "train" / "yolo26n_baseline" / "weights" / "best.pt"

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE    = 1280
CONF_THRES  = 0.15
IOU_THRES   = 0.45
DEVICE      = 0
CLASS_NAMES = ["Brent_Up", "Brent_Down", "Barnacle_Up", "Barnacle_Down"]

# One distinct BGR colour per class (for OpenCV drawing)
CLASS_COLORS_BGR = [
    (0,   200, 255),   # Brent_Up     — amber
    (0,   100, 255),   # Brent_Down   — orange-red
    (50,  220,  50),   # Barnacle_Up  — green
    (255, 100,  50),   # Barnacle_Down— blue
]


# ─── IoU helpers ──────────────────────────────────────────────────────────────

def box_iou_np(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter_w  = np.maximum(0, inter_x2 - inter_x1)
    inter_h  = np.maximum(0, inter_y2 - inter_y1)
    inter    = inter_w * inter_h
    area_a   = (ax2 - ax1) * (ay2 - ay1)
    area_b   = (bx2 - bx1) * (by2 - by1)
    union    = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_predictions(pred_cls, gt_cls, iou_matrix, iou_thresh=0.5):
    tp = np.zeros(len(pred_cls), dtype=bool)
    if len(gt_cls) == 0 or len(pred_cls) == 0:
        return tp
    matched_gt = set()
    iou = iou_matrix.copy()
    iou[iou < iou_thresh] = 0
    for pi in range(len(pred_cls)):
        col = iou[:, pi].copy()
        col[list(matched_gt)] = 0
        best_gt = int(np.argmax(col))
        if col[best_gt] > 0 and gt_cls[best_gt] == pred_cls[pi]:
            tp[pi] = True
            matched_gt.add(best_gt)
    return tp


# ─── Label reader ─────────────────────────────────────────────────────────────

def read_label(label_path: Path, img_w: int, img_h: int):
    if not label_path.exists():
        return np.array([], dtype=int), np.zeros((0, 4), dtype=float)
    cls_ids, boxes = [], []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        cls_ids.append(c)
        boxes.append([x1, y1, x2, y2])
    return np.array(cls_ids, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


# ─── Region filter ────────────────────────────────────────────────────────────

def region_mask(boxes_xyxy: np.ndarray, img_h: int, top_crop: float) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=bool)
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    return cy >= top_crop * img_h


# ─── AP computation ───────────────────────────────────────────────────────────

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def pr_curve_data(conf_arr, tp_arr, n_gt):
    sort_idx  = np.argsort(-conf_arr)
    tp_sorted = tp_arr[sort_idx].astype(float)
    conf_sorted = conf_arr[sort_idx]
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1 - tp_sorted)
    recall    = tp_cum / (n_gt + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return recall, precision, f1, conf_sorted


# ─── Plot helpers ─────────────────────────────────────────────────────────────

CMAP_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def save_confusion_matrix(confusion: np.ndarray, out_dir: Path, nc: int):
    labels = CLASS_NAMES + ["BG"]

    # ── raw counts ──
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(nc + 1)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(nc + 1)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix (counts)")
    plt.colorbar(im, ax=ax)
    for i in range(nc + 1):
        for j in range(nc + 1):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if confusion[i, j] > confusion.max() * 0.6 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # ── normalised (row-wise) ──
    row_sum = confusion.sum(axis=1, keepdims=True).astype(float)
    conf_norm = np.where(row_sum > 0, confusion / row_sum, 0.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(nc + 1)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(nc + 1)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix (row-normalised)")
    plt.colorbar(im, ax=ax)
    for i in range(nc + 1):
        for j in range(nc + 1):
            ax.text(j, i, f"{conf_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if conf_norm[i, j] > 0.6 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_norm.png", dpi=300)
    plt.close(fig)


def save_pr_curves(all_conf, all_tp, all_n_gt, out_dir: Path, nc: int):
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(nc):
        if all_n_gt[c] == 0 or len(all_conf[c]) == 0:
            continue
        rec, prec, _, _ = pr_curve_data(
            np.array(all_conf[c]), np.array(all_tp[c], dtype=float), all_n_gt[c]
        )
        ap = compute_ap(rec, prec)
        ax.plot(rec, prec, color=CMAP_COLORS[c], linewidth=2,
                label=f"{CLASS_NAMES[c]} (AP={ap:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (IoU=0.50)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=300)
    plt.close(fig)


def save_f1_curves(all_conf, all_tp, all_n_gt, out_dir: Path, nc: int):
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(nc):
        if all_n_gt[c] == 0 or len(all_conf[c]) == 0:
            continue
        _, _, f1, conf_sorted = pr_curve_data(
            np.array(all_conf[c]), np.array(all_tp[c], dtype=float), all_n_gt[c]
        )
        ax.plot(conf_sorted, f1, color=CMAP_COLORS[c], linewidth=2, label=CLASS_NAMES[c])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("F1")
    ax.set_title("F1-Confidence Curves")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "f1_curves.png", dpi=300)
    plt.close(fig)


# ─── Worst-sample visualisation ───────────────────────────────────────────────

def score_image_errors(pred_boxes_f, pred_cls_f, gt_boxes_f, gt_cls_f):
    """Return total error count for one image: FP + FN."""
    if len(gt_boxes_f) == 0 and len(pred_boxes_f) == 0:
        return 0
    if len(gt_boxes_f) == 0:
        return len(pred_boxes_f)
    if len(pred_boxes_f) == 0:
        return len(gt_boxes_f)
    iou_mat  = box_iou_np(gt_boxes_f, pred_boxes_f)
    tp_flags = match_predictions(pred_cls_f, gt_cls_f, iou_mat, iou_thresh=0.5)
    fp = int((~tp_flags).sum())
    fn = len(gt_cls_f) - int(tp_flags.sum())
    return fp + max(fn, 0)


def draw_vis_image(img_path: Path, pred_boxes_f, pred_cls_f,
                   img_h: int, top_crop: float) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.zeros((img_h, 1280, 3), dtype=np.uint8)

    # crop line
    y_line = int(top_crop * img.shape[0])
    cv2.line(img, (0, y_line), (img.shape[1], y_line), (255, 255, 255), 3)

    # prediction boxes (below crop line only, no text)
    for box, cls in zip(pred_boxes_f, pred_cls_f):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = CLASS_COLORS_BGR[int(cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    return img


def save_worst_vis(img_error_list, pred_data, gt_data, out_dir: Path,
                   top_crop: float, n: int = 30):
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # sort by error count descending, take top-n
    sorted_items = sorted(img_error_list, key=lambda x: x[1], reverse=True)[:n]

    for rank, (img_path, err_count) in enumerate(sorted_items):
        pred_boxes_f, pred_cls_f = pred_data[img_path]
        gt_boxes_f,   gt_cls_f   = gt_data[img_path]
        img_h = cv2.imread(str(img_path)).shape[0] if img_path.exists() else 1280

        vis = draw_vis_image(img_path, pred_boxes_f, pred_cls_f, img_h, top_crop)

        out_name = f"rank{rank+1:02d}_err{err_count}_{img_path.stem}.jpg"
        cv2.imwrite(str(vis_dir / out_name), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # legend image
    legend_h = 40 * len(CLASS_NAMES) + 20
    legend   = np.ones((legend_h, 320, 3), dtype=np.uint8) * 240
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_BGR)):
        y = 20 + i * 40
        cv2.rectangle(legend, (10, y), (50, y + 25), color, -1)
        cv2.putText(legend, name, (60, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(vis_dir / "legend.png"), legend)

    print(f"[INFO] Visualisations saved to: {vis_dir}  ({len(sorted_items)} images)")


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(weights: Path, top_crop: float, save_dir: Path):
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    test_img_dir   = SPLIT_DIR / "images" / "test"
    test_label_dir = SPLIT_DIR / "labels" / "test"
    if not test_img_dir.exists():
        raise FileNotFoundError(f"Test images not found: {test_img_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model: {weights}")
    model = YOLO(str(weights))

    img_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp",
                      "*.JPG", "*.JPEG", "*.PNG", "*.BMP")
        for p in test_img_dir.glob(ext)
    )
    print(f"[INFO] Found {len(img_paths)} test images  |  top_crop={top_crop}")

    nc = len(CLASS_NAMES)
    confusion = np.zeros((nc + 1, nc + 1), dtype=int)   # rows=pred, cols=true (+BG)

    # per-class accumulators for PR / F1 curves
    all_conf = [[] for _ in range(nc)]
    all_tp   = [[] for _ in range(nc)]
    all_n_gt = [0] * nc

    # for worst-vis
    img_error_list = []   # [(img_path, error_count)]
    pred_data      = {}   # img_path -> (pred_boxes_f, pred_cls_f)
    gt_data        = {}   # img_path -> (gt_boxes_f,   gt_cls_f)

    for img_path in img_paths:
        result = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=DEVICE,
            verbose=False,
        )[0]

        img_h, img_w = result.orig_shape

        # ── ground truth ──
        label_path = test_label_dir / (img_path.stem + ".txt")
        gt_cls_all, gt_boxes_all = read_label(label_path, img_w, img_h)

        gt_mask      = region_mask(gt_boxes_all, img_h, top_crop) if len(gt_boxes_all) else np.array([], dtype=bool)
        gt_cls_f     = gt_cls_all[gt_mask]   if len(gt_cls_all)   else gt_cls_all
        gt_boxes_f   = gt_boxes_all[gt_mask] if len(gt_boxes_all) else gt_boxes_all

        # ── predictions ──
        if result.boxes is not None and len(result.boxes):
            pred_boxes_all = result.boxes.xyxy.cpu().numpy()
            pred_cls_all   = result.boxes.cls.cpu().numpy().astype(int)
            pred_conf_all  = result.boxes.conf.cpu().numpy()
        else:
            pred_boxes_all = np.zeros((0, 4), dtype=float)
            pred_cls_all   = np.array([], dtype=int)
            pred_conf_all  = np.array([], dtype=float)

        pred_mask    = region_mask(pred_boxes_all, img_h, top_crop) if len(pred_boxes_all) else np.array([], dtype=bool)
        pred_boxes_f = pred_boxes_all[pred_mask] if len(pred_boxes_all) else pred_boxes_all
        pred_cls_f   = pred_cls_all[pred_mask]   if len(pred_cls_all)   else pred_cls_all
        pred_conf_f  = pred_conf_all[pred_mask]  if len(pred_conf_all)  else pred_conf_all

        # ── confusion matrix + PR tp_flags ──
        if len(gt_boxes_f) > 0 and len(pred_boxes_f) > 0:
            iou_mat  = box_iou_np(gt_boxes_f, pred_boxes_f)
            tp_flags = match_predictions(pred_cls_f, gt_cls_f, iou_mat, iou_thresh=0.5)

            # IoU-only greedy match (class-agnostic) for confusion matrix
            # so that wrong-class predictions land on the correct GT column
            # instead of being collapsed into the BG column.
            iou_cm = iou_mat.copy()
            iou_cm[iou_cm < 0.5] = 0
            cm_matched_gt = set()
            pred_to_gt = {}
            for pi in range(len(pred_cls_f)):
                col = iou_cm[:, pi].copy()
                col[list(cm_matched_gt)] = 0
                best_gt = int(np.argmax(col))
                if col[best_gt] > 0:
                    pred_to_gt[pi] = best_gt
                    cm_matched_gt.add(best_gt)
        else:
            iou_mat  = np.zeros((len(gt_boxes_f), len(pred_boxes_f)))
            tp_flags = np.zeros(len(pred_cls_f), dtype=bool)
            cm_matched_gt = set()
            pred_to_gt = {}

        for pi, pc in enumerate(pred_cls_f):
            if pi in pred_to_gt:
                confusion[pc, gt_cls_f[pred_to_gt[pi]]] += 1  # TP (diagonal) or misclassification
            else:
                confusion[pc, nc] += 1                          # unmatched FP → BG col

        for gi, gc in enumerate(gt_cls_f):
            if gi not in cm_matched_gt:
                confusion[nc, gc] += 1                          # unmatched FN → BG row

        # ── per-class PR accumulators ──
        for c in range(nc):
            all_n_gt[c] += int((gt_cls_f == c).sum())
            mask_c = pred_cls_f == c
            if mask_c.any():
                all_conf[c].extend(pred_conf_f[mask_c].tolist())
                all_tp[c].extend(tp_flags[mask_c].tolist())

        # ── worst-vis bookkeeping ──
        err = score_image_errors(pred_boxes_f, pred_cls_f, gt_boxes_f, gt_cls_f)
        img_error_list.append((img_path, err))
        pred_data[img_path] = (pred_boxes_f, pred_cls_f)
        gt_data[img_path]   = (gt_boxes_f,   gt_cls_f)

    # ── save outputs ──
    save_confusion_matrix(confusion, save_dir, nc)
    print(f"[INFO] Confusion matrices saved to: {save_dir}")

    save_pr_curves(all_conf, all_tp, all_n_gt, save_dir, nc)
    save_f1_curves(all_conf, all_tp, all_n_gt, save_dir, nc)
    print(f"[INFO] Curve plots saved to: {save_dir}")

    save_worst_vis(img_error_list, pred_data, gt_data, save_dir, top_crop, n=30)

    # ── summary metrics ──
    aps = []
    print("\n─── Per-class AP50 (region-filtered) ───────────────────")
    for c in range(nc):
        if all_n_gt[c] == 0 or len(all_conf[c]) == 0:
            print(f"  {CLASS_NAMES[c]:<15}: N/A  (no GT)")
            continue
        rec, prec, _, _ = pr_curve_data(
            np.array(all_conf[c]), np.array(all_tp[c], dtype=float), all_n_gt[c]
        )
        ap = compute_ap(rec, prec)
        aps.append(ap)
        print(f"  {CLASS_NAMES[c]:<15}: {ap:.4f}")
    if aps:
        print(f"  {'mAP50':<15}: {np.mean(aps):.4f}")
    print("────────────────────────────────────────────────────────")
    print(f"[DONE] All outputs saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLO26n Region-Filtered Evaluation")
    parser.add_argument("--weights",  type=Path,  default=DEFAULT_WEIGHTS)
    parser.add_argument("--top-crop", type=float, default=0.20,
                        help="Fraction of image height to ignore from top (default 0.15)")
    parser.add_argument("--save-dir", type=Path,
                        default=BASE_DIR / "runs" / "test" / "yolo26n_baseline_region")
    args = parser.parse_args()

    evaluate(args.weights, args.top_crop, args.save_dir)


if __name__ == "__main__":
    main()
