"""
merge_first_second_data.py

Merges 'First training' and 'Second training' datasets into a unified split.

Steps:
  1. Collect all images from both dataset folders
  2. Split each dataset independently (70/15/15 train/val/test) with the same seed
  3. Copy images + labels into ./cameratrap_data/images/{train,val,test}
                                  ./cameratrap_data/labels/{train,val,test}
  4. Write ./cameratrap_data/dataset.yaml
"""

import shutil
import random
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
SOURCES     = [
    BASE_DIR / "First training",
    BASE_DIR / "Second training",
]
OUT_DIR     = BASE_DIR / "cameratrap_data"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = remaining 0.15
SEED        = 42
IMG_SUFFIXES = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}

YAML_CONTENT = """\
path: ./cameratrap_data
train: images/train
val:   images/val
test:  images/test

nc: 4
names:
  0: Brent_Up
  1: Brent_Down
  2: Barnacle_Up
  3: Barnacle_Down
"""


def split_images(images: list[Path], seed: int) -> dict[str, list[Path]]:
    """Return {'train': [...], 'val': [...], 'test': [...]}."""
    imgs = images.copy()
    random.seed(seed)
    random.shuffle(imgs)
    n       = len(imgs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return {
        "train": imgs[:n_train],
        "val":   imgs[n_train:n_train + n_val],
        "test":  imgs[n_train + n_val:],
    }


def copy_files(img_path: Path, labels_dir: Path, img_out: Path, lbl_out: Path) -> None:
    """Copy one image and its label (or create empty label) to the output dirs."""
    shutil.copy2(img_path, img_out / img_path.name)
    lbl_src = labels_dir / (img_path.stem + ".txt")
    lbl_dst = lbl_out / (img_path.stem + ".txt")
    if lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)
    else:
        lbl_dst.touch()


def main():
    # ── Validate sources ──────────────────────────────────────────────────────
    for src in SOURCES:
        if not src.exists():
            raise FileNotFoundError(f"Source folder not found: {src}")
        if not (src / "images").exists():
            raise FileNotFoundError(f"'images' subfolder missing in: {src}")
        if not (src / "labels").exists():
            raise FileNotFoundError(f"'labels' subfolder missing in: {src}")

    # ── Create output directory structure ─────────────────────────────────────
    for split in ("train", "val", "test"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Process each source independently ────────────────────────────────────
    totals = {"train": 0, "val": 0, "test": 0}

    for src in SOURCES:
        images_dir = src / "images"
        labels_dir = src / "labels"

        all_images = [
            p for p in images_dir.iterdir()
            if p.suffix in IMG_SUFFIXES
        ]
        all_images.sort()

        if not all_images:
            print(f"[WARN] No images found in {images_dir}, skipping.")
            continue

        splits = split_images(all_images, seed=SEED)

        print(f"\n[INFO] {src.name}")
        for split, imgs in splits.items():
            img_out = OUT_DIR / "images" / split
            lbl_out = OUT_DIR / "labels" / split
            for img_path in imgs:
                copy_files(img_path, labels_dir, img_out, lbl_out)
            print(f"  {split:<6}: {len(imgs):>4} images")
            totals[split] += len(imgs)

    # ── Write dataset.yaml ────────────────────────────────────────────────────
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(YAML_CONTENT)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[DONE] Output: {OUT_DIR}")
    print(f"  train: {totals['train']} images")
    print(f"  val  : {totals['val']} images")
    print(f"  test : {totals['test']} images")
    print(f"  total: {sum(totals.values())} images")
    print(f"  yaml : {yaml_path}")


if __name__ == "__main__":
    main()
