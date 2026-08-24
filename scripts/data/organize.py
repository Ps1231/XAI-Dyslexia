"""Copy/organize extracted datasets into the training directory layout:

    data/processed/
    ├── dysgraphia/<class>/            (dataset 4)
    ├── dyslexia_synthetic/<class>/    (dataset 1, YOLO label parsing)
    ├── dyslexia_handwriting/<class>/  (dataset 3)
    ├── tabular/*.csv                  (dataset 2)
    └── eyetracking/*.csv              (dataset 5)
"""
from __future__ import annotations

import ast
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from scripts.common import IMAGE_EXTS, IMG_SUFFIXES, is_junk

_shutil = shutil


def _copy_if_exists(src, dest_dir: str) -> bool:
    """Copy *src* into *dest_dir* unless an identically named file is already there.

    Creates *dest_dir* if needed. Returns True when a copy happened,
    False when the item already existed.
    """
    dest = os.path.join(dest_dir, os.path.basename(str(src)))
    if os.path.exists(dest):
        return False
    os.makedirs(dest_dir, exist_ok=True)
    _shutil.copy2(str(src), dest)
    return True


# -------------------------------------------------------------
# Dataset 1 — YOLO labels -> binary or 3-class image folders
# -------------------------------------------------------------

def organize_yolo_dyslexia(source_dir: Path, output_dir: str,
                           target_name: str = 'dyslexia_synthetic'):
    """Parse YOLO labels to create image-level classes.

    Strategy:
      1. Try binary: normal (only class-0 objects) vs dyslexic (any class-1/2).
      2. If 'normal' is empty, fall back to 3-class majority vote.
    """
    target_dir = os.path.join(output_dir, target_name)
    os.makedirs(target_dir, exist_ok=True)

    yaml_files = [f for f in source_dir.rglob("data.yaml") if not is_junk(f)]
    if not yaml_files:
        print(f"  WARNING: no data.yaml found in {source_dir}; skipping YOLO org")
        return

    data_yaml = yaml_files[0]
    base = data_yaml.parent

    names = ['Normal', 'Reversal', 'Corrected']
    try:
        with open(data_yaml) as f:
            for line in f:
                line = line.strip()
                if line.startswith('names:'):
                    list_str = line.split(':', 1)[1].strip()
                    names = ast.literal_eval(list_str)
                    break
    except Exception as e:
        print(f"  Could not parse data.yaml names: {e}; using default {names}")

    print(f"  YOLO label names: {names}")

    # First pass: try binary split
    normal_dir = os.path.join(target_dir, 'normal')
    dyslexic_dir = os.path.join(target_dir, 'dyslexic')
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(dyslexic_dir, exist_ok=True)

    n_normal = 0
    n_dyslexic = 0
    n_skipped = 0

    for split in ['train', 'val']:
        img_dir = base / 'images' / split
        lbl_dir = base / 'labels' / split
        if not img_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            if is_junk(img_file):
                continue

            lbl_file = lbl_dir / (img_file.stem + '.txt')
            is_dyslexic = False

            if lbl_file.exists():
                with open(lbl_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                cls_id = int(line.split()[0])
                                if cls_id != 0:
                                    is_dyslexic = True
                                    break
                            except ValueError:
                                continue

            dest = dyslexic_dir if is_dyslexic else normal_dir
            if not _copy_if_exists(img_file, dest):
                n_skipped += 1
            if is_dyslexic:
                n_dyslexic += 1
            else:
                n_normal += 1

    skipped_note = f" ({n_skipped} already existed)" if n_skipped else ""
    print(f"  Binary split: {n_normal} normal, {n_dyslexic} dyslexic{skipped_note}")

    # Fallback: if normal is empty, use 3-class majority vote
    if n_normal == 0 and n_dyslexic > 0:
        print("  WARNING: no purely 'normal' images found (mixed-letter dataset).")
        print("  Falling back to 3-class majority-vote organization...")

        for d in [normal_dir, dyslexic_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
                os.rmdir(d)

        class_dirs = {}
        for idx, name in enumerate(names):
            cname = name.lower().replace(' ', '_')
            cdir = os.path.join(target_dir, cname)
            os.makedirs(cdir, exist_ok=True)
            class_dirs[idx] = cdir

        for split in ['train', 'val']:
            img_dir = base / 'images' / split
            lbl_dir = base / 'labels' / split
            if not img_dir.exists():
                continue

            for img_file in img_dir.iterdir():
                if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTS:
                    continue
                if is_junk(img_file):
                    continue

                lbl_file = lbl_dir / (img_file.stem + '.txt')
                vote = Counter()

                if lbl_file.exists():
                    with open(lbl_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    cls_id = int(line.split()[0])
                                    vote[cls_id] += 1
                                except ValueError:
                                    continue

                winner = vote.most_common(1)[0][0] if vote else 0
                dest = class_dirs.get(winner, class_dirs[0])
                _copy_if_exists(img_file, dest)

        for idx, cdir in class_dirs.items():
            n = len([f for f in os.listdir(cdir) if os.path.isfile(os.path.join(cdir, f))])
            print(f"    Class '{names[idx]}': {n} images")
    else:
        print(f"  Organized YOLO images -> {target_dir}")


# -------------------------------------------------------------
# Dataset 3 — handwriting images grouped by parent folder
# -------------------------------------------------------------

def organize_dyslexia_images(source_dir: Path, output_dir: str,
                             target_name: str = 'dyslexia_handwriting'):
    """Organize a handwriting dataset into class folders.

    Recursively finds all image files and groups them by their immediate
    parent folder name, handling arbitrary nesting depth.
    """
    target_dir = os.path.join(output_dir, target_name)
    os.makedirs(target_dir, exist_ok=True)

    all_images = [f for f in source_dir.rglob("*")
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS and not is_junk(f)]

    if not all_images:
        print(f"  WARNING: no images found under {source_dir}")
        return

    class_images = defaultdict(list)
    for img in all_images:
        class_name = img.parent.name.lower().replace(' ', '_')
        class_images[class_name].append(img)

    # If all images share one parent (wrapper folder like 'gambo'), look deeper
    if len(class_images) == 1:
        wrapper_name = list(class_images.keys())[0]
        print(f"  Detected wrapper folder '{wrapper_name}' — looking deeper...")
        class_images = defaultdict(list)
        for img in all_images:
            if img.parent.name.lower().replace(' ', '_') == wrapper_name:
                gp = img.parent.parent
                if gp and gp != source_dir:
                    class_name = gp.name.lower().replace(' ', '_')
                else:
                    class_name = img.parent.name.lower().replace(' ', '_')
            else:
                class_name = img.parent.name.lower().replace(' ', '_')
            class_images[class_name].append(img)

    for class_name, imgs in sorted(class_images.items()):
        class_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        copied = sum(_copy_if_exists(img, class_dir) for img in imgs)
        skipped = len(imgs) - copied
        note = f" ({skipped} already existed)" if skipped else ""
        print(f"  Class '{class_name}': {len(imgs)} images{note}")

    print(f"  Organized images into {target_dir}")


# -------------------------------------------------------------
# Dataset 4 — Mendeley dysgraphia images
# -------------------------------------------------------------

def organize_dysgraphia_images(source_dir: Path, output_dir: str):
    """Organize Mendeley dysgraphia dataset into class folders."""
    target_dir = os.path.join(output_dir, 'dysgraphia')
    os.makedirs(target_dir, exist_ok=True)

    copied = 0
    skipped = 0
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(IMG_SUFFIXES):
                if f.startswith("._"):
                    continue
                folder_name = os.path.basename(root).lower().replace(' ', '_')
                class_dir = os.path.join(target_dir, folder_name)
                if _copy_if_exists(os.path.join(root, f), class_dir):
                    copied += 1
                else:
                    skipped += 1

    note = f" ({skipped} already existed)" if skipped else ""
    print(f"  Copied {copied} images{note} into {target_dir}")


# -------------------------------------------------------------
# Datasets 2 & 5 — CSV copies
# -------------------------------------------------------------

def organize_tabular(source_dir: Path, output_dir: str):
    """Copy Rello tabular CSVs as-is."""
    target_dir = os.path.join(output_dir, 'tabular')
    os.makedirs(target_dir, exist_ok=True)

    copied = []
    skipped = 0
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith('.csv') and not f.startswith("._"):
                if _copy_if_exists(os.path.join(root, f), target_dir):
                    copied.append(f)
                else:
                    skipped += 1

    skipped_note = f" ({skipped} already existed)" if skipped else ""
    if not copied and not skipped:
        print(f"  WARNING: no CSV files found under {source_dir}")
    else:
        print(f"  Copied {copied}{skipped_note} into {target_dir}")


def organize_eyetracking(source_dir: Path, output_dir: str):
    """Copy ETDD70 label CSVs; skip stimulus images & macOS junk."""
    target_dir = os.path.join(output_dir, 'eyetracking')
    os.makedirs(target_dir, exist_ok=True)

    n_copied = 0
    n_skipped = 0
    for root, dirs, files in os.walk(source_dir):
        if os.path.basename(root).lower() == "__macosx":
            continue
        for f in files:
            if f.startswith("._"):
                continue
            if f.lower().endswith('.csv'):
                if _copy_if_exists(os.path.join(root, f), target_dir):
                    n_copied += 1
                else:
                    n_skipped += 1

    skipped_note = f" ({n_skipped} already existed)" if n_skipped else ""
    print(f"  Copied {n_copied} CSV(s){skipped_note} into {target_dir}")
    print("  NOTE: ETDD70 stimulus images are NOT for image classification.")
    print("        Build a gaze-feature pipeline (fixation duration, saccade length, etc.)")
