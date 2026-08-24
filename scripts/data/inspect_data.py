"""Human-readable dataset inspection (folder stats, label columns, class counts)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.common import IMAGE_EXTS, TABULAR_EXTS, is_junk


def inspect_folder(folder: Path, label: str):
    """Print a summary of everything inside a raw dataset folder."""
    print(f"\n{'=' * 70}\nDATASET: {label}\nPath: {folder}\n{'=' * 70}")

    if not folder.exists():
        print("  [!] Folder does not exist - download may have failed.")
        return

    all_files = list(folder.rglob("*"))
    files_only = [f for f in all_files if f.is_file() and not is_junk(f)]
    print(f"  Total files found: {len(files_only)}")

    if len(files_only) <= 2:
        zips_present = [f for f in folder.rglob("*.zip") if not is_junk(f)]
        rars_present = [f for f in folder.rglob("*.rar") if not is_junk(f)]
        if zips_present or rars_present:
            print(f"  [!] Only {len(files_only)} file(s) found but archives exist.")
            print("      Extraction likely failed. Check passwords / install unrar.")

    for yaml_file in folder.rglob("data.yaml"):
        if is_junk(yaml_file):
            continue
        print(f"\n  --- Found YOLO config: {yaml_file.relative_to(folder)} ---")
        try:
            print(yaml_file.read_text())
        except Exception as e:
            print(f"  Could not read {yaml_file}: {e}")

    tabular_files = [f for f in files_only if f.suffix.lower() in TABULAR_EXTS]
    for tf in tabular_files:
        try:
            df = pd.read_csv(tf, sep=None, engine="python", nrows=1000)
            print(f"\n  --- Tabular file: {tf.relative_to(folder)} ---")
            print(f"      Shape (first 1000 rows): {df.shape}")
            print(f"      Columns: {list(df.columns)}")
            possible_label_cols = [c for c in df.columns
                                   if any(k in c.lower() for k in
                                          ["label", "class", "dyslexi", "target", "risk"])]
            if possible_label_cols:
                for col in possible_label_cols:
                    print(f"      Label column '{col}' value counts:")
                    print(df[col].value_counts().to_string().replace("\n", "\n            "))
        except Exception as e:
            print(f"  Could not read {tf}: {e}")

    image_files = [f for f in files_only if f.suffix.lower() in IMAGE_EXTS]
    if image_files:
        print(f"\n  --- Image files: {len(image_files)} total ---")
        class_counts = {}
        for img in image_files:
            class_name = img.parent.name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print("  Class counts (by containing folder name):")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"      {cls}: {count}")
        print(f"  Sample filenames: {[f.name for f in image_files[:5]]}")

    other_files = [f for f in files_only
                   if f.suffix.lower() not in IMAGE_EXTS | TABULAR_EXTS]
    interesting_exts = {".json", ".txt", ".xml", ".yaml", ".yml", ".md"}
    interesting = [f for f in other_files if f.suffix.lower() in interesting_exts]
    if interesting:
        print(f"\n  --- Other metadata files (first 10) ---")
        for f in interesting[:10]:
            print(f"      {f.relative_to(folder)}")
