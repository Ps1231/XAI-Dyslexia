"""Stage 1 — download, extract, and organize all datasets (idempotent).

Programmatic:
    from scripts.data.prepare import run_prepare
    run_prepare()                        # skips finished datasets
    run_prepare(force=True)              # redo everything

CLI:
    python -m scripts.data.prepare [--force] [--skip-download] [--inspect]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.common import dir_fingerprint, load_state, save_state
from scripts.data.sources import (
    BASE_DIR,
    get_dataset_1_synthetic,
    get_dataset_2_luzrello,
    get_dataset_3_drizasazanitaisa,
    get_dataset_4_mendeley,
    get_dataset_5_etdd70,
)
from scripts.data.organize import (
    organize_dysgraphia_images,
    organize_dyslexia_images,
    organize_eyetracking,
    organize_tabular,
    organize_yolo_dyslexia,
)
from scripts.data.inspect_data import inspect_folder

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def run_prepare(
    force: bool = False,
    output=None,
    skip_download: bool = False,
    inspect_data: bool = False,
) -> dict:
    """Download, extract, and organize all datasets — skipping finished work.

    Each dataset gets a fingerprint marker; when the raw folder is unchanged
    and the organized output already exists, that dataset is skipped entirely.

    Returns:
        Dict mapping dataset key -> "done" | "skipped (up-to-date)".
    """
    output = str(Path(output) if output else DEFAULT_OUTPUT_DIR)
    state_dir = Path(output) / ".state"
    os.makedirs(output, exist_ok=True)

    if not skip_download:
        print("Starting download of all 5 dyslexia-related datasets"
              f"{' (forced re-download)' if force else ''}...\n")
        d1 = get_dataset_1_synthetic(force=force)
        d2 = get_dataset_2_luzrello(force=force)
        d3 = get_dataset_3_drizasazanitaisa(force=force)
        d4 = get_dataset_4_mendeley(force=force)
        d5 = get_dataset_5_etdd70(force=force)
    else:
        print("Skipping download — using existing dyslexia_datasets/ folders.\n")
        d1 = BASE_DIR / "1_synthetic_handwriting"
        d2 = BASE_DIR / "2_luzrello_tabular"
        d3 = BASE_DIR / "3_drizasazanitaisa_handwriting"
        d4 = BASE_DIR / "4_mendeley_dysgraphia"
        d5 = BASE_DIR / "5_etdd70_eyetracking"

    if inspect_data:
        print("\n\nStarting inspection...")
        inspect_folder(d1, "1. Synthetic Dyslexia Handwriting Dataset (YOLO)")
        inspect_folder(d2, "2. Rello et al. / luzrello Dyslexia (Tabular)")
        inspect_folder(d3, "3. Dyslexia Handwriting Dataset (Normal/Reversal/Corrected)")
        inspect_folder(d4, "4. Potential Dysgraphia Handwriting Dataset (Mendeley)")
        inspect_folder(d5, "5. ETDD70 Eye-Tracking Dyslexia Dataset (Zenodo)")

    steps = [
        ("1_synthetic",        d1, lambda o: organize_yolo_dyslexia(d1, o, target_name="dyslexia_synthetic"),    "dyslexia_synthetic"),
        ("2_luzrello",         d2, lambda o: organize_tabular(d2, o),                                             "tabular"),
        ("3_drizasazanitaisa", d3, lambda o: organize_dyslexia_images(d3, o, target_name="dyslexia_handwriting"), "dyslexia_handwriting"),
        ("4_mendeley",         d4, lambda o: organize_dysgraphia_images(d4, o),                                   "dysgraphia"),
        ("5_etdd70",           d5, lambda o: organize_eyetracking(d5, o),                                         "eyetracking"),
    ]

    print(f"\n\nOrganizing into {output}/ for training...")
    results = {}
    for key, src, organize_fn, target_name in steps:
        fp = dir_fingerprint(src)
        target_dir = Path(output) / target_name
        has_files = target_dir.exists() and any(target_dir.iterdir())

        if not force and fp and has_files:
            prev = load_state(state_dir, f"prepare_{key}")
            if prev and prev.get("fingerprint") == fp:
                print(f"\n[{key}] up-to-date — organization already done, skipping")
                results[key] = "skipped (up-to-date)"
                continue

        print(f"\n[{key} - {target_name}]")
        organize_fn(str(output))
        if fp:
            save_state(state_dir, f"prepare_{key}", {"fingerprint": fp})
        results[key] = "done"

    return results


def main():
    """Optional CLI shim."""
    parser = argparse.ArgumentParser(description="Download + organize dyslexia datasets")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-extract everything.")
    parser.add_argument("--output", default=None,
                        help="Where to write organized data for training.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download/extract, just organize what's already there.")
    parser.add_argument("--inspect", action="store_true",
                        help="Print detailed dataset inspection.")
    args = parser.parse_args()

    run_prepare(
        force=args.force,
        output=args.output,
        skip_download=args.skip_download,
        inspect_data=args.inspect,
    )


if __name__ == "__main__":
    main()
