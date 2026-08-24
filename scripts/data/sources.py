"""The five public datasets: sources, URLs, and download orchestration.

| # | Dataset                          | Source            | Type           |
|---|----------------------------------|-------------------|----------------|
| 1 | Synthetic Dyslexia Handwriting   | Kaggle (YOLO)     | Image + labels |
| 2 | Rello et al. Dyslexia            | Kaggle (luzrello) | Tabular        |
| 3 | Drizasazanitaisa Handwriting     | Kaggle (gambo)    | Image (rar)    |
| 4 | Potential Dysgraphia             | Mendeley          | Image          |
| 5 | ETDD70 Eye-Tracking              | Zenodo            | Gaze + CSV     |
"""
from __future__ import annotations

import requests
from pathlib import Path

from scripts.common import already_have_enough
from scripts.data.download import (
    download_file,
    download_kaggle_dataset,
    download_kaggle_dataset_nozip,
)
from scripts.data.extract import extract_all_archives, unzip_file

SCRIPT_DIR = Path(__file__).resolve().parent.parent      # .../scripts
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_DIR = PROJECT_ROOT / "dyslexia_datasets"

GAMBO_ZIP_PASSWORD = "WanAsy321"

ZENODO_RECORD_ID = "13332134"
ZENODO_FILES = [
    "data.zip",
    "dyslexia_class_label.csv",
    "fixation_images.zip",
    "README.md",
    "rois.zip",
    "stimuli.zip",
]

MENDELEY_ZIP_URL = "https://data.mendeley.com/public-api/zip/39hr8dx76p/download/1"


def get_dataset_1_synthetic(force: bool = False) -> Path:
    """Kaggle: synthetic dyslexia handwriting with YOLO labels."""
    dest = BASE_DIR / "1_synthetic_handwriting"
    if already_have_enough(dest, min_files=100, force=force):
        return dest
    download_kaggle_dataset("michaelfink0923/synthetic-dyslexia-handwriting-dataset", dest)
    return dest


def get_dataset_2_luzrello(force: bool = False) -> Path:
    """Kaggle: Rello et al. tabular dyslexia data."""
    dest = BASE_DIR / "2_luzrello_tabular"
    if already_have_enough(dest, min_files=2, force=force):
        return dest
    download_kaggle_dataset("luzrello/dyslexia", dest)
    return dest


def get_dataset_3_drizasazanitaisa(force: bool = False) -> Path:
    """Kaggle: handwriting images inside a password-protected rar."""
    dest = BASE_DIR / "3_drizasazanitaisa_handwriting"
    if already_have_enough(dest, min_files=100, force=force):
        return dest

    existing_zips = list(dest.glob("*.zip")) if dest.exists() else []
    if existing_zips and not force:
        print(f"\n[skip] {dest} already has {[z.name for z in existing_zips]} — skipping re-download.")
    else:
        download_kaggle_dataset_nozip("drizasazanitaisa/dyslexia-handwriting-dataset", dest)

    extract_all_archives(dest, password=GAMBO_ZIP_PASSWORD)
    return dest


def get_dataset_4_mendeley(force: bool = False) -> Path:
    """Mendeley: potential dysgraphia handwriting images."""
    dest = BASE_DIR / "4_mendeley_dysgraphia"
    dest.mkdir(parents=True, exist_ok=True)

    if already_have_enough(dest, min_files=10, force=force):
        return dest

    zip_path = dest / "mendeley_dysgraphia.zip"
    if zip_path.exists() and not force:
        print(f"\n[skip] {zip_path} already downloaded — extracting only.")
        ok = True
    else:
        ok = download_file(MENDELEY_ZIP_URL, zip_path,
                           headers={"User-Agent": "Mozilla/5.0"})
    if ok:
        unzip_file(zip_path, dest)
        extract_all_archives(dest)
    else:
        print("  [!] Automated download failed. Manual fallback:")
        print("      https://data.mendeley.com/datasets/39hr8dx76p/1")
    return dest


def get_dataset_5_etdd70(force: bool = False) -> Path:
    """Zenodo: ETDD70 eye-tracking dataset."""
    dest = BASE_DIR / "5_etdd70_eyetracking"
    dest.mkdir(parents=True, exist_ok=True)

    if already_have_enough(dest, min_files=len(ZENODO_FILES) + 5, force=force):
        return dest

    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    browser_headers = {
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept": "application/json",
    }

    print(f"\n[Zenodo] Querying record metadata: {api_url}")
    files_to_get = []
    try:
        r = requests.get(api_url, headers=browser_headers, timeout=60)
        r.raise_for_status()
        record = r.json()
        for f in record.get("files", []):
            fname = f.get("key") or f.get("filename")
            file_url = f["links"]["self"]
            files_to_get.append((fname, file_url))
        print(f"  API succeeded — found {len(files_to_get)} file(s).")
    except Exception as e:
        print(f"  API failed ({e}). Falling back to hardcoded URLs.")
        files_to_get = [
            (fname, f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{fname}?download=1")
            for fname in ZENODO_FILES
        ]

    for fname, file_url in files_to_get:
        target = dest / fname
        if target.exists() and not force:
            print(f"\n[skip] {target} already downloaded — skipping.")
            ok = True
        else:
            ok = download_file(file_url, target, headers=browser_headers)
        if ok and fname.lower().endswith(".zip"):
            from scripts.common import count_files
            already_extracted = count_files(dest) > len(ZENODO_FILES) + 2
            if not (already_extracted and not force):
                unzip_file(target, dest)

    return dest
