"""
Dyslexia Multi-Dataset Downloader + Inspector  (v3 — fixes applied)
====================================================================
Downloads all 5 datasets and prints a structural report for each.

v3 FIXES:
- Dataset #3: .rar extraction support (unrar/7z/unar fallback chain)
- Dataset #4: nested zip auto-extraction
- Dataset #5: progress bar for large downloads + data.zip extraction
"""

import os
import sys
import argparse
import zipfile
import subprocess
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ -> XAI-Dyslexia/

BASE_DIR = PROJECT_ROOT / "dyslexia_datasets"
BASE_DIR.mkdir(exist_ok=True)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FORCE_REDOWNLOAD = False

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


def count_files(folder: Path, exclude_zip: bool = True) -> int:
    if not folder.exists():
        return 0
    files = [f for f in folder.rglob("*") if f.is_file()
             and (not exclude_zip or f.suffix.lower() != ".zip")]
    return len(files)


def already_have_enough(folder: Path, min_files: int) -> bool:
    if FORCE_REDOWNLOAD:
        return False
    n = count_files(folder)
    if n >= min_files:
        print(f"\n[skip] {folder} already has {n} extracted file(s) "
              f"(>= {min_files}) — skipping download. Use --force to redo.")
        return True
    return False


# -------------------------------------------------------------
# DOWNLOAD HELPERS
# -------------------------------------------------------------

def download_file(url: str, dest_path: Path, headers=None):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[HTTP] Downloading {url} -> {dest_path}")
    try:
        r = requests.get(url, stream=True, timeout=120, headers=headers or {})
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f:
            if total > 0:
                with tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        size_mb = dest_path.stat().st_size / 1e6
        print(f"  Saved {dest_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        return False


def download_kaggle_dataset(kaggle_ref: str, dest_folder: Path):
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[Kaggle] Downloading {kaggle_ref} -> {dest_folder}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_ref,
             "-p", str(dest_folder), "--unzip"],
            check=True
        )
    except FileNotFoundError:
        print("  ERROR: kaggle CLI not found. Run: pip install kaggle")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR downloading {kaggle_ref}: {e}")


def download_kaggle_dataset_nozip(kaggle_ref: str, dest_folder: Path):
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[Kaggle] Downloading {kaggle_ref} -> {dest_folder} (no auto-unzip)")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_ref,
             "-p", str(dest_folder)],
            check=True
        )
        for z in dest_folder.glob("*.zip"):
            print(f"  Downloaded archive: {z.name}")
    except FileNotFoundError:
        print("  ERROR: kaggle CLI not found. Run: pip install kaggle")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR downloading {kaggle_ref}: {e}")


# -------------------------------------------------------------
# EXTRACTION HELPERS
# -------------------------------------------------------------

def unzip_file(zip_path: Path, dest_folder: Path):
    if not zip_path.exists():
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_folder)
        print(f"  Unzipped -> {dest_folder}")
        return True
    except RuntimeError as e:
        print(f"  Could not unzip with zipfile: {e}")
        return False


def unzip_with_password(zip_path: Path, dest_folder: Path, password: str):
    if not zip_path.exists():
        return False
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[unzip] Extracting {zip_path.name} with password -> {dest_folder}")
    try:
        subprocess.run(
            ["unzip", "-o", "-P", password, str(zip_path), "-d", str(dest_folder)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("  Extraction succeeded.")
        return True
    except FileNotFoundError:
        print("  ERROR: `unzip` command not found.")
        return False
    except subprocess.CalledProcessError:
        print(f"  ERROR extracting with password.")
        return False


def extract_rar(rar_path: Path, dest_folder: Path, password: str):
    """Extract .rar with password using unrar, 7z, or unar."""
    if not rar_path.exists():
        return False
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[rar] Extracting {rar_path.name} with password -> {dest_folder}")

    # Try unrar first
    if subprocess.run(["which", "unrar"], capture_output=True).returncode == 0:
        try:
            subprocess.run(
                ["unrar", "x", "-y", f"-p{password}", str(rar_path), str(dest_folder) + "/"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  Extraction succeeded (unrar).")
            return True
        except subprocess.CalledProcessError:
            pass

    # Try 7z
    if subprocess.run(["which", "7z"], capture_output=True).returncode == 0:
        try:
            subprocess.run(
                ["7z", "x", str(rar_path), f"-o{dest_folder}", f"-p{password}", "-y"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  Extraction succeeded (7z).")
            return True
        except subprocess.CalledProcessError:
            pass

    # Try unar
    if subprocess.run(["which", "unar"], capture_output=True).returncode == 0:
        try:
            subprocess.run(
                ["unar", "-f", "-o", str(dest_folder), "-p", password, str(rar_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  Extraction succeeded (unar).")
            return True
        except subprocess.CalledProcessError:
            pass

    print("  ERROR: No .rar extractor found. Install one of: unrar, p7zip-full, unar")
    print("    Ubuntu/Debian: sudo apt install unrar p7zip-full")
    return False


def extract_all_archives(folder: Path, password: str = None, depth: int = 0):
    """Recursively extract all zip/rar files in a folder."""
    if depth > 3:
        return
    archives = list(folder.rglob("*.zip")) + list(folder.rglob("*.rar"))
    for arch in archives:
        # Skip if already extracted (check for folder with same name)
        extract_to = arch.with_suffix("")
        if extract_to.exists() and count_files(extract_to) > 0:
            continue
        if arch.suffix.lower() == ".zip":
            if password:
                ok = unzip_with_password(arch, folder, password)
                if not ok:
                    unzip_file(arch, folder)
            else:
                unzip_file(arch, folder)
        elif arch.suffix.lower() == ".rar":
            if password:
                extract_rar(arch, folder, password)
        # Recurse for nested archives
        extract_all_archives(folder, password, depth + 1)


# -------------------------------------------------------------
# DATASET-SPECIFIC FUNCTIONS
# -------------------------------------------------------------

def get_dataset_1_synthetic():
    dest = BASE_DIR / "1_synthetic_handwriting"
    if already_have_enough(dest, min_files=100):
        return dest
    download_kaggle_dataset("michaelfink0923/synthetic-dyslexia-handwriting-dataset", dest)
    return dest


def get_dataset_2_luzrello():
    dest = BASE_DIR / "2_luzrello_tabular"
    if already_have_enough(dest, min_files=2):
        return dest
    download_kaggle_dataset("luzrello/dyslexia", dest)
    return dest


def get_dataset_3_drizasazanitaisa():
    dest = BASE_DIR / "3_drizasazanitaisa_handwriting"
    if already_have_enough(dest, min_files=100):
        return dest

    existing_zips = list(dest.glob("*.zip")) if dest.exists() else []
    if existing_zips and not FORCE_REDOWNLOAD:
        print(f"\n[skip] {dest} already has {[z.name for z in existing_zips]} — skipping re-download.")
    else:
        download_kaggle_dataset_nozip("drizasazanitaisa/dyslexia-handwriting-dataset", dest)

    # Extract everything (zip -> rar -> images)
    extract_all_archives(dest, password=GAMBO_ZIP_PASSWORD)
    return dest


def get_dataset_4_mendeley():
    dest = BASE_DIR / "4_mendeley_dysgraphia"
    dest.mkdir(parents=True, exist_ok=True)

    if already_have_enough(dest, min_files=10):
        return dest

    zip_path = dest / "mendeley_dysgraphia.zip"
    if zip_path.exists() and not FORCE_REDOWNLOAD:
        print(f"\n[skip] {zip_path} already downloaded — extracting only.")
        ok = True
    else:
        ok = download_file(MENDELEY_ZIP_URL, zip_path,
                           headers={"User-Agent": "Mozilla/5.0"})
    if ok:
        unzip_file(zip_path, dest)
        # Handle nested zips
        extract_all_archives(dest)
    else:
        print("  [!] Automated download failed. Manual fallback:")
        print("      https://data.mendeley.com/datasets/39hr8dx76p/1")
    return dest


def get_dataset_5_etdd70():
    dest = BASE_DIR / "5_etdd70_eyetracking"
    dest.mkdir(parents=True, exist_ok=True)

    if already_have_enough(dest, min_files=len(ZENODO_FILES) + 5):
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
        if target.exists() and not FORCE_REDOWNLOAD:
            print(f"\n[skip] {target} already downloaded — skipping.")
            ok = True
        else:
            ok = download_file(file_url, target, headers=browser_headers)
        if ok and fname.lower().endswith(".zip"):
            already_extracted = count_files(dest) > len(ZENODO_FILES) + 2
            if not (already_extracted and not FORCE_REDOWNLOAD):
                unzip_file(target, dest)

    return dest


# -------------------------------------------------------------
# INSPECTION FUNCTIONS
# -------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TABULAR_EXTS = {".csv", ".tsv"}


def inspect_folder(folder: Path, label: str):
    print(f"\n{'=' * 70}\nDATASET: {label}\nPath: {folder}\n{'=' * 70}")

    if not folder.exists():
        print("  [!] Folder does not exist - download may have failed.")
        return

    all_files = list(folder.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    print(f"  Total files found: {len(files_only)}")

    if len(files_only) <= 2:
        zips_present = list(folder.rglob("*.zip")) + list(folder.rglob("*.rar"))
        if zips_present:
            print(f"  [!] Only {len(files_only)} file(s) found but archives exist: "
                  f"{[z.name for z in zips_present]}")
            print("      Extraction likely failed. Check passwords / install unrar.")

    for yaml_file in folder.rglob("data.yaml"):
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


# -------------------------------------------------------------
# ORGANIZE FUNCTIONS (from friend's prepare_dataset.py, merged in)
# -------------------------------------------------------------

import shutil as _shutil


def organize_dysgraphia_images(source_dir: Path, output_dir: str):
    """Organize Mendeley dysgraphia dataset into class folders (friend's logic, unchanged)."""
    target_dir = os.path.join(output_dir, 'dysgraphia')
    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                folder_name = os.path.basename(root).lower().replace(' ', '_')
                class_dir = os.path.join(target_dir, folder_name)
                os.makedirs(class_dir, exist_ok=True)
                _shutil.copy2(os.path.join(root, f), os.path.join(class_dir, f))

    print(f"  Organized images into {target_dir}")


def organize_dyslexia_images(source_dir: Path, output_dir: str, target_name: str = 'dyslexia_handwriting'):
    """Organize a Kaggle handwriting dataset into class folders (friend's logic, unchanged)."""
    target_dir = os.path.join(output_dir, target_name)
    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                rel_path = os.path.relpath(root, source_dir)
                folder_name = rel_path.replace(os.sep, '_').lower().replace(' ', '_')
                class_dir = os.path.join(target_dir, folder_name)
                os.makedirs(class_dir, exist_ok=True)
                _shutil.copy2(os.path.join(root, f), os.path.join(class_dir, f))

    print(f"  Organized images into {target_dir}")


def organize_tabular(source_dir: Path, output_dir: str):
    """Copy Rello tabular CSVs (Dyt-tablet.csv, Dyt-desktop.csv) as-is."""
    target_dir = os.path.join(output_dir, 'tabular')
    os.makedirs(target_dir, exist_ok=True)

    copied = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                _shutil.copy2(os.path.join(root, f), os.path.join(target_dir, f))
                copied.append(f)

    if not copied:
        print(f"  WARNING: no CSV files found under {source_dir}")
    else:
        print(f"  Copied {copied} into {target_dir}")


def organize_eyetracking(source_dir: Path, output_dir: str):
    """Copy ETDD70 stimulus images + the subject-level label CSV as-is."""
    target_dir = os.path.join(output_dir, 'eyetracking')
    os.makedirs(target_dir, exist_ok=True)

    n_images = 0
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                rel_path = os.path.relpath(root, source_dir)
                folder_name = rel_path.replace(os.sep, '_').lower().replace(' ', '_')
                class_dir = os.path.join(target_dir, folder_name)
                os.makedirs(class_dir, exist_ok=True)
                _shutil.copy2(os.path.join(root, f), os.path.join(class_dir, f))
                n_images += 1
            elif f.lower() == 'dyslexia_class_label.csv':
                _shutil.copy2(os.path.join(root, f), os.path.join(target_dir, f))

    print(f"  Organized {n_images} images + label CSV into {target_dir}")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    global FORCE_REDOWNLOAD
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-extract everything.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                        help="Where to write organized (class-foldered) data for training.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download/extract, just organize what's already in dyslexia_datasets/.")
    parser.add_argument("--skip-organize", action="store_true",
                        help="Skip organizing into data/processed/, just download+extract+inspect.")
    args = parser.parse_args()
    FORCE_REDOWNLOAD = args.force

    if not args.skip_download:
        print("Starting download of all 5 dyslexia-related datasets"
              f"{' (forced re-download)' if FORCE_REDOWNLOAD else ''}...\n")
        d1 = get_dataset_1_synthetic()
        d2 = get_dataset_2_luzrello()
        d3 = get_dataset_3_drizasazanitaisa()
        d4 = get_dataset_4_mendeley()
        d5 = get_dataset_5_etdd70()
    else:
        print("Skipping download — using existing dyslexia_datasets/ folders.\n")
        d1 = BASE_DIR / "1_synthetic_handwriting"
        d2 = BASE_DIR / "2_luzrello_tabular"
        d3 = BASE_DIR / "3_drizasazanitaisa_handwriting"
        d4 = BASE_DIR / "4_mendeley_dysgraphia"
        d5 = BASE_DIR / "5_etdd70_eyetracking"

    print("\n\nStarting inspection...")
    inspect_folder(d1, "1. Synthetic Dyslexia Handwriting Dataset (YOLO)")
    inspect_folder(d2, "2. Rello et al. / luzrello Dyslexia (Tabular)")
    inspect_folder(d3, "3. Dyslexia Handwriting Dataset (Normal/Reversal/Corrected)")
    inspect_folder(d4, "4. Potential Dysgraphia Handwriting Dataset (Mendeley)")
    inspect_folder(d5, "5. ETDD70 Eye-Tracking Dyslexia Dataset (Zenodo)")

    if not args.skip_organize:
        print(f"\n\nOrganizing into {args.output}/ for training...")
        os.makedirs(args.output, exist_ok=True)

        print("\n[1_synthetic]")
        organize_dyslexia_images(d1, args.output, target_name="dyslexia_synthetic")
        print("\n[2_luzrello]")
        organize_tabular(d2, args.output)
        print("\n[3_drizasazanitaisa]")
        organize_dyslexia_images(d3, args.output, target_name="dyslexia_handwriting")
        print("\n[4_mendeley]")
        organize_dysgraphia_images(d4, args.output)
        print("\n[5_etdd70]")
        organize_eyetracking(d5, args.output)

    print("\n\nDone.")


if __name__ == "__main__":
    main()