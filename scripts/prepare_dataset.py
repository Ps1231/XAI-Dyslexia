"""
Dyslexia Multi-Dataset Downloader + Inspector  (v6 — deep nesting + macOS junk fix)
====================================================================

v6 FIXES:
- Dataset #3: Recursively finds ALL image-containing folders regardless of nesting depth.
  Groups by folder name; skips macOS resource forks (._files).
- Dataset #5: Skips __MACOSX folders and ._files during organization.
"""

import os
import sys
import argparse
import zipfile
import subprocess
import ast
from pathlib import Path
from collections import Counter, defaultdict

import requests
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Folders/names to ignore
SKIP_NAMES = {"__macosx", "__pycache__", ".git", "venv", "env"}


def is_junk(path: Path) -> bool:
    """Skip macOS resource forks and hidden files."""
    name = path.name
    if name.startswith("._"):
        return True
    if name.lower() in SKIP_NAMES:
        return True
    if path.is_dir() and path.name.lower() == "__macosx":
        return True
    return False


def count_files(folder: Path, exclude_zip: bool = True) -> int:
    if not folder.exists():
        return 0
    files = [f for f in folder.rglob("*") if f.is_file()
             and (not exclude_zip or f.suffix.lower() != ".zip")
             and not is_junk(f)]
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
    if not rar_path.exists():
        return False
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[rar] Extracting {rar_path.name} with password -> {dest_folder}")

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
    return False


def extract_all_archives(folder: Path, password: str = None, depth: int = 0):
    if depth > 3:
        return
    archives = list(folder.rglob("*.zip")) + list(folder.rglob("*.rar"))
    for arch in archives:
        if is_junk(arch):
            continue
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

TABULAR_EXTS = {".csv", ".tsv"}


def inspect_folder(folder: Path, label: str):
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


# -------------------------------------------------------------
# ORGANIZE FUNCTIONS
# -------------------------------------------------------------
import shutil as _shutil


def organize_yolo_dyslexia(source_dir: Path, output_dir: str, target_name: str = 'dyslexia_synthetic'):
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
            _shutil.copy2(str(img_file), os.path.join(dest, img_file.name))
            if is_dyslexic:
                n_dyslexic += 1
            else:
                n_normal += 1

    print(f"  Binary split: {n_normal} normal, {n_dyslexic} dyslexic")

    # Fallback: if normal is empty, use 3-class majority vote
    if n_normal == 0 and n_dyslexic > 0:
        print("  WARNING: no purely 'normal' images found (mixed-letter dataset).")
        print("  Falling back to 3-class majority-vote organization...")

        # Clean binary folders
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

                if vote:
                    winner = vote.most_common(1)[0][0]
                else:
                    winner = 0

                dest = class_dirs.get(winner, class_dirs[0])
                _shutil.copy2(str(img_file), os.path.join(dest, img_file.name))

        for idx, cdir in class_dirs.items():
            n = len([f for f in os.listdir(cdir) if os.path.isfile(os.path.join(cdir, f))])
            print(f"    Class '{names[idx]}': {n} images")
    else:
        print(f"  Organized YOLO images -> {target_dir}")


def organize_dyslexia_images(source_dir: Path, output_dir: str, target_name: str = 'dyslexia_handwriting'):
    """Organize a handwriting dataset into class folders.

    Recursively finds all image files, groups them by their immediate parent
    folder name. Handles arbitrary nesting depth (e.g. gambo/Corrected/).
    """
    target_dir = os.path.join(output_dir, target_name)
    os.makedirs(target_dir, exist_ok=True)

    # Find all images recursively, skipping junk
    all_images = [f for f in source_dir.rglob("*")
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS and not is_junk(f)]

    if not all_images:
        print(f"  WARNING: no images found under {source_dir}")
        return

    # Group by immediate parent folder name
    class_images = defaultdict(list)
    for img in all_images:
        class_name = img.parent.name.lower().replace(' ', '_')
        class_images[class_name].append(img)

    # If all images share the same parent (wrapper folder like 'gambo'),
    # look one level deeper by using the parent's parent
    if len(class_images) == 1:
        wrapper_name = list(class_images.keys())[0]
        print(f"  Detected wrapper folder '{wrapper_name}' — looking deeper...")
        class_images = defaultdict(list)
        for img in all_images:
            # Use grandparent if parent is wrapper, else parent
            if img.parent.name.lower().replace(' ', '_') == wrapper_name:
                gp = img.parent.parent
                if gp and gp != source_dir:
                    class_name = gp.name.lower().replace(' ', '_')
                else:
                    class_name = img.parent.name.lower().replace(' ', '_')
            else:
                class_name = img.parent.name.lower().replace(' ', '_')
            class_images[class_name].append(img)

    # Copy images to class folders
    for class_name, imgs in sorted(class_images.items()):
        class_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for img in imgs:
            _shutil.copy2(str(img), os.path.join(class_dir, img.name))
        print(f"  Class '{class_name}': {len(imgs)} images")

    print(f"  Organized images into {target_dir}")


def organize_dysgraphia_images(source_dir: Path, output_dir: str):
    """Organize Mendeley dysgraphia dataset into class folders."""
    target_dir = os.path.join(output_dir, 'dysgraphia')
    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                if f.startswith("._"):
                    continue
                folder_name = os.path.basename(root).lower().replace(' ', '_')
                class_dir = os.path.join(target_dir, folder_name)
                os.makedirs(class_dir, exist_ok=True)
                _shutil.copy2(os.path.join(root, f), os.path.join(class_dir, f))

    print(f"  Organized images into {target_dir}")


def organize_tabular(source_dir: Path, output_dir: str):
    """Copy Rello tabular CSVs as-is."""
    target_dir = os.path.join(output_dir, 'tabular')
    os.makedirs(target_dir, exist_ok=True)

    copied = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith('.csv') and not f.startswith("._"):
                _shutil.copy2(os.path.join(root, f), os.path.join(target_dir, f))
                copied.append(f)

    if not copied:
        print(f"  WARNING: no CSV files found under {source_dir}")
    else:
        print(f"  Copied {copied} into {target_dir}")


def organize_eyetracking(source_dir: Path, output_dir: str):
    """Copy ETDD70 label CSV and gaze/fixation data. Skip stimulus images & macOS junk."""
    target_dir = os.path.join(output_dir, 'eyetracking')
    os.makedirs(target_dir, exist_ok=True)

    n_csv = 0
    for root, dirs, files in os.walk(source_dir):
        if os.path.basename(root).lower() == "__macosx":
            continue
        for f in files:
            if f.startswith("._"):
                continue
            if f.lower() == 'dyslexia_class_label.csv':
                _shutil.copy2(os.path.join(root, f), os.path.join(target_dir, f))
                n_csv += 1
            elif f.lower().endswith('.csv'):
                _shutil.copy2(os.path.join(root, f), os.path.join(target_dir, f))
                n_csv += 1

    print(f"  Copied {n_csv} CSV(s) into {target_dir}")
    print("  NOTE: ETDD70 stimulus images are NOT for image classification.")
    print("        Build a gaze-feature pipeline (fixation duration, saccade length, etc.)")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    global FORCE_REDOWNLOAD
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-extract everything.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                        help="Where to write organized data for training.")
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

        print("\n[1_synthetic — YOLO label parsing]")
        organize_yolo_dyslexia(d1, args.output, target_name="dyslexia_synthetic")

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