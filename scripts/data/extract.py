"""Archive extraction: zip/rar, plain and password-protected.

Windows-safe: locates 7z/unrar/unar via PATH or standard install dirs,
and tries pure-Python zipfile first for encrypted zips (ZipCrypto).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from scripts.common import is_junk, count_files


def _find_tool(name: str):
    """Cross-platform tool lookup (Windows-aware for 7z)."""
    found = shutil.which(name)
    if found:
        return found
    if name == "7z":
        candidates = [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\7-Zip\7z.exe"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return None


def unzip_file(zip_path: Path, dest_folder: Path) -> bool:
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


def unzip_with_password(zip_path: Path, dest_folder: Path, password: str) -> bool:
    """Extract an encrypted zip: zipfile (ZipCrypto) -> 7z (AES) -> unzip CLI."""
    if not zip_path.exists():
        return False
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[unzip] Extracting {zip_path.name} with password -> {dest_folder}")

    # 1) Pure-Python attempt (works for ZipCrypto archives)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest_folder, pwd=password.encode())
        print("  Extraction succeeded (zipfile).")
        return True
    except Exception as e:
        print(f"  zipfile failed ({e}); trying external tools...")

    # 2) 7z (also handles AES-encrypted archives)
    seven = _find_tool("7z")
    if seven:
        try:
            subprocess.run(
                [seven, "x", str(zip_path), f"-o{dest_folder}", f"-p{password}", "-y"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  Extraction succeeded (7z).")
            return True
        except subprocess.CalledProcessError:
            pass

    # 3) unzip CLI (Linux/Mac)
    unzip = _find_tool("unzip")
    if unzip:
        try:
            subprocess.run(
                [unzip, "-o", "-P", password, str(zip_path), "-d", str(dest_folder)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  Extraction succeeded (unzip).")
            return True
        except subprocess.CalledProcessError:
            pass

    print("  ERROR: could not extract. Install 7-Zip for AES-encrypted zips.")
    return False


def extract_rar(rar_path: Path, dest_folder: Path, password: str) -> bool:
    """Extract a RAR archive using unrar / 7z / unar, whichever exists."""
    if not rar_path.exists():
        return False
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[rar] Extracting {rar_path.name} with password -> {dest_folder}")

    unrar = _find_tool("unrar")
    if unrar:
        try:
            subprocess.run(
                [unrar, "x", "-y", f"-p{password}", str(rar_path), str(dest_folder) + os.sep],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  Extraction succeeded (unrar).")
            return True
        except subprocess.CalledProcessError:
            pass

    seven = _find_tool("7z")
    if seven:
        try:
            subprocess.run(
                [seven, "x", str(rar_path), f"-o{dest_folder}", f"-p{password}", "-y"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  Extraction succeeded (7z).")
            return True
        except subprocess.CalledProcessError:
            pass

    unar = _find_tool("unar")
    if unar:
        try:
            subprocess.run(
                [unar, "-f", "-o", str(dest_folder), "-p", password, str(rar_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  Extraction succeeded (unar).")
            return True
        except subprocess.CalledProcessError:
            pass

    print("  ERROR: No .rar extractor found. Install one of: unrar, 7-Zip (winget install 7zip.7zip)")
    return False


def extract_all_archives(folder: Path, password: str = None, depth: int = 0):
    """Recursively extract every zip/rar under *folder* (nested archives included)."""
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
