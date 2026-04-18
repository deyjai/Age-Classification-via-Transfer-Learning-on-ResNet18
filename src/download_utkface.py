"""Download and prepare the UTKFace dataset for training.

This script supports:
1) Kaggle download (recommended): `kaggle datasets download -d jangedoo/utkface-new`
2) Direct URL download of an archive file.

After download/extraction, it creates a flat folder of UTKFace images that can be
passed to `train.py --utkface-root <prepared_dir>`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
KAGGLE_DATASET = "jangedoo/utkface-new"


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    """Extract a zip/tar archive into a directory and return that directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
        return extract_dir

    raise ValueError(f"Unsupported archive format: {archive_path}")


def is_utkface_filename(path: Path) -> bool:
    """Basic UTKFace filename validation: [age]_[gender]_[race]_[timestamp].jpg."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    stem_parts = path.stem.split("_")
    return len(stem_parts) >= 4 and stem_parts[0].isdigit()


def collect_utkface_images(source_dir: Path, output_dir: Path) -> int:
    """Copy valid UTKFace images recursively into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for path in source_dir.rglob("*"):
        if not path.is_file() or not is_utkface_filename(path):
            continue
        dst = output_dir / path.name
        if dst.exists():
            continue
        shutil.copy2(path, dst)
        copied += 1

    return copied


def download_from_kaggle(work_dir: Path, kaggle_dataset: str = KAGGLE_DATASET) -> Path:
    """Download UTKFace archive via Kaggle CLI and return archive path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        kaggle_dataset,
        "-p",
        str(work_dir),
    ]
    subprocess.run(cmd, check=True)

    archives = sorted(work_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(
            "No zip archive found after Kaggle download. "
            "Ensure Kaggle CLI is installed and authenticated."
        )
    return archives[-1]


def download_from_url(url: str, work_dir: Path) -> Path:
    """Download an archive from a direct URL and return local archive path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_path = work_dir / Path(url).name
    print(f"Downloading: {url}")
    urlretrieve(url, archive_path)
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare UTKFace dataset")
    parser.add_argument(
        "--output-dir",
        default="datasets/UTKFace",
        help="Directory where prepared UTKFace images will be written.",
    )
    parser.add_argument(
        "--work-dir",
        default="datasets/_downloads",
        help="Temporary directory for downloaded archives and extraction.",
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "url"],
        default="kaggle",
        help="Download source. Kaggle is recommended.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Direct archive URL to use when --source url.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default=KAGGLE_DATASET,
        help="Kaggle dataset identifier used when --source kaggle.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before writing prepared images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    extract_dir = work_dir / "extracted"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    if args.source == "kaggle":
        archive_path = download_from_kaggle(work_dir=work_dir, kaggle_dataset=args.kaggle_dataset)
    else:
        if not args.url:
            raise ValueError("--url is required when --source url")
        archive_path = download_from_url(args.url, work_dir=work_dir)

    print(f"Archive ready: {archive_path}")
    extracted_root = extract_archive(archive_path, extract_dir)
    copied = collect_utkface_images(extracted_root, output_dir)

    print(f"Prepared UTKFace image directory: {output_dir}")
    print(f"Copied {copied} images.")
    print(f"Use with training: python src/train.py --utkface-root {output_dir}")


if __name__ == "__main__":
    main()
