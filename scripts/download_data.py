#!/usr/bin/env python3
"""
Download UTKFace dataset for age prediction training.

UTKFace is a large-scale face dataset with age, gender, and ethnicity annotations.
Images are named in format: [age]_[gender]_[race]_[date&time].jpg
"""

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

# Dataset sources (mirrors)
DATASET_SOURCES = {
    "utkface_part1": {
        "url": "https://drive.usercontent.google.com/download?id=0BxYys69jI14kYVM3aVhKS1VhRUk&export=download&confirm=t",
        "filename": "UTKFace_part1.tar.gz",
        "type": "tar.gz",
    },
    "utkface_part2": {
        "url": "https://drive.usercontent.google.com/download?id=0BxYys69jI14kSVdWWllDMWhnN2c&export=download&confirm=t",
        "filename": "UTKFace_part2.tar.gz",
        "type": "tar.gz",
    },
    "utkface_part3": {
        "url": "https://drive.usercontent.google.com/download?id=0BxYys69jI14kU0I3YUNlUjBPb1E&export=download&confirm=t",
        "filename": "UTKFace_part3.tar.gz",
        "type": "tar.gz",
    },
}


def download_progress(block_num, block_size, total_size):
    """Display download progress."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
    else:
        downloaded = block_num * block_size
        mb_downloaded = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded: {mb_downloaded:.1f} MB")
    sys.stdout.flush()


def download_file(url: str, dest_path: Path, desc: str = "file") -> bool:
    """Download a file with progress indication."""
    print(f"Downloading {desc}...")
    try:
        urlretrieve(url, dest_path, download_progress)
        print()  # New line after progress
        return True
    except URLError as e:
        print(f"\nFailed to download: {e}")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract tar.gz or zip archive."""
    print(f"Extracting {archive_path.name}...")
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest_dir)
        elif archive_path.name.endswith(".tar"):
            with tarfile.open(archive_path, "r") as tf:
                tf.extractall(dest_dir)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def flatten_directory(data_dir: Path) -> int:
    """Move all images from subdirectories to the main data directory."""
    moved = 0
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            for img_file in subdir.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    dest = data_dir / img_file.name
                    if not dest.exists():
                        img_file.rename(dest)
                        moved += 1
            # Remove empty subdirectory
            try:
                subdir.rmdir()
            except OSError:
                pass
    return moved


def count_valid_images(data_dir: Path) -> int:
    """Count images that match UTKFace naming format."""
    count = 0
    for img_file in data_dir.glob("*.*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            name = img_file.stem
            parts = name.split("_")
            if len(parts) >= 1:
                try:
                    age = int(parts[0])
                    if 0 <= age <= 120:
                        count += 1
                except ValueError:
                    pass
    return count


def download_kaggle_dataset(data_dir: Path) -> bool:
    """Download from Kaggle (requires kaggle CLI configured)."""
    print("\nAttempting to download from Kaggle...")
    print("Note: This requires the Kaggle CLI to be installed and configured.")
    print("Run: pip install kaggle")
    print("And place your kaggle.json in ~/.kaggle/")

    try:
        import subprocess

        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "jangedoo/utkface-new",
                "-p",
                str(data_dir),
                "--unzip",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Kaggle download successful!")
            return True
        else:
            print(f"Kaggle download failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it with: pip install kaggle")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download UTKFace dataset for age prediction training"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./data",
        help="Output directory for dataset (default: ./data)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["google-drive", "kaggle"],
        default="google-drive",
        help="Download source (default: google-drive)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archive files after extraction",
    )
    args = parser.parse_args()

    # Setup directories
    data_dir = Path(args.output_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UTKFace Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {data_dir}")
    print(f"Source: {args.source}")
    print("=" * 60)

    if args.source == "kaggle":
        download_kaggle_dataset(data_dir)
    else:
        # Download from Google Drive mirrors
        print("\nDownloading UTKFace parts from Google Drive...")
        print("Note: If downloads fail, try the Kaggle source instead.")
        print()

        for _name, info in DATASET_SOURCES.items():
            archive_path = data_dir / info["filename"]

            if archive_path.exists():
                print(f"{info['filename']} already exists, skipping download...")
            else:
                if not download_file(info["url"], archive_path, info["filename"]):
                    continue

            # Extract
            if extract_archive(archive_path, data_dir):
                if not args.keep_archives:
                    archive_path.unlink()
                    print(f"  Removed archive: {info['filename']}")
            else:
                pass

    # Flatten directory structure if needed
    print("\nOrganizing files...")
    moved = flatten_directory(data_dir)
    if moved > 0:
        print(f"  Moved {moved} files from subdirectories")

    # Count valid images
    valid_count = count_valid_images(data_dir)

    print("\n" + "=" * 60)
    if valid_count > 0:
        print(f"SUCCESS! Found {valid_count} valid images in {data_dir}")
        print("\nYou can now train the model with:")
        print(f"  python scripts/train.py --model resnet50 --epochs 10 --data-dir {data_dir}")
    else:
        print("WARNING: No valid UTKFace images found.")
        print("\nAlternative download options:")
        print("1. Download manually from: https://www.kaggle.com/datasets/jangedoo/utkface-new")
        print("2. Download from: https://susanqq.github.io/UTKFace/")
        print(f"3. Extract images to: {data_dir}")
    print("=" * 60)

    return 0 if valid_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
