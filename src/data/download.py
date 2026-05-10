"""Download MovieLens 100k dataset into data/raw/movielens/."""

import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DEST_DIR = RAW_DIR / "movielens"
ZIP_PATH = RAW_DIR / "ml-100k.zip"
REQUIRED_FILES = ["u.data", "u.item"]

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 / total_size, 100)
        print(f"\r  {pct:5.1f}%  ({downloaded // 1024} / {total_size // 1024} KB)", end="", flush=True)

def download(url : str, dest: Path) -> None:
    logger.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
    print()  # newline after progress bar
    logger.info("Saved to %s (%.1f MB)", dest, dest.stat().st_size / 1024 ** 2)

def extract(zip_path: Path, dest_dir: Path) -> None:
    logger.info("Extracting %s → %s", zip_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # ml-100k.zip contains a top-level ml-100k/ folder; strip it
        for member in zf.infolist():
            # member.filename looks like "ml-100k/u.data"
            parts = Path(member.filename).parts
            if len(parts) < 2:
                continue
            relative = Path(*parts[1:])
            target = dest_dir / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member.filename))
    logger.info("Extraction complete")

def verify(dest_dir: Path, required: list[str]) -> None:
    logger.info("Verifying required files in %s", dest_dir)
    missing = [f for f in required if not (dest_dir / f).exists()]
    if missing:
        logger.error("Missing files: %s", missing)
        raise FileNotFoundError(f"Missing files after extraction: {missing}")
    for name in required:
        size = (dest_dir / name).stat().st_size
        logger.info("  ✓ %s (%.1f KB)", name, size / 1024)

def main() -> None:
    if not ZIP_PATH.exists():
        download(MOVIELENS_URL, ZIP_PATH)
    else:
        logger.info("Archive already present, skipping download: %s", ZIP_PATH)

    if not DEST_DIR.exists() or not any(DEST_DIR.iterdir()):
        extract(ZIP_PATH, DEST_DIR)
    else:
        logger.info("Destination already populated, skipping extraction: %s", DEST_DIR)

    verify(DEST_DIR, REQUIRED_FILES)
    logger.info("Dataset ready at %s", DEST_DIR)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        sys.exit(1)
