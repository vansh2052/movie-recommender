"""Download and extract the MovieLens 1M dataset into data/raw/."""

import zipfile
from pathlib import Path
import requests

from src.config import CONFIG


def download_and_extract(raw_dir: str = None, url: str = None) -> Path:
    """Download ml-1m.zip (if not already present) and extract it.

    Returns the path to the extracted ``ml-1m`` directory containing
    ratings.dat, movies.dat, users.dat.
    """
    raw_dir = Path(raw_dir or CONFIG["paths"]["raw_dir"])
    url = url or CONFIG["paths"]["ml1m_url"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = raw_dir / "ml-1m"
    ratings_file = extracted_dir / "ratings.dat"
    if ratings_file.exists():
        print(f"Dataset already present at {extracted_dir}, skipping download.")
        return extracted_dir

    zip_path = raw_dir / "ml-1m.zip"
    if not zip_path.exists():
        print(f"Downloading {url} ...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        print(f"Saved to {zip_path}")
    else:
        print(f"Zip already downloaded at {zip_path}, skipping download.")

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    if not ratings_file.exists():
        raise FileNotFoundError(
            f"Extraction finished but {ratings_file} not found. "
            "The archive layout may have changed."
        )

    print(f"Dataset ready at {extracted_dir}")
    return extracted_dir


if __name__ == "__main__":
    download_and_extract()
