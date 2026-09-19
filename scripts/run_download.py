"""Download+extract MovieLens 1M, then run the implicit-feedback conversion
and temporal split, saving processed parquet files.

Usage: python -m scripts.run_download
"""

from src.data.download import download_and_extract
from src.data import preprocess


def main():
    download_and_extract()
    stats = preprocess.run()
    print("Preprocessing stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
