"""Build the LightGBM ranker's training table (val-labeled candidates).

Usage: python -m scripts.run_ranking_build
"""

from src.ranking import build_dataset


if __name__ == "__main__":
    build_dataset.run()
