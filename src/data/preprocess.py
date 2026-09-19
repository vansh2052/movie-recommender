"""Implicit-feedback conversion and per-user temporal train/val/test split.

Splitting rule (see docs/DECISIONS.md for the full rationale):
  - Only positive interactions (rating >= threshold) participate in the split.
  - Interactions are ordered by timestamp within each user.
  - >=3 positives: last -> test, second-to-last -> val, rest -> train.
  - exactly 2 positives: last -> test, the other -> train, no val row.
  - 1 positive: -> train only (user never appears in val/test evaluation).
This guarantees a user's val/test items are always chronologically after
everything in that user's train split, so no future information leaks
backward into training or into features computed "as of" an earlier split.
"""

from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.data.loader import load_all


def make_implicit(ratings: pd.DataFrame, threshold: int = None) -> pd.DataFrame:
    threshold = threshold if threshold is not None else CONFIG["data"]["positive_rating_threshold"]
    positives = ratings[ratings["rating"] >= threshold].copy()
    return positives


def temporal_split(
    positives: pd.DataFrame,
    min_positives_for_val: int = None,
    min_positives_for_test: int = None,
) -> pd.DataFrame:
    min_positives_for_val = (
        min_positives_for_val
        if min_positives_for_val is not None
        else CONFIG["split"]["min_positives_for_val"]
    )
    min_positives_for_test = (
        min_positives_for_test
        if min_positives_for_test is not None
        else CONFIG["split"]["min_positives_for_test"]
    )

    df = positives.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort").reset_index(drop=True)

    group_sizes = df.groupby("user_id")["user_id"].transform("size")
    rank_from_end = df.groupby("user_id").cumcount(ascending=False)

    is_test = (group_sizes >= min_positives_for_test) & (rank_from_end == 0)
    is_val = (group_sizes >= min_positives_for_val) & (rank_from_end == 1)

    df["split"] = "train"
    df.loc[is_test, "split"] = "test"
    df.loc[is_val, "split"] = "val"
    return df


def run(raw_dir: Path = None, processed_dir: Path = None) -> dict:
    processed_dir = Path(processed_dir or CONFIG["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies, users = load_all(raw_dir)
    positives = make_implicit(ratings)
    split_df = temporal_split(positives)

    split_df.to_parquet(processed_dir / "interactions.parquet", index=False)
    movies.to_parquet(processed_dir / "movies.parquet", index=False)
    users.to_parquet(processed_dir / "users.parquet", index=False)

    counts = split_df["split"].value_counts().to_dict()
    stats = {
        "n_ratings_raw": len(ratings),
        "n_positives": len(positives),
        "n_users": split_df["user_id"].nunique(),
        "n_movies": movies["movie_id"].nunique(),
        "n_train": counts.get("train", 0),
        "n_val": counts.get("val", 0),
        "n_test": counts.get("test", 0),
    }
    return stats


if __name__ == "__main__":
    stats = run()
    for k, v in stats.items():
        print(f"{k}: {v}")
