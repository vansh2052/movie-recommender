"""Deterministic vocabulary / feature-array builders shared by training,
FAISS-index building, and evaluation for the two-tower retrieval model.

Calling these functions again at index/eval time (instead of serializing
the vocab) is safe because they are pure functions of the processed
parquet files, which don't change between training and evaluation within
a run of the pipeline.
"""

import numpy as np
import pandas as pd

from src.data.loader import GENRES

# MovieLens 1M encodes age as one of these 7 bracket codes, not a raw age.
AGE_VALUES = [1, 18, 25, 35, 45, 50, 56]
GENDER_VALUES = ["F", "M"]


def build_item_vocab(movies_df: pd.DataFrame, year_bucket_size: int = 10) -> dict:
    """Full-catalog item vocabulary: every movie gets an index and a feature
    row, regardless of whether it has any training interactions, since the
    FAISS index must be able to retrieve any movie in the catalog."""
    movies_df = movies_df.sort_values("movie_id").reset_index(drop=True)
    movie_ids = movies_df["movie_id"].tolist()
    movie_id_to_idx = {m: i for i, m in enumerate(movie_ids)}

    genre_cols = [f"genre_{g}" for g in GENRES]
    genre_matrix = movies_df[genre_cols].values.astype("float32")

    min_year = int(movies_df["year"].min())
    year_bucket = ((movies_df["year"] - min_year) // year_bucket_size).astype("int64").values
    n_year_buckets = int(year_bucket.max()) + 1

    return {
        "movie_id_to_idx": movie_id_to_idx,
        "genre_matrix": genre_matrix,
        "year_bucket": year_bucket,
        "n_items": len(movie_ids),
        "n_genres": len(genre_cols),
        "n_year_buckets": n_year_buckets,
        "min_year": min_year,
        "year_bucket_size": year_bucket_size,
    }


def build_user_vocab(train_df: pd.DataFrame, users_df: pd.DataFrame) -> dict:
    """User vocabulary built from TRAIN interactions only: every user who
    can appear in val/test evaluation is guaranteed (by the temporal split)
    to have at least one train interaction, so this covers everyone we'll
    ever need to embed."""
    user_ids = sorted(train_df["user_id"].unique().tolist())
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}

    age_to_idx = {a: i for i, a in enumerate(AGE_VALUES)}
    gender_to_idx = {g: i for i, g in enumerate(GENDER_VALUES)}
    occupations = sorted(users_df["occupation"].unique().tolist())
    occupation_to_idx = {o: i for i, o in enumerate(occupations)}

    ordered_users = pd.DataFrame({"user_id": user_ids}).merge(users_df, on="user_id", how="left")
    age_idx = ordered_users["age"].map(age_to_idx).values.astype("int64")
    occupation_idx = ordered_users["occupation"].map(occupation_to_idx).values.astype("int64")
    gender_idx = ordered_users["gender"].map(gender_to_idx).values.astype("int64")

    return {
        "user_id_to_idx": user_id_to_idx,
        "age_idx": age_idx,
        "occupation_idx": occupation_idx,
        "gender_idx": gender_idx,
        "n_users": len(user_ids),
        "n_ages": len(AGE_VALUES),
        "n_occupations": len(occupations),
        "n_genders": len(GENDER_VALUES),
    }
