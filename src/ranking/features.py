"""Build the LightGBM ranker's candidate table: Stage 1 (FAISS) candidates,
labeled by whether they're the target item, with user/item/cross features.

The same `build_candidate_table` is used for both:
  - the ranker's TRAINING data: candidates from train-only history, labeled
    by the user's val-split positive, features computed from train only.
  - the final TEST evaluation: candidates from train+val history, labeled
    by the user's test-split positive, features computed from train+val.
Only the (relevant, history_for_exclude, history_for_features) arguments
differ between the two calls — see docs/DECISIONS.md for why features must
be computed "as of" the split being predicted to avoid leakage.
"""

import numpy as np
import pandas as pd

from src.data.loader import GENRES
from src.retrieval.features import compute_user_genre_prefs
from src.retrieval.index import compute_user_embeddings, retrieve

GENRE_COLS = [f"genre_{g}" for g in GENRES]

FEATURE_COLUMNS = (
    ["user_activity_count", "user_avg_rating", "item_popularity", "item_avg_rating", "item_year"]
    + GENRE_COLS
    + ["genre_overlap", "retrieval_score"]
)


def compute_user_stats_arrays(history_df: pd.DataFrame, ordered_user_ids: list):
    stats = history_df.groupby("user_id").agg(activity_count=("movie_id", "count"), avg_rating=("rating", "mean"))
    stats = stats.reindex(ordered_user_ids)
    activity = stats["activity_count"].fillna(0).values.astype("float32")
    global_avg = float(history_df["rating"].mean())
    avg_rating = stats["avg_rating"].fillna(global_avg).values.astype("float32")
    return activity, avg_rating


def compute_item_stats_arrays(history_df: pd.DataFrame, ordered_movie_ids: list):
    stats = history_df.groupby("movie_id").agg(popularity=("user_id", "count"), avg_rating=("rating", "mean"))
    stats = stats.reindex(ordered_movie_ids)
    popularity = stats["popularity"].fillna(0).values.astype("float32")
    global_avg = float(history_df["rating"].mean())
    avg_rating = stats["avg_rating"].fillna(global_avg).values.astype("float32")
    return popularity, avg_rating


def build_candidate_table(
    user_ids: list,
    relevant: dict,
    history_for_exclude: dict,
    history_for_features: pd.DataFrame,
    movies_df: pd.DataFrame,
    item_vocab: dict,
    user_vocab: dict,
    user_tower,
    index,
    k: int,
):
    """Returns (table, groups, results):
    - table: one row per (user, candidate), columns = FEATURE_COLUMNS +
      ["user_id", "movie_id", "label"].
    - groups: number of candidate rows per user, same order as user_ids
      (LightGBM's per-user grouping).
    - results: the raw retrieve() output (recs, scores) per user, in FAISS
      similarity order, reused by evaluate.py for the retrieval-only top-10.
    """
    ordered_movie_ids = movies_df.sort_values("movie_id")["movie_id"].tolist()
    item_year_arr = movies_df.sort_values("movie_id")["year"].values.astype("float32")
    item_pop_arr, item_avg_arr = compute_item_stats_arrays(history_for_features, ordered_movie_ids)
    item_genre_matrix = item_vocab["genre_matrix"]

    user_activity_arr, user_avg_arr = compute_user_stats_arrays(history_for_features, user_ids)
    user_genre_pref = compute_user_genre_prefs(history_for_features, movies_df, user_ids, GENRE_COLS)

    user_embeddings = compute_user_embeddings(user_tower, user_vocab, user_ids)
    exclude_list = [history_for_exclude.get(u, set()) for u in user_ids]
    results = retrieve(index, user_embeddings, item_vocab, k, exclude_list)

    user_id_col, movie_id_col, label_col, groups = [], [], [], []
    feature_blocks = []

    for i, user_id in enumerate(user_ids):
        recs, scores = results[i]
        groups.append(len(recs))
        if not recs:
            continue

        target_items = relevant.get(user_id, set())
        item_idx_arr = np.array([item_vocab["movie_id_to_idx"][m] for m in recs])
        n = len(recs)

        candidate_genres = item_genre_matrix[item_idx_arr]
        genre_overlap = candidate_genres @ user_genre_pref[i]

        block = np.column_stack(
            [
                np.full(n, user_activity_arr[i], dtype="float32"),
                np.full(n, user_avg_arr[i], dtype="float32"),
                item_pop_arr[item_idx_arr],
                item_avg_arr[item_idx_arr],
                item_year_arr[item_idx_arr],
                candidate_genres,
                genre_overlap,
                np.array(scores, dtype="float32"),
            ]
        )
        feature_blocks.append(block)

        user_id_col.extend([user_id] * n)
        movie_id_col.extend(recs)
        label_col.extend(1 if m in target_items else 0 for m in recs)

    features = np.vstack(feature_blocks) if feature_blocks else np.zeros((0, len(FEATURE_COLUMNS)))
    table = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    table["user_id"] = user_id_col
    table["movie_id"] = movie_id_col
    table["label"] = label_col
    return table, groups, results
