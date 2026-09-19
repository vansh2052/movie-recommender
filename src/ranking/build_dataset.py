"""Build the LightGBM ranker's training table: Stage 1 retrieval candidates
generated from each user's TRAIN-only history, labeled by whether the
candidate is that user's VAL-split positive, with features computed from
train only (no leakage — see docs/DECISIONS.md)."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.ranking.features import build_candidate_table
from src.retrieval.features import build_item_vocab, build_user_vocab
from src.retrieval.index import build_item_index, load_towers


def load_splits():
    processed_dir = Path(CONFIG["paths"]["processed_dir"])
    df = pd.read_parquet(processed_dir / "interactions.parquet")
    movies = pd.read_parquet(processed_dir / "movies.parquet")
    users = pd.read_parquet(processed_dir / "users.parquet")
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    return train, val, test, movies, users


def build_relevant(df: pd.DataFrame) -> dict:
    out = defaultdict(set)
    for row in df.itertuples():
        out[row.user_id].add(row.movie_id)
    return dict(out)


def build_history(*dfs) -> dict:
    hist = defaultdict(set)
    for df in dfs:
        for row in df.itertuples():
            hist[row.user_id].add(row.movie_id)
    return dict(hist)


def run():
    train, val, test, movies, users = load_splits()
    item_vocab = build_item_vocab(movies, CONFIG["retrieval"]["year_bucket_size"])
    user_vocab = build_user_vocab(train, users, movies)

    user_tower, item_tower, _cfg = load_towers()
    index, _item_emb = build_item_index(item_tower, item_vocab)

    val_relevant = build_relevant(val)
    history_for_val = build_history(train)
    user_ids = list(val_relevant.keys())

    k = CONFIG["retrieval"]["top_k"]
    table, groups, _results = build_candidate_table(
        user_ids, val_relevant, history_for_val, train, movies, item_vocab, user_vocab, user_tower, index, k
    )

    processed_dir = Path(CONFIG["paths"]["processed_dir"])
    table_path = processed_dir / "ranker_train_table.parquet"
    groups_path = processed_dir / "ranker_train_groups.npy"
    table.to_parquet(table_path, index=False)
    np.save(groups_path, np.array(groups, dtype="int64"))

    n_positives = int(table["label"].sum())
    print(f"Built ranker training table: {len(table)} rows, {len(groups)} groups (users), {n_positives} positive labels")
    print(f"Saved to {table_path} and {groups_path}")


if __name__ == "__main__":
    run()
