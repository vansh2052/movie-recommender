"""Evaluate the trained two-tower + FAISS retrieval pipeline: Recall@500 /
NDCG@500 / coverage@500 on val and test, reported alongside the Phase 1
baselines in reports/results.md."""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.eval.metrics import mean_at_k, recall_at_k, ndcg_at_k, catalog_coverage
from src.eval.report import update_section, format_metrics_table
from src.retrieval.features import build_item_vocab, build_user_vocab
from src.retrieval.index import build_item_index, compute_user_embeddings, load_towers, retrieve


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


def evaluate_split(user_tower, index, item_vocab, user_vocab, relevant: dict, history: dict, k: int) -> dict:
    user_ids = list(relevant.keys())
    user_embeddings = compute_user_embeddings(user_tower, user_vocab, user_ids)
    exclude_list = [history.get(u, set()) for u in user_ids]

    results = retrieve(index, user_embeddings, item_vocab, k, exclude_list)
    per_user_recs = {u: recs for u, (recs, _scores) in zip(user_ids, results)}

    return {
        f"recall@{k}": mean_at_k(per_user_recs, relevant, recall_at_k, k),
        f"ndcg@{k}": mean_at_k(per_user_recs, relevant, ndcg_at_k, k),
        f"coverage@{k}": catalog_coverage(list(per_user_recs.values()), item_vocab["n_items"]),
        "n_users_evaluated": len(relevant),
    }


def main():
    train, val, test, movies, users = load_splits()
    item_vocab = build_item_vocab(movies, CONFIG["retrieval"]["year_bucket_size"])
    user_vocab = build_user_vocab(train, users, movies)

    user_tower, item_tower, _cfg = load_towers()
    index, _item_emb = build_item_index(item_tower, item_vocab)

    k = CONFIG["retrieval"]["top_k"]

    val_relevant = build_relevant(val)
    test_relevant = build_relevant(test)
    history_for_val = build_history(train)
    history_for_test = build_history(train, val)

    val_res = evaluate_split(user_tower, index, item_vocab, user_vocab, val_relevant, history_for_val, k)
    test_res = evaluate_split(user_tower, index, item_vocab, user_vocab, test_relevant, history_for_test, k)

    print(f"Two-Tower Retrieval | val  | {val_res}")
    print(f"Two-Tower Retrieval | test | {test_res}")

    rows = [("Two-Tower Retrieval", "val", val_res), ("Two-Tower Retrieval", "test", test_res)]
    table = format_metrics_table(rows, [k])
    results_path = Path(CONFIG["paths"]["reports_dir"]) / "results.md"
    update_section(results_path, "Phase 2: Retrieval (Two-Tower + FAISS)", table)
    print(f"Wrote results table to {results_path}")


if __name__ == "__main__":
    main()
