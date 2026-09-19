"""Evaluate the full retrieval+ranking pipeline on TEST, alongside
retrieval-only top-10 and the Phase 1 baselines refit on the same data, into
reports/results.md.

Test-time candidates and features use TRAIN + VAL history (val has,
chronologically, already happened by the time we're predicting test — see
docs/DECISIONS.md), never test itself.
"""

import os

# Must be set before numpy/scipy/implicit are imported: see
# src/baselines/als.py and scripts/run_baselines.py for why.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.als import ALSBaseline
from src.baselines.popularity import PopularityModel
from src.config import CONFIG
from src.eval.metrics import catalog_coverage, mean_at_k, ndcg_at_k, recall_at_k
from src.eval.report import format_metrics_table, update_section
from src.ranking.build_dataset import build_history, build_relevant, load_splits
from src.ranking.features import FEATURE_COLUMNS, build_candidate_table
from src.retrieval.features import build_item_vocab, build_user_vocab
from src.retrieval.index import build_item_index, load_towers


def load_ranker():
    model_path = Path(CONFIG["paths"]["models_dir"]) / "ranker.pkl"
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"]


def evaluate_baseline(recommend_fn, relevant: dict, history: dict, k: int, catalog_size: int) -> dict:
    per_user_recs = {u: recommend_fn(u, history.get(u, set()), k) for u in relevant}
    return {
        f"recall@{k}": mean_at_k(per_user_recs, relevant, recall_at_k, k),
        f"ndcg@{k}": mean_at_k(per_user_recs, relevant, ndcg_at_k, k),
        f"coverage@{k}": catalog_coverage(list(per_user_recs.values()), catalog_size),
        "n_users_evaluated": len(relevant),
    }


def evaluate_recs(per_user_recs: dict, relevant: dict, k: int, catalog_size: int) -> dict:
    return {
        f"recall@{k}": mean_at_k(per_user_recs, relevant, recall_at_k, k),
        f"ndcg@{k}": mean_at_k(per_user_recs, relevant, ndcg_at_k, k),
        f"coverage@{k}": catalog_coverage(list(per_user_recs.values()), catalog_size),
        "n_users_evaluated": len(relevant),
    }


def main():
    train, val, test, movies, users = load_splits()
    catalog_size = movies["movie_id"].nunique()
    top_k = CONFIG["ranking"]["top_k"]
    retrieval_k = CONFIG["retrieval"]["top_k"]

    test_relevant = build_relevant(test)
    history_for_test = build_history(train, val)
    history_for_features = pd.concat([train, val], ignore_index=True)
    user_ids = list(test_relevant.keys())

    # --- Popularity / ALS baselines, refit on train, evaluated on test ---
    print("Fitting popularity and ALS baselines...")
    pop = PopularityModel().fit(train)
    t0 = time.perf_counter()
    als = ALSBaseline().fit(train)
    print(f"ALS fit in {time.perf_counter() - t0:.1f}s")

    pop_res = evaluate_baseline(lambda u, ex, k: pop.recommend(ex, k), test_relevant, history_for_test, top_k, catalog_size)
    als_res = evaluate_baseline(lambda u, ex, k: als.recommend(u, ex, k), test_relevant, history_for_test, top_k, catalog_size)

    # --- Retrieval-only top-k and full retrieval+ranking pipeline ---
    print("Loading retrieval model and building test candidates...")
    item_vocab = build_item_vocab(movies, CONFIG["retrieval"]["year_bucket_size"])
    user_vocab = build_user_vocab(train, users, movies)
    user_tower, item_tower, _cfg = load_towers()
    index, _item_emb = build_item_index(item_tower, item_vocab)

    table, groups, results = build_candidate_table(
        user_ids, test_relevant, history_for_test, history_for_features, movies, item_vocab, user_vocab, user_tower, index, retrieval_k
    )

    print("Scoring candidates with the trained ranker...")
    ranker = load_ranker()
    scores_pred = ranker.predict(table[FEATURE_COLUMNS])

    retrieval_only_recs = {}
    pipeline_recs = {}
    offset = 0
    for i, user_id in enumerate(user_ids):
        n = groups[i]
        recs, _sim_scores = results[i]
        retrieval_only_recs[user_id] = recs[:top_k]
        if n == 0:
            pipeline_recs[user_id] = []
            continue
        user_movie_ids = table["movie_id"].values[offset : offset + n]
        user_scores = scores_pred[offset : offset + n]
        offset += n
        order = np.argsort(-user_scores)
        pipeline_recs[user_id] = list(user_movie_ids[order][:top_k])

    retrieval_only_res = evaluate_recs(retrieval_only_recs, test_relevant, top_k, catalog_size)
    pipeline_res = evaluate_recs(pipeline_recs, test_relevant, top_k, catalog_size)

    print(f"Popularity           | test | {pop_res}")
    print(f"ALS                  | test | {als_res}")
    print(f"Retrieval-only top-{top_k} | test | {retrieval_only_res}")
    print(f"Full pipeline        | test | {pipeline_res}")

    rows = [
        ("Popularity", "test", pop_res),
        ("ALS", "test", als_res),
        (f"Retrieval-only (top-{top_k})", "test", retrieval_only_res),
        ("Retrieval + Ranking (full pipeline)", "test", pipeline_res),
    ]
    table_md = format_metrics_table(rows, [top_k])
    results_path = Path(CONFIG["paths"]["reports_dir"]) / "results.md"
    update_section(results_path, "Phase 3: Ranking (full pipeline)", table_md)
    print(f"Wrote results table to {results_path}")


if __name__ == "__main__":
    main()
