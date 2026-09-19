"""Fit and evaluate the popularity and ALS baselines on val and test.

Usage: python -m scripts.run_baselines
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.baselines.popularity import PopularityModel
from src.baselines.als import ALSBaseline
from src.eval.metrics import mean_at_k, recall_at_k, ndcg_at_k, catalog_coverage
from src.eval.report import update_section

np.random.seed(CONFIG["seed"])


def load_splits():
    processed_dir = Path(CONFIG["paths"]["processed_dir"])
    df = pd.read_parquet(processed_dir / "interactions.parquet")
    movies = pd.read_parquet(processed_dir / "movies.parquet")
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    return train, val, test, movies


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


def evaluate_model(recommend_fn, relevant: dict, history: dict, ks: list, catalog_size: int) -> dict:
    max_k = max(ks)
    per_user_recs = {}
    for user_id in relevant:
        exclude = history.get(user_id, set())
        per_user_recs[user_id] = recommend_fn(user_id, exclude, max_k)

    results = {}
    for k in ks:
        results[f"recall@{k}"] = mean_at_k(per_user_recs, relevant, recall_at_k, k)
        results[f"ndcg@{k}"] = mean_at_k(per_user_recs, relevant, ndcg_at_k, k)
    results[f"coverage@{max_k}"] = catalog_coverage(list(per_user_recs.values()), catalog_size)
    results["n_users_evaluated"] = len(relevant)
    return results


def format_results_table(rows: list, ks: list) -> str:
    cols = ["Model", "Split", "n_users"] + [f"Recall@{k}" for k in ks] + [f"NDCG@{k}" for k in ks] + [f"Coverage@{max(ks)}"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, split_name, res in rows:
        cells = [name, split_name, str(res["n_users_evaluated"])]
        cells += [f"{res[f'recall@{k}']:.4f}" for k in ks]
        cells += [f"{res[f'ndcg@{k}']:.4f}" for k in ks]
        cells += [f"{res[f'coverage@{max(ks)}']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    train, val, test, movies = load_splits()
    catalog_size = movies["movie_id"].nunique()
    ks = CONFIG["eval"]["ks"]

    val_relevant = build_relevant(val)
    test_relevant = build_relevant(test)
    history_for_val = build_history(train)
    history_for_test = build_history(train, val)

    print(f"train={len(train)} val={len(val)} test={len(test)} catalog_size={catalog_size}")

    print("Fitting popularity baseline...")
    pop = PopularityModel().fit(train)

    print("Fitting ALS baseline...")
    t0 = time.perf_counter()
    als = ALSBaseline().fit(train)
    print(f"ALS fit in {time.perf_counter() - t0:.1f}s")

    models = {
        "Popularity": lambda u, ex, k: pop.recommend(ex, k),
        "ALS": lambda u, ex, k: als.recommend(u, ex, k),
    }

    rows = []
    for name, recommend_fn in models.items():
        val_res = evaluate_model(recommend_fn, val_relevant, history_for_val, ks, catalog_size)
        test_res = evaluate_model(recommend_fn, test_relevant, history_for_test, ks, catalog_size)
        rows.append((name, "val", val_res))
        rows.append((name, "test", test_res))
        print(f"{name} | val  | {val_res}")
        print(f"{name} | test | {test_res}")

    table = format_results_table(rows, ks)
    results_path = Path(CONFIG["paths"]["reports_dir"]) / "results.md"
    update_section(results_path, "Phase 1: Baselines", table)
    print(f"Wrote results table to {results_path}")


if __name__ == "__main__":
    main()
