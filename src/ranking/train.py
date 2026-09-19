"""Train the LightGBM LambdaRank ranker on the candidate table built by
build_dataset.py, grouped by user."""

import pickle
import random
from pathlib import Path

import numpy as np
import lightgbm as lgb
import pandas as pd

from src.config import CONFIG
from src.ranking.features import FEATURE_COLUMNS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_training_table():
    processed_dir = Path(CONFIG["paths"]["processed_dir"])
    table = pd.read_parquet(processed_dir / "ranker_train_table.parquet")
    groups = np.load(processed_dir / "ranker_train_groups.npy")
    return table, groups


def train() -> Path:
    cfg = CONFIG["ranking"]
    set_seed(CONFIG["seed"])

    table, groups = load_training_table()
    X = table[FEATURE_COLUMNS]
    y = table["label"].values

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        num_leaves=cfg["num_leaves"],
        learning_rate=cfg["learning_rate"],
        n_estimators=cfg["n_estimators"],
        min_child_samples=cfg["min_child_samples"],
        random_state=CONFIG["seed"],
        verbose=-1,
    )
    ranker.fit(X, y, group=groups)

    models_dir = Path(CONFIG["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "ranker.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": ranker, "feature_columns": FEATURE_COLUMNS}, f)

    print(f"Trained on {len(table)} rows, {len(groups)} groups, {int(y.sum())} positive labels")
    print(f"Saved ranker to {model_path}")
    return model_path


if __name__ == "__main__":
    train()
