"""Train the two-tower retrieval model with in-batch negatives.

For a batch of B (user, positive item) pairs, every other item in the batch
serves as a negative for that user: we compute the full B x B cosine
similarity matrix and apply softmax cross-entropy with the diagonal as the
positive label. Only train-split positives are used, so no val/test
information leaks into the retrieval model.
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.retrieval.dataset import RetrievalDataset
from src.retrieval.features import build_item_vocab, build_user_vocab
from src.retrieval.model import ItemTower, UserTower


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_train_data():
    processed_dir = Path(CONFIG["paths"]["processed_dir"])
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    movies = pd.read_parquet(processed_dir / "movies.parquet")
    users = pd.read_parquet(processed_dir / "users.parquet")
    train_df = interactions[interactions["split"] == "train"]
    return train_df, movies, users


def train() -> Path:
    cfg = CONFIG["retrieval"]
    set_seed(CONFIG["seed"])

    train_df, movies, users = load_train_data()
    item_vocab = build_item_vocab(movies, cfg["year_bucket_size"])
    user_vocab = build_user_vocab(train_df, users, movies)

    dataset = RetrievalDataset(train_df, user_vocab, item_vocab)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["num_workers"],
    )

    user_tower = UserTower(
        user_vocab["n_users"], user_vocab["n_ages"], user_vocab["n_occupations"], user_vocab["n_genders"], user_vocab["n_genres"], cfg
    )
    item_tower = ItemTower(item_vocab["n_items"], item_vocab["n_genres"], item_vocab["n_year_buckets"], cfg)

    user_age_t = torch.tensor(user_vocab["age_idx"])
    user_occ_t = torch.tensor(user_vocab["occupation_idx"])
    user_gender_t = torch.tensor(user_vocab["gender_idx"])
    user_genre_pref_t = torch.tensor(user_vocab["genre_pref_matrix"])
    item_genre_t = torch.tensor(item_vocab["genre_matrix"])
    item_year_t = torch.tensor(item_vocab["year_bucket"])

    optimizer = torch.optim.Adam(
        list(user_tower.parameters()) + list(item_tower.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    temperature = cfg["temperature"]

    for epoch in range(cfg["epochs"]):
        t0 = time.perf_counter()
        total_loss = 0.0
        for user_idx, item_idx in loader:
            user_idx = user_idx.long()
            item_idx = item_idx.long()

            user_emb = user_tower(
                user_idx, user_age_t[user_idx], user_occ_t[user_idx], user_gender_t[user_idx], user_genre_pref_t[user_idx]
            )
            item_emb = item_tower(item_idx, item_genre_t[item_idx], item_year_t[item_idx])

            user_emb = F.normalize(user_emb, dim=-1)
            item_emb = F.normalize(item_emb, dim=-1)

            logits = user_emb @ item_emb.T / temperature
            labels = torch.arange(logits.shape[0])
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"epoch {epoch + 1}/{cfg['epochs']} loss={avg_loss:.4f} ({time.perf_counter() - t0:.1f}s)")

    models_dir = Path(CONFIG["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "two_tower.pt"
    torch.save(
        {
            "user_tower_state": user_tower.state_dict(),
            "item_tower_state": item_tower.state_dict(),
            "user_vocab_sizes": {
                "n_users": user_vocab["n_users"],
                "n_ages": user_vocab["n_ages"],
                "n_occupations": user_vocab["n_occupations"],
                "n_genders": user_vocab["n_genders"],
                "n_genres": user_vocab["n_genres"],
            },
            "item_vocab_sizes": {
                "n_items": item_vocab["n_items"],
                "n_genres": item_vocab["n_genres"],
                "n_year_buckets": item_vocab["n_year_buckets"],
            },
            "config": cfg,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train()
