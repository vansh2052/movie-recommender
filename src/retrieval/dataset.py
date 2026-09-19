"""Torch Dataset over train-split positive (user, item) pairs, used with
in-batch negative training: side features (age, genres, etc.) are looked up
by index inside the training loop, not stored per-example here."""

import pandas as pd
from torch.utils.data import Dataset


class RetrievalDataset(Dataset):
    def __init__(self, train_df: pd.DataFrame, user_vocab: dict, item_vocab: dict):
        self.user_idx = train_df["user_id"].map(user_vocab["user_id_to_idx"]).values
        self.item_idx = train_df["movie_id"].map(item_vocab["movie_id_to_idx"]).values

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, i: int):
        return self.user_idx[i], self.item_idx[i]
