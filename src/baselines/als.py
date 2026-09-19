"""ALS matrix-factorization baseline using the `implicit` library, trained
on the training split's positive interactions only."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src.config import CONFIG


class ALSBaseline:
    def __init__(self, config: dict = None):
        cfg = (config or CONFIG)["als"]
        self.model = AlternatingLeastSquares(
            factors=cfg["factors"],
            regularization=cfg["regularization"],
            iterations=cfg["iterations"],
            random_state=CONFIG["seed"],
        )
        self.alpha = cfg["alpha"]
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_items = None

    def fit(self, train_df: pd.DataFrame) -> "ALSBaseline":
        user_ids = train_df["user_id"].unique()
        movie_ids = train_df["movie_id"].unique()
        self.user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
        self.idx_to_user_id = {i: u for u, i in self.user_id_to_idx.items()}
        self.movie_id_to_idx = {m: i for i, m in enumerate(movie_ids)}
        self.idx_to_movie_id = {i: m for m, i in self.movie_id_to_idx.items()}

        rows = train_df["user_id"].map(self.user_id_to_idx).values
        cols = train_df["movie_id"].map(self.movie_id_to_idx).values
        data = np.full(len(train_df), self.alpha, dtype=np.float32)

        self.user_items = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(movie_ids)),
        )
        self.model.fit(self.user_items)
        return self

    def recommend(self, user_id: int, exclude_items: set, k: int) -> list:
        """Top-k items for a known user, skipping ``exclude_items``. Returns
        an empty list for users unseen at fit time (cold start is handled
        by the caller, e.g. falling back to popularity)."""
        idx = self.user_id_to_idx.get(user_id)
        if idx is None:
            return []

        # Request extra candidates so that after excluding items the model's
        # own train-only filter didn't know about (e.g. val items, when
        # scoring for test), we still end up with k results.
        n_request = min(k + len(exclude_items), len(self.movie_id_to_idx))
        ids, _scores = self.model.recommend(
            idx,
            self.user_items[idx],
            N=n_request,
            filter_already_liked_items=True,
        )
        out = []
        for item_idx in ids:
            movie_id = self.idx_to_movie_id[int(item_idx)]
            if movie_id in exclude_items:
                continue
            out.append(movie_id)
            if len(out) == k:
                break
        return out
