"""Non-personalized popularity baseline: rank items by how many positive
interactions they received in the training split."""

import pandas as pd


class PopularityModel:
    def __init__(self):
        self.ranked_items: list = []

    def fit(self, train_df: pd.DataFrame) -> "PopularityModel":
        counts = train_df.groupby("movie_id").size().sort_values(ascending=False)
        self.ranked_items = counts.index.tolist()
        return self

    def recommend(self, exclude_items: set, k: int) -> list:
        """Top-k items by global popularity, skipping anything in
        ``exclude_items`` (typically the items already in the user's
        known history)."""
        out = []
        for item in self.ranked_items:
            if item in exclude_items:
                continue
            out.append(item)
            if len(out) == k:
                break
        return out
