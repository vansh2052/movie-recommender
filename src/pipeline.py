"""End-to-end inference: load every trained artifact once, then serve
recommend(user_id, k) and get_history(user_id, k) — used by both the
FastAPI app (src/api/main.py) and the Streamlit demo (app_streamlit.py).

Known users go through the full retrieval + ranking pipeline, using
train+val as "history" (the same "as of now" convention used for test-time
evaluation in Phase 3 — val has already happened by serving time). Unknown
user ids fall back to popularity, never an error (see docs/DECISIONS.md,
"Cold start").
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.popularity import PopularityModel
from src.config import CONFIG
from src.ranking.build_dataset import build_history
from src.ranking.features import FEATURE_COLUMNS, build_candidate_table
from src.ranking.evaluate import load_ranker
from src.retrieval.features import build_item_vocab, build_user_vocab
from src.retrieval.index import build_item_index, load_towers


class RecommenderPipeline:
    def __init__(self):
        processed_dir = Path(CONFIG["paths"]["processed_dir"])
        interactions = pd.read_parquet(processed_dir / "interactions.parquet")
        self.movies = pd.read_parquet(processed_dir / "movies.parquet")
        users = pd.read_parquet(processed_dir / "users.parquet")

        train = interactions[interactions["split"] == "train"]
        val = interactions[interactions["split"] == "val"]
        test = interactions[interactions["split"] == "test"]

        # "As of now": everything that's actually happened, matching the
        # train+val convention used for test-time evaluation in Phase 3.
        self.history_for_features = pd.concat([train, val], ignore_index=True)
        self.known_history = build_history(train, val)

        # Full history (including test) is fine for the /history DISPLAY
        # endpoint — it's just showing what a user already liked, not an
        # input to any model.
        self._history_lookup = (
            pd.concat([train, val, test])
            .sort_values("timestamp", ascending=False)
            .groupby("user_id")["movie_id"]
            .apply(list)
            .to_dict()
        )

        self.item_vocab = build_item_vocab(self.movies, CONFIG["retrieval"]["year_bucket_size"])
        self.user_vocab = build_user_vocab(train, users, self.movies)

        self.user_tower, item_tower, _cfg = load_towers()
        self.index, _item_emb = build_item_index(item_tower, self.item_vocab)

        self.ranker = load_ranker()
        self.popularity = PopularityModel().fit(self.history_for_features)

        self.movies_by_id = self.movies.set_index("movie_id")
        self.retrieval_k = CONFIG["retrieval"]["top_k"]

    def is_known_user(self, user_id: int) -> bool:
        return user_id in self.user_vocab["user_id_to_idx"]

    def recommend(self, user_id: int, k: int = 10) -> list:
        if self.is_known_user(user_id):
            recs = self._recommend_known(user_id, k)
            if recs:
                return recs
        return self._recommend_cold_start(k)

    def _recommend_known(self, user_id: int, k: int) -> list:
        table, groups, _results = build_candidate_table(
            [user_id],
            relevant={},
            history_for_exclude=self.known_history,
            history_for_features=self.history_for_features,
            movies_df=self.movies,
            item_vocab=self.item_vocab,
            user_vocab=self.user_vocab,
            user_tower=self.user_tower,
            index=self.index,
            k=self.retrieval_k,
        )
        if groups[0] == 0:
            return []

        scores = self.ranker.predict(table[FEATURE_COLUMNS])
        order = np.argsort(-scores)[:k]
        movie_ids = table["movie_id"].values[order]
        top_scores = scores[order]
        return self._format(movie_ids, top_scores)

    def _recommend_cold_start(self, k: int) -> list:
        movie_ids = self.popularity.recommend(exclude_items=set(), k=k)
        # recommend() already returns items most-popular-first; turn that
        # position into a descending score so "higher is better" holds the
        # same way it does for the ranker's scores above.
        scores = [float(len(movie_ids) - i) for i in range(len(movie_ids))]
        return self._format(movie_ids, scores)

    def get_history(self, user_id: int, k: int = 20) -> list:
        movie_ids = self._history_lookup.get(user_id, [])[:k]
        return self._format(movie_ids, scores=None)

    def _format(self, movie_ids, scores) -> list:
        out = []
        for i, movie_id in enumerate(movie_ids):
            row = self.movies_by_id.loc[int(movie_id)]
            out.append(
                {
                    "movie_id": int(movie_id),
                    "title": row["title"],
                    "genres": row["genres"],
                    "score": float(scores[i]) if scores is not None else None,
                }
            )
        return out
