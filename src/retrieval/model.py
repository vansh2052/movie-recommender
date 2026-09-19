"""Two-tower retrieval model: a user tower and an item tower that each
produce an embedding in the same space, trained so a user's embedding is
close (cosine similarity) to the embeddings of movies they liked."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """The genre-preference input is not a lookup embedding: it's the
    user's actual train-history genre-preference vector (see
    src/retrieval/features.py), giving the tower real behavioral signal
    about the user's taste instead of relying solely on the user_id
    embedding to memorize it."""

    def __init__(self, n_users: int, n_ages: int, n_occupations: int, n_genders: int, n_genres: int, cfg: dict):
        super().__init__()
        d = cfg["embedding_dim"]
        self.user_emb = nn.Embedding(n_users, d)
        self.age_emb = nn.Embedding(n_ages, cfg["age_bucket_dim"])
        self.occupation_emb = nn.Embedding(n_occupations, cfg["occupation_dim"])
        self.gender_emb = nn.Embedding(n_genders, cfg["gender_dim"])
        self.genre_proj = nn.Linear(n_genres, cfg["genre_hidden_dim"])

        input_dim = d + cfg["age_bucket_dim"] + cfg["occupation_dim"] + cfg["gender_dim"] + cfg["genre_hidden_dim"]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, cfg["mlp_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["mlp_hidden_dim"], d),
        )

    def forward(self, user_idx, age_idx, occupation_idx, gender_idx, genre_pref):
        x = torch.cat(
            [
                self.user_emb(user_idx),
                self.age_emb(age_idx),
                self.occupation_emb(occupation_idx),
                self.gender_emb(gender_idx),
                F.relu(self.genre_proj(genre_pref)),
            ],
            dim=-1,
        )
        return self.mlp(x)


class ItemTower(nn.Module):
    def __init__(self, n_items: int, n_genres: int, n_year_buckets: int, cfg: dict):
        super().__init__()
        d = cfg["embedding_dim"]
        self.item_emb = nn.Embedding(n_items, d)
        self.genre_proj = nn.Linear(n_genres, cfg["genre_hidden_dim"])
        self.year_emb = nn.Embedding(n_year_buckets, cfg["year_bucket_dim"])

        input_dim = d + cfg["genre_hidden_dim"] + cfg["year_bucket_dim"]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, cfg["mlp_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["mlp_hidden_dim"], d),
        )

    def forward(self, item_idx, genre_multihot, year_idx):
        x = torch.cat(
            [
                self.item_emb(item_idx),
                F.relu(self.genre_proj(genre_multihot)),
                self.year_emb(year_idx),
            ],
            dim=-1,
        )
        return self.mlp(x)


def normalized_embed(tower_output: torch.Tensor) -> torch.Tensor:
    return F.normalize(tower_output, dim=-1)
