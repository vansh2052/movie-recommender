"""Build a FAISS index over the trained item tower's embeddings for the
full catalog, and retrieve top-k candidates per user."""

import os

# Must be set before torch and faiss are imported: on macOS, pip-installed
# faiss-cpu and torch each bundle their own copy of libomp.dylib, and
# loading both in one process aborts with "OMP: Error #15" otherwise. This
# is the standard, widely-used workaround for that combination.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

# Import order matters here: importing faiss before torch causes a hard
# segfault (not just the OMP warning above) the first time a torch tensor
# op runs afterwards, on this macOS + faiss-cpu + torch combination.
# Importing torch first avoids it.
import torch
import torch.nn.functional as F
import faiss
import numpy as np

# Running torch's own (already-parallel) tensor ops and then calling into
# FAISS's OpenMP-parallel search in the same process reliably segfaults on
# this macOS + faiss-cpu + torch combination, even with the duplicate-lib
# workaround above. Restricting FAISS to a single thread avoids the
# collision; the catalog here is only ~3,700 items so there's no speed cost.
faiss.omp_set_num_threads(1)

from src.config import CONFIG
from src.retrieval.model import ItemTower, UserTower


def load_towers(checkpoint_path: Path = None):
    checkpoint_path = checkpoint_path or Path(CONFIG["paths"]["models_dir"]) / "two_tower.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    uv = ckpt["user_vocab_sizes"]
    iv = ckpt["item_vocab_sizes"]

    user_tower = UserTower(uv["n_users"], uv["n_ages"], uv["n_occupations"], uv["n_genders"], cfg)
    user_tower.load_state_dict(ckpt["user_tower_state"])
    user_tower.eval()

    item_tower = ItemTower(iv["n_items"], iv["n_genres"], iv["n_year_buckets"], cfg)
    item_tower.load_state_dict(ckpt["item_tower_state"])
    item_tower.eval()

    return user_tower, item_tower, cfg


def build_item_index(item_tower: ItemTower, item_vocab: dict):
    """Embed the full catalog and build an exact (IndexFlatIP) cosine index."""
    with torch.no_grad():
        item_idx_t = torch.arange(item_vocab["n_items"])
        genre_t = torch.tensor(item_vocab["genre_matrix"])
        year_t = torch.tensor(item_vocab["year_bucket"])
        item_emb = item_tower(item_idx_t, genre_t, year_t)
        item_emb = F.normalize(item_emb, dim=-1).numpy().astype("float32")

    index = faiss.IndexFlatIP(item_emb.shape[1])
    index.add(item_emb)
    return index, item_emb


def compute_user_embeddings(user_tower: UserTower, user_vocab: dict, user_ids: list) -> np.ndarray:
    idxs = [user_vocab["user_id_to_idx"][u] for u in user_ids]
    idx_t = torch.tensor(idxs, dtype=torch.long)
    with torch.no_grad():
        age_t = torch.tensor(user_vocab["age_idx"])[idx_t]
        occ_t = torch.tensor(user_vocab["occupation_idx"])[idx_t]
        gender_t = torch.tensor(user_vocab["gender_idx"])[idx_t]
        emb = user_tower(idx_t, age_t, occ_t, gender_t)
        emb = F.normalize(emb, dim=-1).numpy().astype("float32")
    return emb


def retrieve(index: faiss.Index, user_embeddings: np.ndarray, item_vocab: dict, k: int, exclude_items_per_user: list = None):
    """Top-k candidate movie ids (+ similarity scores) per user row in
    ``user_embeddings``, skipping each user's ``exclude_items_per_user`` set
    (e.g. their known history). Requests extra candidates from FAISS so that
    after excluding known items we still end up with k results."""
    idx_to_movie_id = {i: m for m, i in item_vocab["movie_id_to_idx"].items()}

    n_request = k
    if exclude_items_per_user:
        max_exclude = max((len(e) for e in exclude_items_per_user), default=0)
        n_request = min(k + max_exclude, item_vocab["n_items"])

    scores, ids = index.search(user_embeddings, n_request)

    results = []
    for row in range(ids.shape[0]):
        exclude = exclude_items_per_user[row] if exclude_items_per_user else set()
        recs, rec_scores = [], []
        for col in range(ids.shape[1]):
            movie_id = idx_to_movie_id[int(ids[row, col])]
            if movie_id in exclude:
                continue
            recs.append(movie_id)
            rec_scores.append(float(scores[row, col]))
            if len(recs) == k:
                break
        results.append((recs, rec_scores))
    return results
