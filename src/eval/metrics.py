"""Ranking metrics for implicit-feedback recommendation.

All functions take a single user's ranked list of recommended item ids and
a set of relevant (ground-truth positive) item ids for that user. Binary
relevance is assumed throughout, which matches the implicit-feedback setup
used across this project (rating >= 4 -> relevant, nothing else).
"""

from math import log2


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in the top-k recommendations.

    Returns 0.0 if ``relevant`` is empty (nothing to recall).
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k, binary relevance.

    Returns 0.0 if ``relevant`` is empty.
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    dcg = sum(
        1.0 / log2(rank + 2)  # rank is 0-indexed -> position 1 has log2(2)
        for rank, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 2) for rank in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def catalog_coverage(recommended_lists: list, catalog_size: int) -> float:
    """Fraction of the full catalog that appears across all users'
    recommendation lists at least once."""
    if catalog_size == 0:
        return 0.0
    recommended_items = set()
    for lst in recommended_lists:
        recommended_items.update(lst)
    return len(recommended_items) / catalog_size


def mean_at_k(per_user_recommended: dict, per_user_relevant: dict, metric_fn, k: int) -> float:
    """Average a per-user metric (recall_at_k or ndcg_at_k) over every user
    in ``per_user_relevant`` that has at least one relevant item."""
    scores = []
    for user_id, relevant in per_user_relevant.items():
        if not relevant:
            continue
        recommended = per_user_recommended.get(user_id, [])
        scores.append(metric_fn(recommended, relevant, k))
    return sum(scores) / len(scores) if scores else 0.0
