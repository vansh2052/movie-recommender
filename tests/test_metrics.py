import math

from src.eval.metrics import recall_at_k, ndcg_at_k, catalog_coverage, mean_at_k


def test_recall_at_k_full_hit():
    assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_recall_at_k_partial_hit():
    # 1 of 2 relevant items found in top-3
    assert recall_at_k([1, 5, 6], {1, 2}, k=3) == 0.5


def test_recall_at_k_no_relevant_items():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_recall_at_k_respects_k():
    # relevant item is at position 4 (0-indexed 3), outside k=3
    assert recall_at_k([1, 2, 3, 4], {4}, k=3) == 0.0
    assert recall_at_k([1, 2, 3, 4], {4}, k=4) == 1.0


def test_ndcg_at_k_perfect_ranking():
    # single relevant item ranked first -> NDCG = 1.0
    assert ndcg_at_k([1, 2, 3], {1}, k=3) == 1.0


def test_ndcg_at_k_no_relevant_items():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_ndcg_at_k_hand_computed():
    # relevant = {2}, recommended = [1, 2, 3] -> hit at rank 2 (0-indexed 1)
    # DCG = 1 / log2(3); IDCG (best case, hit at rank 1) = 1 / log2(2) = 1
    recommended = [1, 2, 3]
    relevant = {2}
    expected = (1.0 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(recommended, relevant, k=3), expected, rel_tol=1e-9)


def test_ndcg_at_k_multiple_relevant_items():
    # relevant = {1, 3}, recommended = [1, 2, 3]
    # DCG = 1/log2(2) + 1/log2(4); IDCG (both relevant items ranked first) = 1/log2(2) + 1/log2(3)
    recommended = [1, 2, 3]
    relevant = {1, 3}
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert math.isclose(ndcg_at_k(recommended, relevant, k=3), dcg / idcg, rel_tol=1e-9)


def test_catalog_coverage_full():
    recs = [[1, 2], [3, 4]]
    assert catalog_coverage(recs, catalog_size=4) == 1.0


def test_catalog_coverage_partial():
    recs = [[1, 2], [1, 2]]
    assert catalog_coverage(recs, catalog_size=4) == 0.5


def test_catalog_coverage_empty_catalog():
    assert catalog_coverage([[1, 2]], catalog_size=0) == 0.0


def test_mean_at_k_skips_users_without_relevant_items():
    per_user_recommended = {1: [10, 20], 2: [30, 40]}
    per_user_relevant = {1: {10}, 2: set()}
    # user 2 has no relevant items and should be excluded from the average
    result = mean_at_k(per_user_recommended, per_user_relevant, recall_at_k, k=2)
    assert result == 1.0
