# Results

## Phase 1: Baselines

| Model | Split | n_users | Recall@10 | Recall@500 | NDCG@10 | NDCG@500 | Coverage@500 |
|---|---|---|---|---|---|---|---|
| Popularity | val | 6035 | 0.0456 | 0.5920 | 0.0217 | 0.1031 | 0.3544 |
| Popularity | test | 6037 | 0.0393 | 0.5647 | 0.0193 | 0.0970 | 0.3544 |
| ALS | val | 6035 | 0.0829 | 0.7969 | 0.0394 | 0.1534 | 0.8290 |
| ALS | test | 6037 | 0.0654 | 0.7502 | 0.0329 | 0.1402 | 0.8290 |

## Phase 2: Retrieval (Two-Tower + FAISS)

| Model | Split | n_users | Recall@500 | NDCG@500 | Coverage@500 |
|---|---|---|---|---|---|
| Two-Tower Retrieval | val | 6035 | 0.6800 | 0.1105 | 1.0000 |
| Two-Tower Retrieval | test | 6037 | 0.6404 | 0.1031 | 1.0000 |
