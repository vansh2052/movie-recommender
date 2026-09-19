# Movie Recommender (MovieLens 1M)

A two-stage recommender system: candidate **retrieval** (a PyTorch two-tower
model + FAISS, optimized for recall) followed by **ranking** (LightGBM
LambdaRank, optimized for NDCG), evaluated with a leakage-free per-user
temporal split.

> Status: under active development. This README is filled in incrementally,
> phase by phase — see `docs/IMPLEMENTATION_LOG.md` for the detailed,
> chronological build log and `docs/DECISIONS.md` for design rationale. Full
> problem statement, architecture diagram, and results table land in Phase 5.

## Quickstart

```bash
make setup           # create .venv and install requirements.txt
make download-data   # download MovieLens 1M, build implicit + temporal split
make test            # run unit tests
make eda             # generate EDA figures into reports/figures/
make baselines       # fit + evaluate popularity and ALS baselines
```

See `Makefile` for the remaining pipeline stages (retrieval training/eval,
ranking training/eval, API, Streamlit demo) as they are added.

## Phase 1 results (real run on MovieLens 1M)

575,281 positive interactions (rating >= 4) from 6,038 users over 3,883
movies, split per-user by timestamp (see `docs/DECISIONS.md`).

| Model | Split | Recall@10 | Recall@500 | NDCG@10 | NDCG@500 | Coverage@500 |
|---|---|---|---|---|---|---|
| Popularity | test | 0.0393 | 0.5647 | 0.0193 | 0.0970 | 0.3544 |
| ALS | test | 0.0654 | 0.7502 | 0.0329 | 0.1402 | 0.8290 |

Full table (val + test) in `reports/results.md`. EDA figures in
`reports/figures/`.
