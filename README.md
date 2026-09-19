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

> **macOS only:** LightGBM's native library requires Homebrew's `libomp`,
> which isn't installed by `pip`: `brew install libomp` once, before
> `make train-ranking`. See `docs/IMPLEMENTATION_LOG.md` (Phase 3, Step 8)
> for why this is easy to miss.

```bash
make setup           # create .venv and install requirements.txt
make download-data   # download MovieLens 1M, build implicit + temporal split
make test            # run unit tests
make eda             # generate EDA figures into reports/figures/
make baselines       # fit + evaluate popularity and ALS baselines
make train-retrieval # train the two-tower retrieval model
make eval-retrieval  # evaluate retrieval Recall@500 vs baselines
make train-ranking   # build ranker training data + train LightGBM LambdaRank
make eval-ranking    # evaluate the full retrieval+ranking pipeline on test
make api             # serve the FastAPI demo at http://localhost:8000
make demo            # launch the Streamlit demo at http://localhost:8501
```

## Phase 1 results (real run on MovieLens 1M)

575,281 positive interactions (rating >= 4) from 6,038 users over 3,883
movies, split per-user by timestamp (see `docs/DECISIONS.md`).

| Model | Split | Recall@10 | Recall@500 | NDCG@10 | NDCG@500 | Coverage@500 |
|---|---|---|---|---|---|---|
| Popularity | test | 0.0393 | 0.5647 | 0.0193 | 0.0970 | 0.3544 |
| ALS | test | 0.0654 | 0.7502 | 0.0329 | 0.1402 | 0.8290 |

Full table (val + test) in `reports/results.md`. EDA figures in
`reports/figures/`.

## Phase 2 results (two-tower retrieval + FAISS)

| Model | Split | Recall@500 | NDCG@500 | Coverage@500 |
|---|---|---|---|---|
| Popularity | test | 0.5647 | 0.0970 | 0.3544 |
| ALS | test | 0.7502 | 0.1402 | 0.8290 |
| Two-Tower Retrieval | test | 0.6507 | 0.1061 | 1.0000 |

The two-tower retriever beats popularity but not ALS on Recall@500 here —
an expected result on a small, dense dataset like MovieLens 1M, not a bug.
See `docs/DECISIONS.md` ("The two-tower retriever underperforms ALS on
Recall@500") for the full explanation and what this implies for a
production system at larger scale. Feeding the user tower a real
train-history genre-preference vector (instead of relying solely on the
learned `user_id` embedding) improved test Recall@500 from 0.6404 to 0.6507
and NDCG@500 from 0.1031 to 0.1061 — a real but modest gain, confirming the
user tower's lack of behavioral signal was part of the gap, not all of it.

## Phase 3 results (full retrieval + ranking pipeline)

| Model | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|
| Popularity | 0.0393 | 0.0193 | 0.0296 |
| ALS | 0.0654 | 0.0329 | 0.4378 |
| Retrieval-only (top-10) | 0.0318 | 0.0140 | 0.9593 |
| **Retrieval + Ranking (full pipeline)** | **0.0828** | **0.0423** | 0.5228 |

The full pipeline beats every baseline, including ALS, on both Recall@10
(+27% relative) and NDCG@10 (+29% relative) — even though the retrieval
stage alone underperforms ALS on Recall@500 (see Phase 2 above). Taking the
retrieval model's raw top-10 directly ("Retrieval-only") is actually the
*worst* of the four, because the two-tower model is trained and evaluated to
rank candidates well within the top-500, not to place the single best item
in the top-10 — that's exactly the job LightGBM's LambdaRank reranking
stage does, and the combination is what makes the two-stage architecture
work. See `docs/DECISIONS.md` for the full breakdown of why retrieval-only
underperforms and why the combined pipeline still wins.

**Why these absolute numbers look low:** Recall@10 = 8% means finding the
one specific movie a user watches next among ~3,700+ unseen movies in just
10 guesses — a random recommender would score ~0.27% at that task, so 8.28%
is ~30x better than chance. This also ranks against the *entire* catalog
rather than a small sampled set of negatives (a common simplification in
published benchmarks that makes their numbers look higher but isn't
comparable here). See `docs/DECISIONS.md` ("Why the absolute Recall@10 (8%)
and NDCG@10 (4%) look low") for the full explanation.

## Phase 4: demo API

`src/pipeline.py` loads every trained artifact once (two-tower model +
FAISS index, LightGBM ranker, a popularity model for cold start) and
reuses the exact `build_candidate_table()` function from Phase 3's
evaluation for known users, so the API returns recommendations produced by
the identical code path that was measured offline — see `docs/DECISIONS.md`
("Serving reuses the exact Phase 3 evaluation code").

- `GET /recommend/{user_id}?k=10` — ranked titles/genres/scores.
- `GET /user/{user_id}/history?k=20` — movies the user has liked.
- Unknown `user_id` → `is_cold_start: true` with popularity-based
  recommendations, never an error.

Real example: user 1's history is entirely Disney/Pixar animated family
films (Pocahontas, Hercules, Mulan, A Bug's Life, Antz); `/recommend/1`
returns Lion King, Little Mermaid, Charlotte's Web, Pinocchio, Fantasia —
genuinely genre-consistent with that history, a useful qualitative check
beyond the offline metrics. An unrecognized id like `/recommend/999999`
correctly falls back to popularity (American Beauty, Star Wars, Saving
Private Ryan, Raiders of the Lost Ark) with `is_cold_start: true`.

Per-request latency is logged server-side via `time.perf_counter()`: ~85-
125ms for a known-user recommendation after warmup, sub-millisecond for
cold-start/history lookups. No Docker, Redis, or cloud deployment — this
runs entirely as a local process.

`app_streamlit.py` is a small optional UI: pick a user id, see their
history next to their recommendations side by side, calling
`RecommenderPipeline` directly rather than going through the API.
