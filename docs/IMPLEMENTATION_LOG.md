# Implementation Log

A chronological record of how this project was built, in the order it
actually happened. Every entry is written in the same commit as the step it
describes, and every number reported here comes from an actual run of the
relevant script.

## Phase 1 — Data and baselines

### Step 1: Project scaffold

Created the repository skeleton: `src/` package layout (`data`, `eval`,
`baselines`, `retrieval`, `ranking`, `api`), `scripts/` CLI entry points,
`tests/`, `data/raw` and `data/processed` (gitignored), `models/`
(gitignored), `reports/figures/`, `docs/`. Added `config.yaml` as the single
source of paths/seeds/hyperparameters and `src/config.py` to load it.
Environment: Python 3.9.6 (system, arm64) in a project-local `.venv`;
dependencies pinned in `requirements.txt` (PyTorch 2.8, faiss-cpu, LightGBM
4.6, implicit 0.7.3, FastAPI, Streamlit). All imported cleanly on first try —
no libomp workaround was needed on this machine.

### Step 2: Data download + implicit conversion + temporal split

`src/data/download.py` downloads and extracts `ml-1m.zip` from GroupLens into
`data/raw/` (idempotent — skips if already present). `src/data/loader.py`
parses `ratings.dat`/`movies.dat`/`users.dat` (`::` separator, latin-1
encoding), parsing the release year out of each title and expanding genres
into an 18-column multi-hot matrix. `src/data/preprocess.py` implements the
implicit-feedback conversion (rating >= 4) and the per-user temporal split
(see `docs/DECISIONS.md`), saving `data/processed/{interactions,movies,
users}.parquet`.

**How to run:** `make download-data` (or `python -m scripts.run_download`).

**Results (real run on ml-1m):**
- 1,000,209 raw ratings -> 575,281 positive interactions (rating >= 4)
- 6,038 users with at least one positive, 3,883 movies
- Split sizes: 563,209 train / 6,035 val / 6,037 test

**Bug fixed along the way:** the first implementation of `temporal_split`
used `groupby("user_id").apply(...)` with a per-group Python function, which
triggered a pandas 2.x `FutureWarning` about grouping-column behavior
changing. Rewrote it as a fully vectorized version using
`groupby.cumcount(ascending=False)` (rank from the end of each user's
timestamp-sorted history) compared against each group's size — same split
semantics, verified to produce identical counts, no warning, and faster.

### Step 3: Evaluation metrics + unit tests

`src/eval/metrics.py` implements `recall_at_k`, `ndcg_at_k` (binary
relevance) and `catalog_coverage`, plus a `mean_at_k` aggregator. Unit tested
in `tests/test_metrics.py` with hand-computed small examples (perfect
ranking, partial hits, empty relevant sets, multi-item NDCG, coverage).

**How to run:** `make test` (or `python -m pytest tests/ -v`).

### Step 4: EDA

`src/data/eda.py` produces 4 figures into `reports/figures/`: rating
distribution, ratings-per-user (long tail, log scale), ratings-per-movie
(long tail, log scale), genre popularity by rating count.

**How to run:** `make eda`.

**Results:** 4 PNGs saved to `reports/figures/`. Rating distribution
confirms the well-known MovieLens-1M skew toward 4-star ratings (~349k
4-star vs. ~56k 1-star); genre popularity confirms Comedy and Drama dominate
rating volume, Documentary/Film-Noir/Western are the long tail.

### Step 5: Popularity and ALS baselines

`src/baselines/popularity.py` ranks items by train-split positive count.
`src/baselines/als.py` wraps `implicit.als.AlternatingLeastSquares`, fit on a
sparse user-item matrix built from train positives only (confidence =
`alpha` from `config.yaml`). `scripts/run_baselines.py` evaluates both on val
and test at K=10 and K=500 (Recall@K, NDCG@K, catalog coverage), excluding
each user's known history from their own recommendations, and writes the
results table into `reports/results.md` under "Phase 1: Baselines".

**How to run:** `make baselines`.

**Results (real run on ml-1m, see `reports/results.md`):**

| Model | Split | Recall@10 | Recall@500 | NDCG@10 | NDCG@500 | Coverage@500 |
|---|---|---|---|---|---|---|
| Popularity | test | 0.0393 | 0.5647 | 0.0193 | 0.0970 | 0.3544 |
| ALS | test | 0.0654 | 0.7502 | 0.0329 | 0.1402 | 0.8290 |

ALS clearly beats popularity on every metric on held-out test data (e.g.
Recall@10 0.065 vs 0.039, a ~67% relative improvement), which is the sanity
check for personalization actually adding value before building the more
complex retrieval/ranking stages on top.

**Problem encountered + fix:** `implicit`'s ALS raised a `RuntimeWarning`
that OpenBLAS's own 8-thread pool conflicts with implicit's internal
parallelism, which can silently cause severe slowdowns. Fixed by setting
`OPENBLAS_NUM_THREADS=1` at the top of `scripts/run_baselines.py` (must be
set before numpy/scipy/implicit are imported, since OpenBLAS reads it at
library load time) and, defensively, wrapping the `model.fit(...)` call in
`threadpoolctl.threadpool_limits(1, "blas")` inside `ALSBaseline.fit` for
any other entry point that constructs this class.

## Phase 2 — Retrieval (two-tower + FAISS)

### Step 6: Two-tower model, training, FAISS index, evaluation

`src/retrieval/features.py` builds deterministic vocabularies/feature arrays
for users (id, age-bucket, occupation, gender) and the full item catalog (id,
18-dim multi-hot genre vector, release-year bucketed into 10-year buckets).
`src/retrieval/model.py` defines `UserTower` and `ItemTower` (each:
embeddings -> concat -> 2-layer MLP -> shared-dimension output).
`src/retrieval/train.py` trains on **train-split positives only** with
in-batch softmax negatives (full batch similarity matrix, cross-entropy with
the diagonal as the positive, temperature 0.05). `src/retrieval/index.py`
embeds the full catalog, builds a FAISS `IndexFlatIP` (cosine, since vectors
are L2-normalized), and retrieves top-k candidates per user, excluding a
caller-supplied history set. `src/retrieval/evaluate.py` reports Recall@500/
NDCG@500/coverage@500 on val and test into `reports/results.md`.

**How to run:** `make train-retrieval` then `make eval-retrieval`.

**Results (real run on ml-1m, see `reports/results.md`):**

| Model | Split | Recall@500 | NDCG@500 | Coverage@500 |
|---|---|---|---|---|
| Popularity | test | 0.5647 | 0.0970 | 0.3544 |
| ALS | test | 0.7502 | 0.1402 | 0.8290 |
| Two-Tower Retrieval | test | 0.6404 | 0.1031 | 1.0000 |

The two-tower retriever beats the popularity baseline (0.640 vs 0.565) but
does **not** beat ALS (0.640 vs 0.750) on Recall@500, despite tuning epochs
(15 -> 40; loss dropped from 5.43 -> 4.67 but with clearly diminishing
returns per epoch, and Recall@500 only moved 0.625 -> 0.640 for ~2.7x more
training time). It does reach 100% catalog coverage, vs. ALS's 83% and
popularity's 35% — it's willing to recommend the entire catalog rather than
concentrating on a subset. This is a real, expected result for this dataset
size, not a bug — see `docs/DECISIONS.md` for the full explanation, and
`README.md`/Limitations for what it implies about the project.

**Bugs encountered + fixed (both are the well-known macOS `faiss` +
`torch` co-installation problems, not application logic bugs):**
1. **`OMP: Error #15`** on first `import faiss` after `import torch`: both
   packages' pip wheels bundle their own `libomp.dylib`, and loading both in
   one process aborts by default. Fixed with the standard workaround,
   `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`, set at the very top of
   `src/retrieval/index.py` before either library is imported.
2. **Hard segfault** the first time `faiss.Index.search(...)` ran after
   torch had already done real tensor computation (embedding lookups + MLP
   forward) in the same process — the OMP workaround above silences the
   *abort* but not this crash, because it's a genuine thread-pool collision
   between torch's parallel ops and FAISS's own OpenMP-parallel search, not
   just a duplicate-symbol warning. Fixed with `faiss.omp_set_num_threads(1)`
   right after import — with a ~3,700-item catalog, single-threaded exact
   search has no measurable speed cost. Also reordered the module's imports
   to `torch` before `faiss`, which independently avoided an earlier segfault
   during `load_state_dict(...)`. Both fixes are isolated to
   `src/retrieval/index.py` since that's the only module that imports both
   libraries in the same process.
