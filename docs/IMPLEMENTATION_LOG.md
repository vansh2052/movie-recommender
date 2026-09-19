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

### Step 7: Give the user tower real behavioral signal (genre preference)

The retrieval gap vs. ALS above was diagnosed as: `UserTower` only received
`user_id` + demographics, so it had no direct signal about what the user
actually liked — everything had to be learned indirectly through the
`user_id` embedding via the contrastive loss, the same job ALS's per-user
factor does but through a noisier gradient path. Fixed by computing each
user's **train-only** genre-preference vector (the mean of the genre
multi-hot vectors of movies they positively interacted with in train — see
`_compute_user_genre_prefs` in `src/retrieval/features.py`) and feeding it
into `UserTower` through its own projection layer, the same pattern already
used for `ItemTower`'s genre input. Threaded through
`build_user_vocab`/`train.py`/`index.py`'s `compute_user_embeddings`, and
the checkpoint's `user_vocab_sizes` now also stores `n_genres`.

**How to run:** same as Step 6 (`make train-retrieval` then
`make eval-retrieval`) — no new commands, the feature is baked into the
existing pipeline.

**Results (real run, same 40 epochs as Step 6 for a fair comparison):**

| Model | Split | Recall@500 | NDCG@500 |
|---|---|---|---|
| Two-Tower (no genre pref) | test | 0.6404 | 0.1031 |
| Two-Tower (+ genre pref) | test | 0.6507 | 0.1061 |

A real but modest improvement (+0.0103 Recall@500, +0.0030 NDCG@500),
confirming the diagnosis was directionally correct without closing the gap
to ALS (0.7502) — consistent with the dataset-size argument in
`docs/DECISIONS.md`: even with real behavioral signal, a small feed-forward
tower trained with noisy contrastive gradients is still working with less
information per update than ALS's exact alternating least-squares solve.

## Phase 3 — Ranking (LightGBM LambdaRank)

### Step 8: Ranker training data, LambdaRank training, full-pipeline evaluation

`src/ranking/features.py` builds the shared candidate table used for both
training and evaluation: Stage 1 (frozen two-tower + FAISS) candidates for a
set of users, joined with user features (activity count, avg rating,
train-only-computed genre-preference vector, reusing
`compute_user_genre_prefs` from `src/retrieval/features.py`), item features
(popularity, avg rating, genre multi-hot, year), and cross features (genre
overlap, the Stage 1 retrieval similarity score) — vectorized with numpy per
user rather than row-by-row pandas lookups, since a full candidate table is
~3M rows. `src/ranking/build_dataset.py` builds the **training** table:
candidates from **train-only** history, labeled 1 if the candidate is the
user's **val**-split positive, features from **train only**.
`src/ranking/train.py` fits `lightgbm.LGBMRanker(objective="lambdarank")`
grouped by user. `src/ranking/evaluate.py` builds the **test** evaluation
table (candidates + features from **train + val**, labels from **test**),
scores it with the trained ranker, reranks each user's candidates, and
reports Recall@10/NDCG@10 for popularity, ALS (both refit here for a
self-contained comparison), retrieval-only top-10, and the full pipeline.

**How to run:** `make train-ranking` (build + train) then `make eval-ranking`.

**Results (real run on ml-1m, see `reports/results.md`):**

| Model | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|
| Popularity | 0.0393 | 0.0193 | 0.0296 |
| ALS | 0.0654 | 0.0329 | 0.4378 |
| Retrieval-only (top-10) | 0.0318 | 0.0140 | 0.9593 |
| Retrieval + Ranking (full pipeline) | 0.0828 | 0.0423 | 0.5228 |

The training table had 3,017,500 rows across 6,035 user-groups, with 4,179
positive labels (matches the Phase 2 val Recall@500 of 0.6925: 0.6925 x 6035
≈ 4,179 — the val item was retrieved into the candidate set for exactly that
fraction of users, a useful cross-check that the pipeline is wired up
correctly). The full pipeline beats ALS on both metrics (+27% Recall@10,
+29% NDCG@10) despite retrieval alone losing to ALS on Recall@500 — see
`docs/DECISIONS.md` for why retrieval-only top-10 is actually the *worst*
of the four (it's optimized for recall within top-500, not precision at
top-10) and why the two-stage combination still wins.

**Bugs encountered + fixed:**
1. **`LGBMRanker` failed to import** with
   `OSError: ... Library not loaded: @rpath/libomp.dylib` the first time
   `src/ranking/train.py` ran standalone, even though `import lightgbm` had
   worked earlier in the project. Root cause: the earlier working import was
   in a line that also imported `torch` first, which happened to load a
   compatible `libomp.dylib` into the process as a side effect — masking
   that Homebrew's `libomp` was never actually installed on this machine.
   `src/ranking/train.py` has no reason to import `torch`, so this surfaced
   as a hard failure. Fixed properly with `brew install libomp` rather than
   relying on the incidental torch import order.
2. **Harmless `UserWarning`**: `LGBMRanker` was fit on a plain numpy array
   (`table[FEATURE_COLUMNS].values`) but predicted on a pandas DataFrame
   slice (or vice versa) between `train.py` and `evaluate.py`, and
   scikit-learn's validation warned about the feature-name mismatch even
   though the values themselves were correct. Fixed by passing the pandas
   DataFrame (`table[FEATURE_COLUMNS]`) consistently in both places instead
   of converting to `.values`; results were bit-for-bit identical before and
   after, confirming it was purely a warning, not a bug. Also added the
   `OPENBLAS_NUM_THREADS=1` fix (from Step 5) to `src/ranking/evaluate.py`
   since it also refits the ALS baseline.

## Phase 4 — Demo API

### Step 9: FastAPI app, Streamlit demo, cold start

`src/pipeline.py` (`RecommenderPipeline`) loads every artifact exactly once
(parquet files, the two-tower model + FAISS index, the trained ranker, a
popularity model fit on train+val for cold start) and exposes
`recommend(user_id, k)` and `get_history(user_id, k)`, reusing the exact
same `build_candidate_table` from Phase 3 for known users — the API's
recommendations are produced by the identical code path that was evaluated
in Phase 3, not a reimplementation. Known-user history/features use
train+val ("as of now"); an unknown `user_id` falls back to popularity,
never an error. `src/api/main.py` (FastAPI, lifespan-based startup so the
pipeline loads once, not per request) exposes `GET /recommend/{user_id}?k=`
and `GET /user/{user_id}/history?k=`, logging each request's latency via
`time.perf_counter()`. `app_streamlit.py` calls `RecommenderPipeline`
directly (no HTTP round-trip) behind `st.cache_resource`, showing history
next to recommendations side by side.

**How to run:** `make api` (FastAPI on :8000) or `make demo` (Streamlit on
:8501).

**Results (real manual test run):**
- Pipeline loads in 1.84s at startup.
- `GET /recommend/1?k=5` (known user, history = Pocahontas/Hercules/Mulan/
  Bug's Life/Antz — all Disney animated family films) returned Lion King,
  Little Mermaid, Charlotte's Web, Pinocchio, Fantasia — genuinely
  genre-consistent with the user's actual history, a strong qualitative
  sanity check beyond the offline metrics.
- `GET /recommend/999999?k=5` (a user id with zero training history)
  correctly returned `is_cold_start: true` with popularity-based
  recommendations (American Beauty, Star Wars IV/V, Saving Private Ryan,
  Raiders of the Lost Ark) instead of an error.
- `GET /user/999999/history` correctly returned `is_known_user: false` and
  an empty history list.
- Per-request latency (logged, not asserted): 490ms on the very first
  request (cold caches), settling to ~85-125ms on subsequent known-user
  requests; cold-start and history requests are sub-millisecond (no FAISS/
  LightGBM work needed).
- The Streamlit demo was verified in an actual browser: user 1's history
  and recommendations render side by side as designed. Verifying the
  cold-start path visually in the browser hit a transient Chrome-extension
  tooling error unrelated to the app; the same cold-start code path was
  already confirmed correct via the API test above, since both entry points
  call the identical `RecommenderPipeline` methods.

## Phase 5 — Documentation

### Step 10: Reproducibility check, dependency pinning, final README

Pinned `requirements.txt` to the exact versions installed in this project's
`.venv` (via `pip freeze`), rather than the loose `>=` ranges used during
development, for full reproducibility. Re-ran the **entire pipeline from a
clean invocation** of `make all` (download-data through eval-ranking) as a
final end-to-end check.

**Result: every number reproduced bit-for-bit identical** to the numbers
already recorded in `reports/results.md` across all three phases —
including the two-tower training loss curve matching to 4 decimal places
epoch-by-epoch (e.g. final epoch 40 loss = 4.5890 in both runs). This
confirms the fixed seed (`config.yaml -> seed: 42`, applied via
`set_seed()` in both `src/retrieval/train.py` and `src/ranking/train.py`)
genuinely makes the pipeline reproducible end to end, not just "probably
similar." Rewrote `README.md` as a single coherent document (problem
statement, mermaid architecture diagram, dataset description, consolidated
results tables, Limitations & Future Work) instead of the phase-by-phase
appended sections used while building — the appended history itself lives
on unedited in this log and in `docs/DECISIONS.md`. Expanded
`docs/DECISIONS.md`'s Interview Questions section to 29 entries covering
the problem, data, retrieval, ranking, evaluation, leakage, cold start,
scaling, and limitations, per the original project brief's 25-30 target.

**How to run:** `make all`.
