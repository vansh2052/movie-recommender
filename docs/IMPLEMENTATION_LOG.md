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

**Results:** *(filled in after running against the real dataset — see below)*

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

### Step 5: Popularity and ALS baselines

`src/baselines/popularity.py` ranks items by train-split positive count.
`src/baselines/als.py` wraps `implicit.als.AlternatingLeastSquares`, fit on a
sparse user-item matrix built from train positives only (confidence =
`alpha` from `config.yaml`). `scripts/run_baselines.py` evaluates both on val
and test at K=10 and K=500 (Recall@K, NDCG@K, catalog coverage), excluding
each user's known history from their own recommendations, and writes the
results table into `reports/results.md` under "Phase 1: Baselines".

**How to run:** `make baselines`.

**Results:** *(filled in after running against the real dataset — see below)*
