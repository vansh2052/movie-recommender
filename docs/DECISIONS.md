# Design Decisions

This document records every significant design decision made while building
this project, why it was made, what alternatives were considered, and the
trade-offs accepted. It is updated as decisions are made, not written
retroactively.

## Positive threshold: rating >= 4

**Decision:** treat a rating of 4 or 5 (out of 5) as an implicit positive
interaction; everything else (rating 1-3, or no rating at all) is treated as
"not known to be positive."

**Alternatives considered:**
- Treat *any* rating as a positive (i.e. "rated" = "interacted positively").
  Rejected because a 1-star rating is an explicit signal of dislike, not
  interest — folding it into "positive" would teach the model to recommend
  movies people rate poorly.
- Use the raw 1-5 rating as a graded relevance label everywhere. Considered
  for the ranking stage but rejected for the primary evaluation metrics to
  keep Recall@K/NDCG@K binary and directly comparable across all four models
  (popularity, ALS, retrieval, ranker) — binary relevance is also the more
  realistic assumption for a real system, which only observes implicit
  signals (clicks, watches), not five-point ratings.

**Trade-off:** a small number of borderline "meh" 4-star ratings are counted
as positive, and the threshold is a hyperparameter (`config.yaml ->
data.positive_rating_threshold`) rather than a principled cutoff.

## Temporal split, not random

**Decision:** for each user, order their positive interactions by timestamp
and hold out the most recent one for test and the second-most-recent for
validation; everything earlier is training data.

**Alternatives considered:**
- Random split of interactions (e.g. 80/10/10 per user). Rejected: a random
  split lets the model "see the future" — training on a rating from 2001
  while being tested on a rating from 1998 for the same user is not a
  realistic recommendation scenario and inflates offline metrics relative to
  what a deployed system would actually achieve.
- Global timestamp cutoff (e.g. everything before date X is train, after is
  test) instead of per-user. Rejected for this dataset because MovieLens 1M
  users have very different activity windows; a global cutoff would leave
  many users with zero test interactions (they stopped rating before the
  cutoff) or zero train interactions (they started after it).

**Trade-off:** per-user leave-one/two-out is more realistic but each user
contributes exactly one test example, so test-set metrics are an average over
users rather than over independent random draws — variance is higher than a
larger random test set would give, and users with very few positives (1-2)
contribute no val, or no val/test at all.

## Two-stage architecture (retrieval then ranking) instead of one model

**Decision:** a lightweight retrieval model narrows ~3,700 movies to ~500
candidates optimized for recall, then a more expensive ranking model reorders
those 500 candidates optimized for NDCG.

**Alternatives considered:**
- A single model that scores all movies for every user directly. At
  MovieLens-1M scale (~3,700 movies) this is computationally fine, but the
  point of building this project is to demonstrate the two-stage pattern used
  by real-world recommenders (YouTube, Facebook, etc.) at scale, where
  scoring every item with a heavy model is infeasible.

**Trade-off:** two models to train, tune, and keep consistent (e.g. the
ranker depends on retrieval's candidate list and similarity score), and
recall lost at the retrieval stage is a hard ceiling the ranker can never
recover from.

## In-batch negatives for the two-tower retrieval model

**Decision:** train the two-tower model with in-batch negatives — for a
batch of B (user, positive item) pairs, every other item in the batch serves
as a negative for that user, via a softmax over the batch similarity matrix.

**Alternatives considered:**
- Explicit negative sampling (uniformly sampled or popularity-sampled
  negative items per positive). More control over negative distribution but
  more code and another hyperparameter (negatives-per-positive); in-batch
  negatives are simpler, are the standard technique for two-tower retrieval
  models, and get "harder" automatically as batch size grows.

**Trade-off:** batch composition affects the effective negative distribution
(popular items appear as negatives more often, which is actually desirable —
it teaches the model to distinguish a user's taste from generic popularity).

## FAISS for the candidate index

**Decision:** store trained item embeddings in a FAISS `IndexFlatIP` index and
retrieve the top-500 items per user by inner product (cosine, since vectors
are normalized).

**Alternatives considered:**
- Brute-force NumPy matrix multiply. At ~3,700 items this would actually be
  fast enough — FAISS is used here specifically to demonstrate the
  vector-search pattern used at production scale, not because it's strictly
  necessary at this dataset size.
- An approximate FAISS index (IVF, HNSW). Rejected for this dataset size:
  exact search over 3,700 items is already sub-millisecond, so approximate
  search would only add complexity (nprobe tuning) with no speed benefit.

## LambdaRank instead of binary classification for the ranker

**Decision:** train the LightGBM ranker with `objective="lambdarank"`,
grouped by user, instead of a binary classifier (e.g. logistic
regression/GBM predicting P(relevant)).

**Alternatives considered:**
- Binary classification (label = relevant/not relevant) with cross-entropy
  loss. Optimizes for calibrated probability of relevance, not for the
  ordering within a user's candidate list — but what we actually care about
  is ranking quality (NDCG@10), and LambdaRank directly optimizes a smooth
  approximation of NDCG's ranking-swap gradient.

**Trade-off:** LambdaRank scores are not calibrated probabilities, only
useful for sorting within a group; and lambdarank needs at least one
positive per group to produce a useful gradient signal for that group.

## Leakage prevention in ranker features

**Decision:** every statistic used as a feature is computed "as of" the split
it's predicting, never using future interactions:
- Features used to predict a user's **val** item (used to build the ranker's
  training data) are computed from **train only**.
- Features used to predict a user's **test** item (used for final evaluation)
  are computed from **train + val only** — val has, chronologically, already
  happened by the time we're predicting test.
- Item statistics (popularity, average rating) follow the same rule: a
  movie's popularity feature for a val-time prediction only counts train
  interactions with that movie, even from *other* users.

This is the single most important anti-leakage rule in the project and is
enforced by threading an explicit `as_of_split` argument through
`src/ranking/features.py` rather than ever computing a global statistic once
and reusing it everywhere.

## Cold start

**Decision:** a user id not seen during training (no train interactions)
falls back to the global popularity ranking rather than raising an error or
returning an empty/garbage recommendation.

**Alternatives considered:**
- Return an error / empty list for unknown users. Rejected: unrealistic for a
  demo API meant to be queried with arbitrary user ids, and it's a common,
  simple, real production pattern to fall back to non-personalized popularity
  for genuinely cold users.

---

## Interview Questions

*(Filled in during Phase 5 with 25-30 questions and grounded answers covering
the problem, data, retrieval, ranking, evaluation, leakage, cold start,
scaling, and limitations.)*
