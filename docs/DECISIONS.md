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

## Training only on positive interactions (not using low ratings as explicit negatives)

**Decision:** every model in this project (ALS, the two-tower retriever, the
LightGBM ranker) is trained as an implicit-feedback problem — the only label
that exists is "positive" (rating >= 4). Ratings of 1-3 are never used to
construct an explicit "the user dislikes this" training label; negatives are
instead generated implicitly and automatically by each model:
- **ALS** treats every *unobserved* (user, movie) pair as a low-confidence
  negative and every *positive* pair as high-confidence, per the standard
  implicit-feedback matrix factorization formulation (Hu, Koren & Volinsky,
  2008) — this is exactly the setting `implicit`'s ALS is built for.
- **The two-tower retriever** uses in-batch negatives: only positive (user,
  item) pairs are fed in, and every other item in the same training batch
  becomes an implicit negative for that user via the softmax denominator.
- **The LightGBM ranker** labels exactly one candidate 1 (the item the user
  actually interacted with next) per group, and the rest 0 — again, no
  rating value below 4 is ever consulted.

**Alternatives considered:**
- Use ratings 1-3 as explicit negative labels (e.g. hard negatives for the
  retriever, or a richer 3-class label for the ranker). Rejected for three
  reasons:
  1. **Realism** — this is deliberately an implicit-feedback problem (one of
     the project's original requirements), because that's what production
     recommenders actually observe: clicks, watches, purchases, never a
     clean "the user dislikes this." Pulling in explicit low ratings would
     make this an easier, less representative problem than the one the
     project is meant to demonstrate.
  2. **A low rating is still engagement, not absence** — a user who watched a
     movie and rated it 2 stars invested more attention in it than a user who
     never watched it at all. Treating "watched and disliked" as equal to or
     worse than "never watched" conflates two different signals.
  3. **Selection bias** — which movies a user chooses to watch (and
     sometimes rates low) is itself non-random. Using those low ratings as
     hard negatives risks penalizing widely-watched mainstream movies simply
     because they attract more 2-3 star ratings from a broad audience, not
     because they are poor recommendations for other users.

**Trade-off:** the model never receives a true "actively disliked" signal,
only "positive" vs. "unknown" — so it cannot distinguish a movie a user would
dislike from one they simply haven't encountered yet. Mining ratings 1-3 as
hard negatives (rather than random/in-batch negatives) for the retrieval
stage is a reasonable future extension, noted in Limitations & Future Work.

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

**Why hold out exactly one item per split (leave-one-out), not a percentage
(e.g. "last 20% of each user's interactions"):**
- **Users don't have much history to spare.** MovieLens 1M only guarantees
  users have >=20 *ratings* total, and after filtering to positives
  (rating >= 4) many users have well under that. A percentage-based holdout
  would need special-casing for small users anyway (rounding to 0), while a
  fixed count of 1 works uniformly and leaves the maximum possible history in
  `train`, which matters most since `train` is what the retrieval model, ALS,
  and every ranker feature are built from.
- **It matches the actual problem statement.** The task is "given a user's
  past interactions, predict what they'll like next," not "given a random
  slice of their history, fill in a random missing chunk." Holding out just
  the single most recent positive is the direct instantiation of "next
  positive interaction."
- **It keeps the aggregate metric from being dominated by power users.** A
  percentage-based holdout gives a 500-rating user ~100 test examples and a
  5-rating user ~1, so the aggregate Recall@K/NDCG@K would mostly reflect
  performance for the most active users. With exactly one held-out item per
  user, every user — casual or heavy — contributes equally to the average.

This is the classic "leave-one-out" evaluation protocol widely used in
implicit-feedback recommendation research (e.g. He et al., *Neural
Collaborative Filtering*, 2017), applied here as leave-two-out (one for val,
one for test).

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

Populated incrementally as questions come up while building the project
(each one grounded in an actual question asked and answered during
development); expanded to the full 25-30 question set covering retrieval,
ranking, scaling, and limitations in Phase 5.

**Q: `config.yaml` has both `data.positive_rating_threshold: 4` and
`split.min_positives_for_val: 3`. Doesn't that contradict "rating >= 4 is
positive"?**
A: No — they're unrelated thresholds applied to different things.
`positive_rating_threshold` filters the 1-5 star `rating` column to decide
what counts as a positive interaction at all (used once, in
`make_implicit()`). `min_positives_for_val` never looks at the rating value;
it counts *how many positive interactions a user already has* (after that
filtering) to decide whether their history is long enough to also carve out
a validation example, inside `temporal_split()`. By the time the split logic
runs, the rating column has already done its only job.

**Q: What's the difference between the Popularity baseline and ALS?**
A: Popularity is non-personalized — it ranks movies by how many positive
interactions they received in training and recommends the same ranked list
to everyone (minus what they've already seen). ALS (Alternating Least
Squares) is personalized matrix factorization: it learns a low-dimensional
embedding for every user and every movie such that their dot product
approximates affinity, alternating between solving for user embeddings
(item embeddings fixed) and item embeddings (user embeddings fixed) until
convergence. Because it learns per-user vectors, it can rank differently
for different users, which is why it beats popularity on every metric in
our results (test Recall@10 0.065 vs. 0.039 — see `reports/results.md`).

**Q: What do Recall@K and NDCG@K measure, and why use Recall@K for
retrieval but NDCG@K for ranking?**
A: Recall@K = (relevant items found in the top-K) / (total relevant items)
— it only asks whether the relevant items were found, not where. NDCG@K
additionally discounts hits by position (`1/log2(rank+1)`) and normalizes
against the best possible ordering, so it rewards ranking relevant items
higher. Stage 1 (retrieval) only needs to narrow ~3,700 movies down to the
right ~500 candidates — order doesn't matter yet, so it's optimized for
Recall@500. Stage 2 (ranking) determines the actual order the user sees in
the final top-10, which is exactly what NDCG@10 measures, so that's the
ranker's objective.

**Q: Why train only on positive interactions instead of also using ratings
1-3 as explicit negative labels?**
A: See the "Training only on positive interactions" decision above — in
short: this is deliberately an implicit-feedback problem (matching what
production recommenders actually observe), a low rating still represents
real engagement rather than absence of interest (so it isn't equivalent to
"never watched"), and which movies a user chooses to watch and rate poorly
is itself a biased sample that could unfairly penalize widely-watched
mainstream movies. Negatives are instead generated implicitly per model:
ALS's confidence weighting, the two-tower model's in-batch negatives, and
the ranker's 0-labeled non-target candidates.

**Q: With exactly one relevant item per user, what does Recall@K actually
measure?**
A: It collapses to a binary hit/miss for that user: since
`|relevant items| = 1`, the numerator (relevant items found in top-K) can
only be 0 or 1, so `recall_at_k` is exactly 1.0 if the held-out movie is in
the top-K and 0.0 otherwise — no partial credit is possible. Averaging that
0/1 value across all evaluated users is the same as computing the fraction
of users whose held-out movie was successfully surfaced — i.e., with a
single held-out item per user, Recall@K and Hit Rate@K are mathematically
identical. NDCG@K does not collapse the same way, because it still credits
*where* in the top-K the hit landed.

**Q: Why hold out only one interaction per user for val/test instead of a
percentage-based split (e.g. last 20% of each user's history)?**
A: See "Why hold out exactly one item per split" under the temporal-split
decision above — in short: MovieLens users often don't have much positive
history to spare, so a fixed count of 1 avoids needing special-casing for
small users while maximizing what's left for `train`; it directly matches
the "predict the next positive interaction" problem statement; and it
prevents power users (with hundreds of ratings) from dominating the
aggregate test metric the way a percentage-based holdout would.
