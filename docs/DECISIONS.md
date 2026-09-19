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

## The two-tower retriever underperforms ALS on Recall@500

**Observation:** on this dataset, the two-tower neural retriever gets test
Recall@500 = 0.651 (after the fix below; 0.640 before it), below ALS's 0.750
(though above popularity's 0.565). More training alone didn't close the gap
(epochs 15 -> 40 only moved it from 0.625 -> 0.640, with clearly diminishing
returns per epoch — see `docs/IMPLEMENTATION_LOG.md`).

**Why this is an expected result, not a bug:**
- **MovieLens 1M is small and dense** (~3,700 items, ~575k positives) —
  exactly the regime where a pure matrix-factorization model like ALS is
  hardest to beat. ALS dedicates its entire parameter budget to a per-user
  and per-item latent vector directly fit to the interaction matrix via
  alternating closed-form least-squares solves — a very sample-efficient,
  well-posed optimization for small dense implicit-feedback matrices.
- **The two-tower model splits its capacity** between an ID embedding (doing
  the same job as ALS's factors) and side features (age, occupation, gender,
  genres, year) that are only weakly predictive on their own, and it's
  trained with noisier, gradient-based in-batch softmax updates rather than
  ALS's exact alternating solves — a less sample-efficient training signal
  for a dataset this small.
- **The two-tower architecture's real advantages don't show up at this
  scale**: it can embed a brand-new item from its metadata alone (genre,
  year) without any interaction history, and it scales to catalogs far too
  large for exact or even approximate matrix factorization to serve directly
  — neither of those advantages is exercised at MovieLens-1M's ~3,700-item
  scale, so here it mostly pays the cost (harder optimization) without the
  benefit.

**What we did about it:** first doubled the epoch budget (15 -> 40) and
confirmed diminishing returns rather than a stalled/broken run (0.625 ->
0.640). Diagnosed the actual gap: `UserTower` only saw `user_id` +
demographics, no direct signal about what the user actually liked — that
had to be learned indirectly through the `user_id` embedding via the
contrastive loss, the same job ALS's per-user factor does but through a
noisier gradient path. Fixed by feeding `UserTower` each user's real
**train-only** genre-preference vector (mean of the genre multi-hot vectors
of the movies they liked in train), through its own projection layer, the
same pattern `ItemTower` already used for its genre input (see
`docs/IMPLEMENTATION_LOG.md`, Step 7). That moved test Recall@500 from 0.640
to 0.651 and NDCG@500 from 0.1031 to 0.1061 — confirming the diagnosis was
directionally right without closing the gap to ALS. We did not go further
with a full hyperparameter sweep (embedding dim, learning rate, harder
negative mining, larger batch size) because the two-stage pipeline's overall
quality is decided at Stage 2 (NDCG@10 after ranking), not by Stage 1 recall
in isolation, and this recall level is still enough to hand the ranker
(Phase 3) a substantially better-than-random-popularity, real-taste-based
candidate set.

**Trade-off / limitation:** if this were a from-scratch production decision
rather than a portfolio project demonstrating the two-stage pattern, a
strong case exists for using ALS (or a hybrid: ALS embeddings as an
additional input feature to the two-tower model) as the retrieval stage
instead, at least until the catalog grows large enough or cold-start
coverage becomes the binding constraint. This is called out again in
`README.md`'s Limitations & Future Work.

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

## Why retrieval-only top-10 is the worst of the four models, yet the full pipeline is the best

**Observation:** on test, taking the two-tower retriever's raw top-10 by
similarity score gets Recall@10 = 0.0318 / NDCG@10 = 0.0140 — worse than
*both* popularity (0.0393 / 0.0193) and ALS (0.0654 / 0.0329). But reranking
those same retrieval candidates with the LightGBM ranker gets Recall@10 =
0.0828 / NDCG@10 = 0.0423 — beating every other model, including ALS, by a
wide margin (+27% Recall@10, +29% NDCG@10 relative to ALS).

**Why retrieval-only top-10 is weak:** the two-tower model is trained with
in-batch softmax loss over the *entire batch of candidates*, and evaluated
at Recall@500 — its objective only asks "is the right item somewhere in a
list of 500," never "is the right item in the top 10 specifically." A model
can be very good at that recall-oriented objective while producing a poor
*fine-grained ordering* within its own candidate list, because nothing in
its training signal ever rewarded getting the single best item into the
first 10 positions out of 500. This is expected, not a bug — it's precisely
the reason a two-stage system exists instead of shipping retrieval's output
directly.

**Why the full pipeline still wins:** the ranker only has to solve an easier
problem — reordering ~500 candidates that retrieval already identified as
plausible — and it does so with a rich, supervised, position-aware objective
(LambdaRank) plus features retrieval never had access to (item popularity,
average rating, explicit genre overlap, user activity level). It also
starts from a candidate pool with much higher recall headroom (0.65 at
K=500) than ALS's implicit top-10 cutoff ever offers the ranker a chance to
exploit for *popularity/rating* signal on top of embedding similarity. The
combination — broad, taste-aware candidate generation (Stage 1) plus
precise, feature-rich reordering (Stage 2) — is a better division of labor
than either a single similarity score (retrieval-only) or a single
collaborative-filtering score (ALS) alone.

**What this demonstrates:** the two-stage architecture's value isn't that
Stage 1 needs to beat every single-stage baseline on its own — it's that
combining a recall-oriented Stage 1 with a precision-oriented Stage 2
produces a better final Top-10 than any single model here, even one (ALS)
that beats Stage 1 in isolation.

## Why the absolute Recall@10 (8%) and NDCG@10 (4%) look low

**Observation:** even the best model (the full pipeline) only gets
Recall@10 = 0.0828 and NDCG@10 = 0.0423 on test. In isolation these numbers
look weak, but they're a property of how hard this specific evaluation task
is by construction, not a sign of a broken pipeline.

**Why the task is intrinsically hard:** Recall@10 here means: out of the
entire catalog minus what the user has already seen (~3,700+ candidate
movies), did the model place the *one specific movie* the user will interact
with next somewhere in 10 guesses? That's a needle-in-a-haystack task, not
"did we recommend something reasonable." A **random** recommender choosing
10 movies uniformly from ~3,700 unseen candidates would get an expected
Recall@10 of about 10/3,700 ≈ **0.27%**. The full pipeline's 8.28% is
roughly **30x better than random**, and even popularity alone (3.93%) is
already ~15x better than random just from recommending generally popular
movies. Judged against that baseline — not against an intuitive "8% sounds
low" reaction — the result is strong, not weak.

**Why this looks lower than numbers reported in some papers:** many
published MovieLens leave-one-out results (e.g. the original Neural
Collaborative Filtering paper) don't rank the held-out item against the
full catalog — they rank it against just 99 randomly sampled negative items,
so their "HR@10" means "top-10 out of 100 candidates," not "top-10 out of
~3,700." That's a fundamentally easier task and the two numbers are not
directly comparable. This project ranks against the entire remaining
catalog every time (the FAISS retrieval step searches the whole item set,
not a small sampled pool), which is more realistic and more representative
of a real deployed system, at the cost of lower absolute metric values.

**Why NDCG@10 (0.042) is roughly half of Recall@10 (0.083):** NDCG
additionally discounts by rank position (a hit at position 1 counts fully;
a hit at position 10 counts only `1/log2(11) ≈ 30%` as much). NDCG@10 being
about half of Recall@10 says that when the pipeline does find the right
movie in the top 10, it tends to land more toward the middle/back of that
list rather than at #1 — itself a legitimate (if unflattering) signal about
ranking sharpness, not an error.

**The comparison that actually matters:** every model (popularity, ALS,
retrieval-only, full pipeline) is evaluated identically — same test users,
same full-catalog candidate pool, same K=10 — so the *relative* comparison,
not the absolute value, is the meaningful result: the full pipeline beats
ALS by +27% Recall@10 / +29% NDCG@10. That relative lift is what
demonstrates the two-stage architecture is working, independent of what the
raw percentages look like.

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

**Q: Your two-tower retrieval model scores lower than ALS on Recall@500
(0.651 vs 0.750 on test). Why, and why did you keep it instead of just using
ALS for retrieval?**
A: See "The two-tower retriever underperforms ALS on Recall@500" above — in
short: MovieLens 1M is small and dense (~3,700 items), which is exactly the
regime where ALS's closed-form alternating-least-squares fit to the
interaction matrix is hardest to beat, while the two-tower model splits its
capacity between an ID embedding and side features, trained with noisier
gradient-based updates. Its real advantages — embedding brand-new items from
metadata alone, and scaling to catalogs too large for matrix factorization
to serve directly — aren't exercised at this scale, so it mostly pays the
optimization cost without the benefit here. Feeding the user tower a real
train-history genre-preference vector (instead of relying purely on the
learned `user_id` embedding) closed part of the gap (0.640 -> 0.651) but not
all of it, confirming missing behavioral signal was a real but partial
cause. It was kept anyway to demonstrate the two-stage retrieval-then-ranking
pattern used in production-scale recommenders (the actual point of this
project), with the trade-off reported honestly rather than hidden; in a
from-scratch production system at this scale, using ALS (or ALS embeddings
as a feature into the two-tower model) for retrieval would be a defensible
alternative.

**Q: What did `faiss.omp_set_num_threads(1)` fix, and why was it needed?**
A: Running PyTorch's own parallel tensor operations and then calling FAISS's
OpenMP-parallel `index.search(...)` in the same process caused a hard
segfault on this machine (macOS, pip-installed `faiss-cpu` + `torch`, both
of which bundle their own OpenMP runtime). Restricting FAISS to a single
thread avoids the collision; since the catalog is only ~3,700 items, exact
single-threaded search has no meaningful speed cost. A separate, milder
symptom of the same root cause (`OMP: Error #15`, a duplicate-runtime abort)
was fixed with the standard `KMP_DUPLICATE_LIB_OK=TRUE` workaround, and
import order (`torch` before `faiss`) mattered for a third variant of the
same underlying issue. All three fixes live together in
`src/retrieval/index.py`, the one module that imports both libraries.

**Q: Your retrieval-only top-10 (0.0318 Recall@10) is worse than both
popularity (0.0393) and ALS (0.0654), yet your full pipeline (0.0828) beats
all of them. How can Stage 1 alone be the worst option while Stage 1 + Stage
2 together is the best?**
A: See "Why retrieval-only top-10 is the worst of the four models, yet the
full pipeline is the best" above — in short: the two-tower model is trained
and evaluated to get the right item *somewhere in 500 candidates*
(Recall@500), and nothing in that objective rewards precise ordering within
the first 10 of those 500, so taking its raw top-10 directly is a poor use
of what it's actually good at. The LightGBM ranker solves an easier,
different problem — reordering a pool retrieval already narrowed down — with
a position-aware objective (LambdaRank) and features retrieval never sees
(popularity, average rating, explicit genre overlap). The two-stage
architecture's value is in that division of labor, not in Stage 1 winning
on its own; a broad recall-oriented candidate generator feeding a precise
reordering model can beat every single-stage baseline even when the
candidate generator loses to one of those baselines by itself.

**Q: How do you know your ranker training data was built correctly, without
just trusting the code?**
A: A concrete numeric cross-check: the ranker's training table has 4,179
positive labels out of 6,035 user-groups. Phase 2 independently reported
Recall@500 = 0.6925 on val for the same retrieval model. 0.6925 x 6035 ≈
4,179 — the exact same number, derived two different ways (one from the
metrics module's `mean_at_k`, the other from literally counting `label == 1`
rows in the candidate table). That agreement is strong evidence the
candidate generation, exclusion logic, and labeling in
`src/ranking/features.py` are consistent with the retrieval evaluation in
`src/retrieval/evaluate.py`, rather than two independently-buggy
implementations that happen to run without crashing.

**Q: Why did `import lightgbm` work earlier in the project but fail with a
`libomp.dylib` load error inside `src/ranking/train.py`?**
A: The earlier successful import happened in a line that also imported
`torch` first; torch's own bundled OpenMP runtime happened to satisfy
LightGBM's native library's dependency on `libomp.dylib` as a side effect,
which masked the fact that Homebrew's `libomp` — LightGBM's actual declared
dependency on macOS — was never installed on this machine. `train.py` has
no reason to import `torch`, so the missing dependency surfaced as a hard
`OSError` there. The correct fix was installing the real dependency
(`brew install libomp`), not relying on an incidental import order in a
script that shouldn't need `torch` at all.

**Q: Your final Recall@10 is only 8% and NDCG@10 only 4%. Isn't that a bad
result?**
A: See "Why the absolute Recall@10 (8%) and NDCG@10 (4%) look low" above —
in short, no: those numbers have to be judged against the task's difficulty,
not read as a percentage in isolation. Recall@10 here means finding the one
specific movie a user will watch next among the ~3,700+ movies they haven't
seen, using only 10 guesses — a random recommender would score about 0.27%
on that task, so 8.28% is roughly 30x better than chance. Numbers like 0.6-
0.7 seen in some published leave-one-out papers typically rank the true
item against only 99 sampled negatives (top-10 out of 100 candidates), a
much easier task than ranking against the full catalog the way this project
does — the two aren't directly comparable. The metric that actually matters
for judging this project is the *relative* one: the full pipeline beats ALS
by +27% Recall@10 / +29% NDCG@10 under an identical evaluation protocol,
which is what demonstrates the two-stage architecture adds real value.
