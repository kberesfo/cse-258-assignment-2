# Plan: predicting play on Steam

How we got to the current approach, and what we're building next. Numbers are from the
10% user sample (`users_0.1`) and will be re-checked on the full data.

## 1. Task

**Main task (A), ranking:** given a user's library, rank 100 candidate games so the game
they go on to **play for at least 2 hours** comes first. The other 99 candidates are games
they don't own.

**Supporting task (B), engagement:** given that a user owns a game, predict how long they
play it, as log(1 + minutes).

### Why "play", not "own"

We started with "predict which games a user owns." The EDA argued against it:

- **Ownership is a weak preference signal.** 36% of owned games were never played, and 62%
  were played for under 2 hours (bundles, sales, free-to-play installs).
- **Part of the ownership task is just recovering bundles.** A personalized baseline
  (item kNN: Jaccard similarity to the user's library) beat popularity by +0.21 Hit@10
  on ownership, but only by about +0.09 on play. Games sold together are owned together
  whether or not the user wants them; predicting play removes most of that.
- **Playtime helps once play is the target.** Ranking by how many users *played* a game
  beats ranking by how many *own* it (Hit@10 0.405 vs 0.357 at 120 minutes), and the lead
  grows with the threshold (+0.028 at 5 min, +0.063 at 240). On the ownership task the two
  are identical.

### Why 120 minutes

- It's Steam's refund window: a refund is offered within two weeks of purchase and with
  under two hours of playtime. Past 2 hours, the user has kept and committed to the game.
- It keeps 91% of validation users (5,946 of 6,535 for ownership; `explore_play` §2).
- The data doesn't pick a threshold: above 0 minutes, item kNN's margin over the best
  popularity score is flat (+0.102 at 5 min, +0.094 at 30, +0.095 at 60, +0.092 at 120,
  +0.093 at 240). Higher thresholds only cost users (88% kept at 240 min).
- Scores rise with the threshold because longer-played games skew popular, so we compare
  margins within a threshold, never scores across thresholds.
- **5 minutes is kept as a robustness check** ("played at all, excluding launch-only
  sessions").

### Why task B is its own task

Engagement is only weakly related to popularity: across games with ≥ 5 owners, owners vs
median log playtime has Spearman ρ = 0.19 (play rate 0.20, playtime among players 0.13).
Predicting how long someone plays an owned game is not just re-learning which games are
popular: within a user, owner popularity orders their held-out games poorly (Spearman
0.165), while each game's average playtime orders them well (0.536).

## 2. Evaluation protocol

Both tasks are built once in `prep` from the same user sample and cached; each model trains
on its own task's split (training on one task's train set and scoring the other's test set
would leak held-out pairs).

### Task A (play)

- **Split:** leave-one-out per user among *played* (≥ 120 min) games: one for validation,
  one for test. Users need at least 3 played games (5,946 validation users). Owned games
  under 120 minutes stay in training as context and are never held out.
- **Candidates:** the held-out game + 99 games the user owns in **no** split, drawn from
  training games and **sampled in proportion to popularity**, once, in `prep`, so every
  model ranks the same rows.
  - *Why popularity-weighted:* with ownership as the positive and uniform negatives, the
    held-out game was ~90× more popular than the negatives (median owners 630 vs 7), so
    popularity alone found it. With popularity weighting and play as the positive, the gap
    is ~3× (median 821 vs 260), so the test measures personalization.
- **Owned but under 120 minutes:** neither a positive nor a negative in evaluation.
- **What the protocol rewards:** the held-out game is always one the user owns, and the 99
  negatives never are, so a model that predicts ownership well gets credit. That's not a
  leak (the held-out game isn't in training), but it means task A partly rewards "will
  they own it" as well as "will they play it" (see BPR's positives ablation, §4).
- **Metrics:** Hit@10, NDCG@10, MRR per user, averaged. 95% bootstrap intervals over
  users (1,000 resamples), and paired bootstrap intervals on the difference between a
  model and its baseline (`core/evaluate.py`: `evaluate_ranking`, `compare`). Pairing
  resamples the same users for both models, which roughly halves the interval on a
  difference (±0.005 Hit@10 vs ±0.012 for one model's score), so small real gaps show up.

### Task B (engagement)

- **Split:** 3 owned games held out per user for validation and 3 for test (users need at
  least 7 owned games; 6,020 users, 99.6% of whom keep all 3 after cold drops). Includes
  never-played games (target 0).
- **Weighting:** every user contributes 3 held-out games, so metrics weight users equally,
  while train rows are dominated by large, mostly unplayed libraries. That's why valid has
  fewer never-played games than train (31.0% vs 36.3%); weighted by user, train matches
  (30.8%), so the split is representative.
- **Metrics:** MSE and MAE on log(1 + minutes), plus within-user Spearman over each user's
  3 held-out games (does the model order a user's games correctly?). Spearman is reported
  over the 94% of users where it's defined; the rest have all 3 held-out games tied,
  nearly always all at 0 minutes.

### Both tasks

- **Cold items:** held-out rows whose game never appears in training are dropped (45 per
  task, under 0.4% for A). For task A this check runs before the under-120-minute rows are
  added back to training, so a few games owned (but not played) by others are dropped too.
  It removes rows, never leaks; left as is.
- **Leakage guards:** no training negative is one of the task's held-out pairs
  (`TrainData.exclude`). A review of a held-out game reveals the answer (284 play / 408
  engagement held-out pairs have one), so `data/tasks.py` drops reviews of each task's
  held-out pairs before any model sees them. Reviews themselves are kept unsplit: no task
  predicts them.
- Tune on validation; score test once. Test has been looked at only as a leak check
  (does it match validation?), never to choose anything.

## 3. Models: two ways to adapt latent factors to implicit feedback

Both have the same form, μ + b_u + b_i + γ_u · γ_i. They differ in what they're trained to do.

| | **BPR-MF** | **LFM on log-hours** |
|---|---|---|
| Trained on | pairs (u, owned game i, non-owned game j) | owned games only |
| Objective | score(i) > score(j) (pairwise, ranking) | squared error on log(1 + minutes) |
| Assumption about non-owned games | ranks *below* played games (relative) | **none**: never seen in training |
| Native task | A (ranking) | B (engagement) |

One latent factor model, two training objectives. The point of the comparison: **"not
owned" is not "disliked."** A non-owned game may be unknown, unaffordable, or exactly what
the user would love, which is what a recommender is for. BPR only assumes it ranks below a
played game; the LFM makes no assumption at all. Each model is tested on both tasks:

- **BPR on task B:** its scores aren't in hours, so we use within-user rank correlation
  (does it order a user's owned games by playtime?).
- **LFM on task A:** rank candidates by predicted hours (does predicting engagement
  transfer to recommending?). Its limitation: it never saw a non-owned game in training.

Within one user, b_u is the same for every game, so a bias model orders a user's games
exactly as the item averages do (Spearman 0.540 vs 0.536). Only γ_u · γ_i can reorder a
user's games, so on task B, **within-user Spearman measures what the latent factors add
beyond biases**, while MSE mostly rewards getting the biases right.

### Training choices to ablate

- ~~Owned games under 120 min, for BPR: positives, or left out (then sampleable as
  negatives)~~ (done: "owned" positives win; §4).
- LFM target: log(1 + minutes) vs played ≥ 120 min (0/1). *Deferred.*
- ~~Two-part (hurdle) model for task B: P(played) × E[log minutes | played]~~ (checked, not
  pursued: no signal ranks never-played games better than the LFM itself; §4, §5).
- Implicit ALS (Hu, Koren & Volinsky 2008) with playtime as confidence, as a third
  latent-factor variant. *Deferred.*
- Similarity baselines: Jaccard vs cosine for item kNN; user kNN with only each user's
  top-k neighbors, or normalized by the game's player count (the plain sum favors widely
  played games).

## 4. Baselines

All on validation. Task A is scored on the cached `play_valid_candidates`
(`explore_play` §4); task B on `own_valid` (`explore_engagement` §4).

**Task A** (120 min, 5,946 users)

| Baseline | Hit@10 | NDCG@10 | MRR |
|---|---|---|---|
| random | 0.090 | 0.041 | 0.049 |
| play rate (shrunk, a = 10) | 0.350 | 0.192 | 0.169 |
| owner popularity | 0.357 | 0.213 | 0.190 |
| play popularity | 0.405 | 0.246 | 0.220 |
| user kNN (cosine, sum over similar users) | 0.440 | 0.268 | 0.237 |
| item kNN (cosine, max over history) | 0.493 | 0.304 | 0.266 |
| **item kNN (Jaccard, max over history)** | **0.497** | **0.303** | **0.264** |

Item kNN (Jaccard) is the bar for BPR. Paired bootstrap at 120 min (`explore_play` §4b,
1,000 resamples; Δ = row − column, 95% CI):

| Comparison | ΔHit@10 | ΔNDCG@10 |
|---|---|---|
| item kNN Jaccard − cosine | +0.005 [−0.001, +0.010] | −0.001 [−0.003, +0.002] |
| item kNN − play popularity | +0.093 [+0.083, +0.102] | +0.057 [+0.051, +0.063] |
| item kNN − user kNN | +0.058 [+0.048, +0.066] | +0.035 [+0.029, +0.042] |
| user kNN − play popularity | +0.035 [+0.030, +0.041] | +0.022 [+0.020, +0.024] |

Jaccard and cosine are statistically tied, so Jaccard stays the default; every other gap is
real. User kNN trails item kNN by 0.17 at 0 minutes, where item–item bundle structure
dominates. The similarity functions are vectorized in `core/similarity.py`; the sparse item
kNN reproduces the original pure-Python Jaccard exactly.

**Task B** (log1p minutes, 6,020 users)

| Baseline | MSE | MAE | Within-user Spearman |
|---|---|---|---|
| global mean | 9.890 | 2.784 | undefined (one value per user) |
| user mean (λ = 5) | 8.902 | 2.567 | undefined |
| owner popularity | | | 0.165 |
| item mean (λ = 1) | 5.587 | 1.871 | 0.536 |
| **bias model** (μ + b_u + b_i, λ = 5) | **5.029** | **1.781** | **0.540** |

The bias model is the bar for the LFM on both MSE and Spearman. The game matters far more
than the user: the item mean removes ~4× more error than the user mean (−4.30 vs −0.99 MSE
from the global mean). Owner popularity's 0.165 is the bar for BPR on task B.

The registered bias model (`bias_eng`) reproduces this row exactly. With task B predictions
clipped at 0 (log1p(minutes) can't be negative; §4) it scores 5.007 / 1.765 / 0.542, which
is the bar used in `task_b_engagement`.

### Task A models (`task_a_play`)

**BPR-MF** (`models/ranking/bpr.py`, `implicit`, one thread so fits repeat exactly). Tuned
on validation in stages, 3 seeds per setting; seed noise is ~0.001–0.002 NDCG, so cells
within ~2 std are ties. Every grid row is saved as `grid_bpr_*.csv`.

| Stage | Finding (NDCG@10, mean of 3 seeds) |
|---|---|
| played: factors × reg | regularization dominates: 0.001 and 0.1 both ~0.27, 0.01 ~0.35 |
| played: reg (fine) | peak at 0.02 (0.362 vs 0.348 at 0.01) |
| played: factors at 0.02 | 32 – 128 tied; 256 slightly lower → 32 |
| played: iterations | peak at 100; 200+ overfits (0.339 at 400) |
| played: iterations × reg | 200 / 0.03 ties 100 / 0.02: longer runs need more reg |
| played: learning rate | 0.01 (0.005 under-trains, 0.02+ overshoots) |
| owned: reg | peak at 0.01 (0.384); ~3× the positives need less regularization |
| owned: factors × iterations | 64 beats 32 (+0.009, ~4 std); 64 = 128 = 256; iterations flat |

**Final BPR:** positives "owned", 64 factors, regularization 0.01, 100 iterations,
learning rate 0.01, which are the library defaults apart from the positives. The best
hyperparameters depend on the positives: "played" (fewer positives) needs stronger
regularization and a smaller model; "owned" supports a larger, less regularized one.

| Model (validation, seed 258) | Hit@10 | NDCG@10 | MRR |
|---|---|---|---|
| item kNN (Jaccard) | 0.497 | 0.303 | 0.264 |
| best "played" BPR (32 factors, reg 0.02) | 0.589 | 0.362 | 0.309 |
| **final BPR** (owned, 64 factors, reg 0.01) | **0.619** [0.606, 0.631] | **0.388** [0.379, 0.398] | **0.335** [0.325, 0.344] |

| Comparison (paired bootstrap, 95% CI) | ΔHit@10 | ΔNDCG@10 |
|---|---|---|
| **final BPR − item kNN** | **+0.122 [+0.106, +0.139]** | **+0.085 [+0.074, +0.097]** |
| final BPR − user kNN | +0.179 [+0.164, +0.196] | +0.121 [+0.110, +0.132] |
| owned − played (final vs best "played") | +0.031 [+0.018, +0.043] | +0.027 [+0.018, +0.036] |
| best "played" − untuned "played" | +0.011 [+0.003, +0.020] | +0.013 [+0.008, +0.019] |
| untuned BPR (played) − item kNN | +0.080 [+0.065, +0.095] | +0.045 [+0.035, +0.056] |

So of the final model's +0.122 Hit@10 over item kNN, about +0.080 is BPR itself (untuned,
played positives), +0.011 is tuning, and +0.031 is the choice of positives.

**Why "owned" positives win:** owned-but-unplayed games still show taste (with "played"
positives, BPR can sample them as *negatives*, teaching that games like ones you own are
bad), and the protocol partly rewards predicting ownership (§2). Both explanations are
reported; the second is a limitation, not a leak.

**Leak checks:** no held-out pair appears anywhere in training, and test matches
validation for every BPR checked: untuned (Hit@10 0.566 vs 0.578), best "played" (0.578 vs
0.589) and final (0.615 vs 0.619; NDCG 0.382 vs 0.388).

### Task B models (`task_b_engagement`)

**LFM** (`lfm`: Surprise `SVD` in `models/rating/lfm.py`): μ + b_u + b_i + γ_u · γ_i,
trained on owned games only with target log1p(minutes) and no sampled negatives. Task B
predictions are clipped at 0, which removes impossible negative predictions (bias model
−0.020 MSE, LFM −0.007). Tuned on validation in stages; fits repeat exactly for a seed, and
seed noise is ~0.003–0.03 MSE and ~0.001–0.003 Spearman. Every grid row is saved as
`grid_lfm_*.csv`.

**Selection rule** (set before stage 2): lowest validation MSE, provided Spearman is not
clearly below the bias model's (an LFM that only wins through its biases hasn't shown
personalization).

| Stage | Finding (MSE / within-user Spearman) |
|---|---|
| factors × reg_all (seed 0) | Surprise's defaults overfit (50 factors, reg 0.02: 5.38 / 0.507); best MSE 4.819 at reg 0.1, but Spearman peaks at reg 0.2 (0.558): biases and factors want different regularization |
| bias vs factor reg (seed 0) | separating them helps (4.772); bias regularization doesn't matter (0.02 = 0.05) → 0.02 |
| factor reg × factors (3 seeds) | factor reg 0.15 is best on MSE and ties the best Spearman; 100 factors beat 50 (4.684 vs 4.715) |
| more factors (3 seeds) | 100 = 200 = 400 (4.684 / 4.681 / 4.685) → 100 |
| epochs × learning rate (3 seeds) | roughly epochs × lr matters; 80 × 0.002 ties the default 20 × 0.005 on MSE (4.671 vs 4.684) and is worse on Spearman → defaults |

**Final LFM:** 100 factors, bias regularization 0.02 (`reg_bu`, `reg_bi`), factor
regularization 0.15 (`reg_pu`, `reg_qi`), 20 epochs, learning rate 0.005.

| Model (validation, seed 258) | MSE | MAE | Within-user Spearman |
|---|---|---|---|
| bias model (`bias_eng`) | 5.007 [4.907, 5.104] | 1.765 [1.746, 1.785] | 0.542 [0.526, 0.556] |
| untuned LFM (Surprise defaults, 50 factors) | 5.300 [5.170, 5.423] | 1.769 [1.747, 1.792] | 0.516 [0.501, 0.530] |
| **final LFM** | **4.690** [4.589, 4.787] | **1.700** [1.680, 1.720] | **0.556** [0.542, 0.571] |

| Comparison (paired bootstrap, 95% CI) | ΔMSE | ΔMAE | ΔSpearman |
|---|---|---|---|
| **final LFM − bias model** | **−0.317 [−0.354, −0.280]** | **−0.065 [−0.073, −0.058]** | **+0.015 [+0.006, +0.023]** |
| final LFM − untuned LFM | −0.610 [−0.681, −0.536] | −0.070 [−0.082, −0.057] | +0.041 [+0.029, +0.052] |

- **The LFM beats the bias model on every metric** (−6.3% MSE). The Spearman gain shows
  γ_u · γ_i personalizes (within a user only it can reorder games), but it is small: the
  step from owner popularity to the item mean was 0.165 → 0.536, and the factors add 0.015.
  Which game it is explains most of how much people play it.
- **Tuning was essential:** untuned, the LFM is worse than the bias model; tuning gains about
  twice the LFM's whole margin over it.
- **In plain terms,** a prediction is typically off by a factor of 4.0 in (1 + minutes)
  (bias model 4.4, global mean 28); 72% are within 10×. Error in minutes is misleading here
  (the global mean has the lowest median minutes error, because a third of the truths are 0).

**Where the error comes from** (`task_b_engagement` §4, row-level):

- The target has two peaks (31% never played, and a broad hump of played games); the model
  has one, between them. It is well calibrated (in every prediction decile the mean truth
  matches the mean prediction) but shrinks toward the middle at both ends (prediction std
  2.17 vs 3.12).
- Never-played games are 41% of the squared error from 31% of rows; games played 10 h+ are
  31% from 24% of rows. The LFM beats the bias model in every playtime bucket, most at the
  extremes.
- A perfect never-played detector would cut MSE from 4.692 to 2.780 (−41%), but no signal
  ranks never-played games better than the LFM's own prediction (AUC 0.865; item zero rate
  0.829, already in the LFM through b_i; item kNN ~0.70; user zero rate 0.674). Its hedge is
  close to the best squared error allows, so a hurdle model built on these signals is not
  pursued (§5).

## 5. Rejected or deferred

- **Treating non-owned games as confident zeros** (pointwise SVD with sampled negatives):
  it trains the model to treat the very games it should recommend as disliked. The original
  Surprise models did this (`neg_ratio = 4`); they are removed, and task B's models train on
  owned games only.
- **Hurdle model / zero predictor for task B:** zeros are the largest source of error, but
  the LFM already ranks them best (AUC 0.865 vs ≤ 0.829 for user, item and kNN signals), so
  there is no information for a first stage to add. Future work: a classifier combining
  these signals on top of the LFM.
- **kNN regressors for task B:** item kNN on co-ownership (similarity-weighted mean of the
  user's log-minutes on similar games) scores MSE 8.65, far worse than the bias model: it
  ignores how much the game itself is played, which is most of the signal.
- **Low playtime as "dislike"** for the ownership task: playtime is a snapshot with no
  purchase date, so "under 2 hours" can mean disliked, not started yet, or just bought.
  Refunded games never appear in the data at all.
- **Negative reviews as negatives:** the cleanest dislike signal, but ~11% of ~6K reviews,
  far too few against 5M ownership rows. Possible as an EDA check instead.
- **Predicting the review `recommend` flag:** 89.1% positive and few reviews.
- **Bundle and metadata features:** bundles cover 14% of games; metadata is missing for
  several of the most-owned titles. Possible later ablations.
- **Stricter task A negatives:** add the user's owned-but-unplayed games to the candidate
  negatives, so a model must pick the game they *played* over ones they merely own. Removes
  the ownership credit in today's protocol; deferred, a robustness check if time allows.

## 6. Next steps

1. ~~Move the 120-minute play split into `prep`~~ (done: `[split.play] min_minutes = 120`,
   `[split.engagement]` for task B).
2. ~~Threshold table, task B tie check, held-out vs train, baselines on the cache~~ (done:
   `explore_play`, `explore_engagement`; similarity vectorized in `core/similarity.py`).
3. ~~`evaluate_ranking`, `evaluate_engagement` and `compare` (bootstrap intervals)~~ (done:
   same `(summary, per_user)` shape for both tasks; task B's Spearman skips undefined users).
4. ~~BPR-MF (`implicit`), with the under-120-minute ablation~~ (done: tuned in
   `task_a_play`; "owned" positives win; §4).
5. ~~The negative-free LFM (Surprise, owned games only: `neg_ratio = 0`), with per-task
   model settings~~ (done: `lfm`, tuned in `task_b_engagement`; §4).
6. ~~Baselines as registered models~~ (done: `item_knn_rank`, `user_knn_rank` for task A,
   `bias_eng` for task B; the global mean stays a notebook baseline; kNN regressors
   rejected, §5).
7. *Optional, if time:* cross-task tests (BPR on B, LFM on A).
8. Re-run on `user_frac = 1.0` for the final numbers: final models only, with the settings
   tuned on the 10% sample (re-tuning on full data is dropped for time and reported as a
   limitation), then score test once per task.
9. *Maybe, for the write-up:* report placement alongside Hit@10 / NDCG@10: Hit@5, median
   rank of the held-out game (overall and among top-10 hits), and the top-1 rate. All come
   from per-user ranks (`1 / mrr`), so no change to `evaluate_ranking`; compute them in the
   notebook for the final models only (medians, not means: misses go out to rank 100).
10. Workbook (`workbook.ipynb`, one section per rubric item) and the ~20-minute video.

## References

- Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008.
- Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009.
- Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018 (Steam data; 100-candidate protocol).
- Koren, Bell & Volinsky, *Matrix Factorization Techniques for Recommender Systems*, IEEE
  Computer 2009 (latent factor models with biases).
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020.
- Sarwar, Karypis, Konstan & Riedl, *Item-Based Collaborative Filtering Recommendation
  Algorithms*, WWW 2001.
- Steam refund policy: https://store.steampowered.com/steam_refunds/