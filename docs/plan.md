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
  (Jaccard similarity to the user's library) beat popularity by +0.21 Hit@10 on ownership,
  but only by about +0.09 on play. Games sold together are owned together whether or not
  the user wants them; predicting play removes most of that.
- **Playtime helps once play is the target.** Ranking by how many users *played* a game
  beats ranking by how many *own* it (Hit@10 0.405 vs 0.357 at 120 minutes), and the lead
  grows with the threshold (+0.028 at 5 min, +0.063 at 240). On the ownership task the two
  are identical.

### Why 120 minutes

- It's Steam's refund window: a refund is offered within two weeks of purchase and with
  under two hours of playtime. Past 2 hours, the user has kept and committed to the game.
- It keeps 91% of validation users (5,946 of 6,535 for ownership; `explore_play` §2).
- The data doesn't pick a threshold: above 0 minutes, Jaccard's margin over the best
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
- **Metrics:** Hit@10, NDCG@10, MRR per user, averaged. 95% bootstrap intervals over
  users, and paired bootstrap intervals on the difference between a model and its
  baseline.

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
- Tune on validation; score test once.

## 3. Models: two ways to adapt latent factors to implicit feedback

Both have the same form, μ + b_u + b_i + γ_u · γ_i. They differ in what they're trained to do.

| | **BPR-MF** | **LFM on log-hours** |
|---|---|---|
| Trained on | pairs (u, played game i, non-owned game j) | owned games only |
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

- Owned games under 120 min, for BPR: included in the matrix as positives ("owned beats
  not owned"), or left out of it, in which case `implicit` can sample them as negatives
  (*hard negatives*: owned but not played).
- LFM target: log(1 + minutes) vs played ≥ 120 min (0/1).
- Two-part (hurdle) model for task B: P(played) × E[log minutes | played]. 31% of held-out
  targets are 0, but an item-level hurdle equals the item mean (MSE 5.586 vs 5.587); it can
  only help once P(played) depends on the user, so it belongs with the models.
- Implicit ALS (Hu, Koren & Volinsky 2008) with playtime as confidence, as a third
  latent-factor variant if time allows.

## 4. Baselines

All on validation. Task A is scored on the cached `play_valid_candidates`
(`explore_play` §4); task B on `own_valid` (`explore_own` §4).

**Task A** (120 min, 5,946 users)

| Baseline | Hit@10 | NDCG@10 | MRR |
|---|---|---|---|
| random | 0.090 | 0.041 | 0.049 |
| play rate (shrunk, a = 10) | 0.350 | 0.192 | 0.169 |
| owner popularity | 0.357 | 0.213 | 0.190 |
| play popularity | 0.405 | 0.246 | 0.220 |
| **Jaccard to played history** | **0.497** | **0.303** | **0.264** |

Jaccard is the bar for BPR.

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

## 5. Rejected or deferred

- **Treating non-owned games as confident zeros** (pointwise SVD with sampled negatives):
  it trains the model to treat the very games it should recommend as disliked. The current
  Surprise models still do this (`neg_ratio = 4`) until the negative-free LFM is in.
- **Low playtime as "dislike"** for the ownership task: playtime is a snapshot with no
  purchase date, so "under 2 hours" can mean disliked, not started yet, or just bought.
  Refunded games never appear in the data at all.
- **Negative reviews as negatives:** the cleanest dislike signal, but ~11% of ~6K reviews,
  far too few against 5M ownership rows. Possible as an EDA check instead.
- **Predicting the review `recommend` flag:** 89.1% positive and few reviews.
- **Bundle and metadata features:** bundles cover 14% of games; metadata is missing for
  several of the most-owned titles. Possible later ablations.

## 6. Next steps

1. ~~Move the 120-minute play split into `prep`~~ (done: `[split.play] min_minutes = 120`,
   `[split.engagement]` for task B).
2. ~~Threshold table, task B tie check, held-out vs train, baselines on the cache~~ (done:
   `explore_play`, `explore_own`).
3. `evaluate_scores` and `compare` (bootstrap intervals) for task A; MSE / MAE /
   within-user Spearman for task B.
4. BPR-MF (`implicit`), with the under-120-minute ablation.
5. The negative-free LFM (Surprise, owned games only: `neg_ratio = 0`), with per-task
   model settings; task B baselines (global mean, bias model).
6. Cross-task tests (BPR on B, LFM on A).
7. Re-run everything on `user_frac = 1.0` for the final numbers.

## References

- Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008.
- Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009.
- Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018 (Steam data; 100-candidate protocol).
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020.
- Steam refund policy: https://store.steampowered.com/steam_refunds/