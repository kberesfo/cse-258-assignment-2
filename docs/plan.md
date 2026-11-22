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
  beats ranking by how many *own* it (Hit@10 0.399 vs 0.351 at 120 minutes). On the
  ownership task the two are identical.

### Why 120 minutes

- It's Steam's refund window: a refund is offered within two weeks of purchase and with
  under two hours of playtime. Past 2 hours, the user has kept and committed to the game.
- It keeps 91% of evaluated users (5,949 vs 6,534 for ownership).
- Every baseline scores higher than at 5 minutes, and Jaccard's margin over popularity is
  about the same (+0.091 vs +0.094).
- **5 minutes is kept as a robustness check** ("played at all, excluding launch-only
  sessions").

## 2. Evaluation protocol (task A)

- **Split:** leave-one-out per user among *played* (≥ 120 min) games: one for validation,
  one for test. Users need at least 3 played games. Owned games under 120 minutes stay in
  training as context and are never held out.
- **Candidates:** the held-out game + 99 games the user owns in **no** split, drawn from
  training games and **sampled in proportion to popularity**, once, in `prep`, so every
  model ranks the same rows.
  - *Why popularity-weighted:* with uniform negatives the held-out game was ~90× more
    popular than the negatives (median owners 630 vs 7), so popularity alone found it.
    Popularity weighting reduces that to ~2.4×, so the test measures personalization.
- **Owned but under 120 minutes:** neither a positive nor a negative in evaluation.
- **Metrics:** Hit@10, NDCG@10, MRR per user, averaged. 95% bootstrap intervals over
  users, and paired bootstrap intervals on the difference between a model and its
  baseline.
- **Leakage guards:** no training negative is a validation or test candidate; training
  reviews of held-out games are removed before any model sees them.
- Tune on validation; score test once.

## 3. Models: two ways to adapt latent factors to implicit feedback

Both have the same form, μ + b_u + b_i + γ_u · γ_i. They differ in what they're trained to do.

| | **BPR** | **LFM on log-hours** |
|---|---|---|
| Trained on | pairs (u, played game i, non-owned game j) | owned games only |
| Objective | score(i) > score(j) (pairwise, ranking) | squared error on log(1 + minutes) |
| Assumption about non-owned games | ranks *below* played games (relative) | **none**: never seen in training |
| Native task | A (ranking) | B (engagement) |

The point of the comparison: **"not owned" is not "disliked."** A non-owned game may be
unknown, unaffordable, or exactly what the user would love, which is what a recommender is
for. BPR only assumes it ranks below a played game; the LFM makes no assumption at all.
Each model is tested on both tasks:

- **BPR on task B:** its scores aren't in hours, so we use within-user rank correlation
  (does it order a user's owned games by playtime?).
- **LFM on task A:** rank candidates by predicted hours (does predicting engagement
  transfer to recommending?). Its limitation: it never saw a non-owned game in training.

### Training choices to ablate

- Owned games under 120 min, for BPR: left out, or used as negatives (*hard negatives*:
  owned but not played).
- LFM target: log(1 + minutes) vs played ≥ 120 min (0/1).
- Implicit ALS (Hu, Koren & Volinsky 2008) with playtime as confidence, as a third
  latent-factor variant if time allows.

## 4. Baselines

| Task A (Hit@10, valid, 120 min) | | Task B (MSE on log1p minutes) |
|---|---|---|
| random | 0.090 | global mean |
| owner popularity | 0.351 | bias model (μ + b_u + b_i) |
| play rate (shrunk) | 0.345 | |
| play popularity | 0.399 | |
| **Jaccard to played history** | **0.490**, the bar for BPR | |

## 5. Rejected or deferred

- **Treating non-owned games as confident zeros** (pointwise SVD with sampled negatives):
  it trains the model to treat the very games it should recommend as disliked.
- **Low playtime as "dislike"** for the ownership task: playtime is a snapshot with no
  purchase date, so "under 2 hours" can mean disliked, not started yet, or just bought.
  Refunded games never appear in the data at all.
- **Negative reviews as negatives:** the cleanest dislike signal, but ~10% of ~6K reviews,
  far too few against 5M ownership rows. Possible as an EDA check instead.
- **Predicting the review `recommend` flag:** 89.5% positive and few reviews.
- **Bundle and metadata features:** bundles cover 14% of games; metadata is missing for
  several of the most-owned titles. Possible later ablations.

## 6. Next steps

1. Move the 120-minute play split into `prep` (`[split] min_minutes = 120`), keeping the
   ownership split for task B.
2. `evaluate_scores` and `compare` (bootstrap intervals).
3. BPR (`implicit`) and the negative-free LFM (Surprise, owned games only).
4. Task B baselines: global mean and bias model.
5. Re-run everything on `user_frac = 1.0` for the final numbers.

## References

- Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008.
- Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009.
- Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018 (Steam data; 100-candidate protocol).
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020.
- Steam refund policy: https://store.steampowered.com/steam_refunds/