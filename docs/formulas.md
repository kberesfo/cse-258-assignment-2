# Metrics: definitions, and why we use them

How each task is scored, the formulas behind `core/metrics.py` and `core/evaluate.py`, and
why these are the right metrics for this data. Protocol details (splits, candidates, cold
drops): [`plan.md`](plan.md) §2.

**Notation.** $U$ is the set of evaluation users, $u \in U$ one user, $i$ a game.
Every metric is computed **per user, then averaged over users**, so each user counts once
however large their library is.

---

## Task A: ranking (will the user play this game?)

### Setup

For each user $u$, the candidate set $C_u$ has 100 games:

- **1 positive** $i_u^+$: a held-out game the user played for at least 120 minutes
- **99 negatives** $N_u$: games the user owns in no split, sampled from training games
  in proportion to popularity, once, in `prep` (the same rows for every model)

A model scores all 100 and sorts them. $\text{rank}_u \in \{1, \dots, 100\}$ is the
position of $i_u^+$ in that order (1 = top). Ties are broken at random with the same seed
for every model, so a model that outputs constant scores gets a random rank, not rank 1.

### Hit@k

$$
\text{Hit@}k = \frac{1}{|U|} \sum_{u \in U} \mathbb{1}\left[\text{rank}_u \le k\right]
$$

The share of users whose played game is in the top $k$. With one positive per user it
equals Recall@k. **It ignores position inside the top $k$:** rank 1 and rank 10 both count
as a hit.

### NDCG@k

In general, with graded relevance $\text{rel}_{u,r}$ of the game at rank $r$:

$$
\text{DCG@}k(u) = \sum_{r=1}^{k} \frac{\text{rel}_{u,r}}{\log_2(r + 1)},
\qquad
\text{NDCG@}k(u) = \frac{\text{DCG@}k(u)}{\text{IDCG@}k(u)}
$$

where IDCG is the DCG of the best possible ordering. With **one** relevant game the ideal
ordering puts it first ($\text{IDCG} = 1$), so this reduces to:

$$
\text{NDCG@}k(u) =
\begin{cases}
\dfrac{1}{\log_2(\text{rank}_u + 1)} & \text{rank}_u \le k \\[1ex]
0 & \text{otherwise}
\end{cases}
\qquad
\text{NDCG@}k = \frac{1}{|U|}\sum_{u} \text{NDCG@}k(u)
$$

Like Hit@k, but a hit near the top counts more.

### MRR (mean reciprocal rank)

$$
\text{MRR} = \frac{1}{|U|} \sum_{u \in U} \frac{1}{\text{rank}_u}
$$

No cutoff: a game at rank 30 still earns $1/30$. The discount is steep (rank 2 is already
worth half of rank 1), so MRR mostly measures how often the game is at or very near the
top, while still using the whole 100-game list.

### Credit by rank

| $\text{rank}_u$ | Hit@10 | NDCG@10 | MRR |
|---|---|---|---|
| 1 | 1 | 1.000 | 1.000 |
| 2 | 1 | 0.631 | 0.500 |
| 3 | 1 | 0.500 | 0.333 |
| 5 | 1 | 0.387 | 0.200 |
| 10 | 1 | 0.289 | 0.100 |
| 30 | 0 | 0 | 0.033 |

### Relationships worth knowing

- **NDCG@k ≤ Hit@k**, per user and so on average: a miss scores 0 on both, and a hit
  scores 1 on Hit@k but $1/\log_2(\text{rank}+1) \le 1$ on NDCG@k. They are equal only if
  every hit is at rank 1. **Hit@5 ≤ Hit@10** for the same reason.
- **NDCG@10 / Hit@10** is the average NDCG among hits: a rough sense of how high hits land
  (final BPR: 0.388 / 0.619 ≈ 0.63, about the rank-2 value). It is *not* an average
  rank: the discount is nonlinear, so it can't be inverted to one. The same goes for
  $1/\text{MRR}$.
- **MRR is not ordered against the other two.** It is below NDCG@10 at ranks 2–10 (steeper
  discount) and above it past rank 10 (no cutoff).
- **A random ranker** scores, in expectation,

  $$
  \mathbb{E}[\text{Hit@}10] = \tfrac{10}{100} = 0.100, \quad
  \mathbb{E}[\text{NDCG@}10] = \tfrac{1}{100}\sum_{r=1}^{10}\tfrac{1}{\log_2(r+1)} \approx 0.045, \quad
  \mathbb{E}[\text{MRR}] = \tfrac{1}{100}\sum_{r=1}^{100}\tfrac{1}{r} \approx 0.052
  $$

  (measured on validation: 0.090, 0.041, 0.049).

### Placement (optional, for the write-up)

With one positive per user, `ranking_per_user` returns $\text{MRR}_u = 1/\text{rank}_u$,
so each user's exact rank is $1 / \text{MRR}_u$. From it: Hit@5, the **median** rank
(overall and among top-10 hits), and the top-1 rate. Medians, not means: misses go out to
rank 100 and would dominate a mean.

### Why these metrics fit task A

- **The task is top-N recommendation.** A store page shows a short list, so the question is
  whether the game the user goes on to play is on it (Hit@10) and how high (NDCG@10). MRR
  adds the rest of the list, and reacts to changes past the cutoff that Hit@10 and NDCG@10
  can't see.
- **One held-out game per user** (leave-one-out) makes every metric a function of a single
  rank, so all three are easy to read and to compare between models.
- **Comparable to prior work on this data.** SASRec, on a larger Steam review dataset, uses
  Hit@10 and NDCG@10 on the held-out item ranked against 100 sampled negatives (Kang &
  McAuley 2018); we use the same metrics on 1 + 99 candidates.
- **Popularity-weighted negatives** make the test about personalization. With uniform
  negatives the held-out game was ~90× more popular than the negatives, so "recommend
  popular games" found it; with popularity weighting the gap is ~3× (`plan.md` §2).

**Limitations.**

- **Sampled metrics are not full-catalogue metrics.** Krichene & Rendle (2020) show that a
  metric computed on a sampled candidate set can rank two models differently than the same
  metric computed over every item, even in expectation. Our numbers are comparisons on a
  fixed, shared candidate table, not estimates of full-catalogue Hit@10.
- **The protocol partly rewards ownership:** the positive is always owned and the negatives
  never are (`plan.md` §2, and BPR's positives ablation in §4).

---

## Task B: engagement (how long will the user play a game they own?)

### Target

$$
y_{ui} = \log(1 + \text{minutes}_{ui})
$$

Playtime is extremely heavy-tailed: a few users play some games for thousands of hours,
and about a third of owned games have 0 minutes. On raw minutes, squared error would be
dominated by a handful of heavy players; on the log scale, being off by a factor of 2 costs
the same at 1 hour as at 100. $\log(1 + \cdot)$ keeps never-played games at $y = 0$.
Models predict $\hat{y}_{ui}$ on this scale.

### MSE and MAE, per user

Each user $u$ has a held-out set $H_u$ of (up to) 3 owned games:

$$
\text{MSE}(u) = \frac{1}{|H_u|}\sum_{i \in H_u} (\hat{y}_{ui} - y_{ui})^2,
\qquad
\text{MAE}(u) = \frac{1}{|H_u|}\sum_{i \in H_u} |\hat{y}_{ui} - y_{ui}|
$$

and MSE, MAE are their averages over users. MSE punishes large misses most (e.g.
predicting "never played" for a game played 500 hours); MAE is less sensitive to them.
Reporting both shows whether a gain comes from fixing a few large errors or from many small
ones.

### Within-user Spearman

For user $u$, rank the 3 true targets and the 3 predictions within that user (ties get
their average rank, as in `scipy.stats.spearmanr`), giving $R(y_{ui})$ and
$R(\hat{y}_{ui})$. Then:

$$
\rho_u = \text{corr}\left(R(y_{u\cdot}),\; R(\hat{y}_{u\cdot})\right),
\qquad
\bar{\rho} = \frac{1}{|U_\rho|}\sum_{u \in U_\rho} \rho_u
$$

the Pearson correlation of the ranks. $\rho_u = 1$ means the model orders the user's games
exactly right; $-1$, exactly backwards. It is **undefined** when the user's 3 targets are
all equal (nearly always all 0 minutes) or the model gives all 3 the same score; those
users are skipped, and $U_\rho$ is the users where it is defined (94% of validation users
for models that vary by game).

### Why these metrics fit task B

- **MSE on log minutes is the natural loss** for a model that predicts
  $\mu + b_u + b_i + \gamma_u \cdot \gamma_i$ by least squares: we evaluate what the
  model is trained to minimize.
- **MSE alone can't see personalization; Spearman can.** Within one user, $\mu + b_u$ is
  the same for every game, so it can't change the order of the user's games:

  $$
  \hat{y}_{ui} - \hat{y}_{uj} = (b_i - b_j) + \gamma_u \cdot (\gamma_i - \gamma_j)
  $$

  A bias model ($\gamma = 0$) therefore orders every user's games by $b_i$, the same order
  for everyone, which is why it ties the item mean on Spearman (0.540 vs 0.536) while
  improving MSE a lot (5.03 vs 5.59). **Spearman above the bias model measures what the
  latent factors $\gamma_u \cdot \gamma_i$ add**; MSE mostly rewards getting the biases
  right.
- **Spearman works for models that don't predict minutes.** It depends only on order, so it
  doesn't change under any increasing transformation of the scores. That is what lets us
  score task A's ranking models (BPR) on task B: their scores aren't in minutes, but their
  order within a user is still meaningful. The bar is owner popularity (0.165).
- **User-weighted.** Every user contributes 3 held-out games, while training rows are
  dominated by users with huge libraries. Averaging per user keeps the metric about typical
  users, as the split intends (`plan.md` §2).

---

## Uncertainty: bootstrap intervals (both tasks)

Let $m_u$ be a metric's value for user $u$ (e.g. Hit@10 or per-user MSE), and
$\bar{m} = \frac{1}{|U|}\sum_u m_u$ the reported score.

**95% interval for one model.** Draw $B = 1{,}000$ resamples of $|U|$ users *with
replacement*; for resample $b$, compute the mean $\bar{m}^{(b)}$. The interval is the 2.5th
and 97.5th percentiles of $\bar{m}^{(1)}, \dots, \bar{m}^{(B)}$ (`bootstrap_ci`). It shows
how much the score would move with a different sample of users. Users where the metric is
undefined (NaN Spearman) are skipped.

**Paired interval for a difference.** To compare models $A$ and $B$, use the same resampled
users for both, which is the same as bootstrapping the per-user differences
$d_u = m_u^A - m_u^B$ (`compare`). If the interval on $\bar{d}$ excludes 0, the difference
is real at the 95% level (the "real?" column). Pairing makes the interval narrower because
user-level difficulty cancels:

$$
\text{Var}(\bar{m}^A - \bar{m}^B) = \text{Var}(\bar{m}^A) + \text{Var}(\bar{m}^B)
- 2\,\text{Cov}(\bar{m}^A, \bar{m}^B)
$$

A user who is easy for one model is usually easy for the other, so the covariance is large
and positive. Only users where both models have a defined value are paired.

**Seeds are a separate source of noise.** The bootstrap covers *which users* were sampled,
not training randomness. Grids report the mean and standard deviation over 3 seeds; for
BPR, seed noise is ~0.001–0.002 NDCG@10 (`plan.md` §4).

---

## References

- Järvelin & Kekäläinen, *Cumulated Gain-Based Evaluation of IR Techniques*, ACM TOIS 2002
  (NDCG).
- Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018 (Steam data;
  Hit@10 / NDCG@10 against 100 sampled negatives).
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020.
- Efron & Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall 1993.