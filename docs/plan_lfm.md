# Plan: Surprise latent factor model (LFM) for Steam game ranking

Base model for the ranking task. A pointwise latent factor model (Surprise `SVD`) trained
to predict how much a user plays a game, then used to rank each user's fixed candidate
games. BPR is added later as the pairwise comparison.

## 1. Task and evaluation (fixed by `prep`)

- **Data:** `load_prepped(get_config())` → `interactions_{train,valid,test}`,
  `interactions_{valid,test}_candidates`.
- **Candidates:** per user, their held-out games (`label = 1`) + 99 sampled games they own
  in no split (`label = 0`), drawn from train games only. Every model scores these exact rows.
- **Metrics:** Recall@k and NDCG@k (primary), Hit@k and MRR (secondary, SASRec-comparable),
  per-user AUC (what BPR optimizes). Bootstrap 95% CIs over users; `compare()` for the
  CI of the difference vs a baseline.
- **Protocol:** tune on `valid`, report `test` once.

## 2. Training target

| Row | Target |
|---|---|
| owned game (train split) | `1 + log1p(playtime_forever)` |
| sampled non-owned game | `0` |

- The `+1` keeps owned-but-unplayed (0 minutes) distinct from not-owned, so the model
  learns both ownership and engagement.
- `Reader(rating_scale=(0, max_target))`.
- Write-up: this is a pointwise regression on sampled negatives; state the ratio and
  sampling rule in the Modeling section.

## 3. Training negatives

Sampled **once** per run, with their own seed (not the candidate seed):

- From **train games** the user owns in **no** split.
- **Excluding** that user's valid/test candidate negatives, so the model never trains on
  the exact games it is evaluated against.
- **Ratio** `neg_ratio` negatives per positive (start at 4; sweep 1, 2, 4, 8).
- Uniform over train games to start; popularity-weighted is an ablation.

## 4. Model

- `surprise.SVD(biased=True)`: prediction = μ + b_u + b_i + p_u · q_i.
  - `b_i` absorbs popularity; `p_u · q_i` adds personalization on top. This is why it
    should beat the popularity baseline.
- Hyperparameters in `config.toml` under `[models.svd]`, read via `cfg.params("svd")`:

```toml
[models.svd]
n_factors = 50
n_epochs = 20
lr_all = 0.005
reg_all = 0.02
neg_ratio = 4
```

## 5. Scoring the candidates

- Don't call `model.predict()` per row (slow, and it clips to the rating scale → ties).
- Map IDs with `trainset.to_inner_uid` / `to_inner_iid`, then compute
  `μ + bu[u] + bi[i] + (pu[u] * qi[i]).sum(1)` with NumPy on the whole candidate table.
- Every candidate game is a train game by construction; any unmapped ID is a bug, so
  assert none are missing rather than defaulting.

## 6. Experiments (each a row in the ablation table)

| # | Model | Question it answers |
|---|---|---|
| 0 | Popularity (train counts) | Baseline every model must beat |
| 1 | SVD, ownership only (owned = 1, negatives = 0) | Does personalization beat popularity? |
| 2 | SVD, `1 + log1p(playtime)` target | Does playtime add signal beyond ownership? |
| 3 | Sweep `neg_ratio`, `n_factors`, `reg_all` | Sensitivity; pick settings on `valid` |
| 4 | Popularity-weighted training negatives | Does a harder negative distribution help? |
| 5 | BPR (pairwise) | Pointwise vs pairwise objective for ranking |

Report each with CIs, and each improvement as a `compare()` difference vs the row it builds on.

## 7. Scale

- Develop on `[sample] user_frac = 0.1`; final numbers on `1.0`.
- Full data: ~5M positives × (1 + `neg_ratio`) = 10–25M training rows, and Surprise holds
  them in Python dicts. Time the 10% run first; if the full run is too slow or too large,
  lower `neg_ratio` or report on a larger sample (e.g. 0.25) and say so.

## 8. Code layout

- `models/ranking/base.py`: `IdIndex`, `interaction_matrix`, and the Surprise trainset
  builder (renamed to `user_id` / `item_id`).
- `models/ranking/svd.py`: `train_negatives(...)`, `fit_svd(cfg, train)`, `score_svd(model, candidates)`.
- `models/ranking/popularity.py`: baseline.
- Register each in `MODELS`; evaluation goes through `core/evaluate.py` only.

## 9. Write-up notes (for the presentation sections)

- **Modeling:** why a pointwise target on sampled negatives; why `+1` on owned rows;
  bias terms vs factors; tradeoffs vs BPR (objective, sampling, cost).
- **Evaluation:** fixed candidates, train-only negatives, exclusion of candidate
  negatives from training, tie-breaking, bootstrap over users, sampled-metric caveat
  (Krichene & Rendle, KDD 2020).
- **Related work:** compare protocol (not exact numbers) with SASRec (Kang & McAuley,
  ICDM 2018), which ranks against 100 sampled negatives on Steam.

## Checklist

- [ ] Training-negative sampler with exclusions + its own seed
- [ ] Target column and Surprise trainset builder
- [ ] `fit_svd` + vectorized `score_svd`
- [ ] Popularity baseline in the same interface
- [ ] Experiments 0–2 on the 10% sample
- [ ] Sweep (3) on `valid`; record chosen settings in `config.toml`
- [ ] Ablations 4–5
- [ ] Full-data run; test numbers once
