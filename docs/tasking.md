# Tasking: finishing A2

Due **Monday**; we submit **Sunday evening**. Modeling is done for both tasks and the
full-data results are committed, so what's left is the workbook (graded on 5 sections:
predictive task, EDA, modeling, evaluation, related work) and the ~20-minute video.

Read first: [`README.md`](../README.md), [`docs/plan.md`](plan.md) (how we got here, final
results in Section 4), [`docs/formulas.md`](formulas.md) (metrics).

## Status

- [x] Task A (BPR) and task B (LFM) tuned and merged
- [x] Full-data run, test scored once; everything committed to `results/`
- [x] Workbook rebuilt to read only `results/` (runs anywhere, no data needed)
- [x] Workbook Section 3 Modeling and Section 4 Evaluation drafted
- [ ] Workbook Section 1, Section 2, Section 5
- [ ] Video
- [ ] `make workbook`, `make check`, submit

## Who's doing what

**@kberesfo: modeling, evaluation, final run, submission**
- [x] Merge task B
- [x] Full-data run (final models only), test scored once per task
- [x] Workbook Section 3 Modeling and Section 4 Evaluation
- [ ] Review the other sections as they land; final `make workbook`, `make check`, upload

**@hetasj: predictive task + EDA**
- [ ] Workbook Section 1 Predictive task: the two tasks (A: rank the game a user will play; B:
  predict log playtime of owned games), splits, metrics and why (`docs/plan.md` Section 1–2,
  `docs/formulas.md`)
- [ ] Workbook Section 2 EDA: the full-data tables and plots are already in the section (library
  size, game popularity, playtime, popularity vs engagement, reviews per year,
  metadata/bundle coverage); write the context, pre-processing and what each plot implies
  for modeling. Note: on the full data, owners vs median playtime has Spearman ≈ 0.02 (the
  10% sample gave 0.19); describe the split plot (play rate vs playtime among players).

**@Amir-Mirkazemi: related work + consistency check**
- [ ] Workbook Section 5 Related work: a paragraph each on SASRec (same Steam data; 100 sampled
  negatives), BPR, implicit-feedback CF (Hu et al.), item-based CF, latent factor models
  (Koren et al.), sampled-metric caveats (Krichene & Rendle). References: `docs/plan.md`.
  Say how our results match or differ.
- [ ] Run the workbook top to bottom on a clean kernel (`make workbook`) and flag anything
  broken or inconsistent (numbers in the text vs the tables)

**@coyote-jadey: video**
- [ ] Slide outline for ~20 min following the 5 sections
- [ ] Figures and tables from `results/` (see below), not from re-running models
- [ ] Schedule recording for Sat/Sun; each of us records our own section; edit, upload to
  Google Drive (Anyone with the link), put the link as the only line of
  `submit/video_url.txt`

## How to work on it

- **No full data needed.** Everything to plot is committed in `results/`:
  `metrics.json`, `comparisons.csv`, `summaries/*.csv`, `figures/*.png` (README → "Results").
  From a notebook: `pd.read_csv("../results/summaries/per_game.csv")`.
- **Use the default container**, not the GPU one (the GPU container runs on the full data).
- **One branch per section** (`task/workbook-eda`, `task/workbook-related`, ...), and edit
  only your section of `workbook.ipynb`, so merges don't conflict. Keep cells short.
- Check your section renders: `make workbook` → `submit/workbook.html`.
- Don't re-run `make results` / `make final`: test has been scored, and `results/` is final.

## Timeline

| When | What |
|---|---|
| Wed | ~~Final test numbers in the repo~~ (done) |
| Thu | Thanksgiving: optional |
| **Sat noon** | All workbook sections drafted and merged |
| Sat–Sun | Record the video; everyone reviews the whole workbook |
| **Sun evening** | Submit (Kiernan has a midterm Monday) |

Shout in the group chat if you want a different piece or can't make the timeline.