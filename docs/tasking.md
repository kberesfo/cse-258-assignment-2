Hey all, A2 is due Monday and modeling is done for both tasks, so the rest is the workbook (graded on 5 sections: predictive task, EDA, modeling, evaluation, related work) and the ~20 min video. Proposed split:

**Kiernan: modeling + evaluation + final run**
- Merge task B (tonight)
- Full-data run: final models only, score test once per task (by Wed)
- Workbook §3 Modeling and §4 Evaluation (tables/figures from task_a_play and task_b_engagement, metrics from docs/formulas.md)
- Final `make check` and submission

**[Task 1]: predictive task + EDA**
- Workbook §1 Predictive task: the two tasks (A: rank the game a user will play; B: predict log playtime of owned games), splits, metrics and why (docs/plan.md §1–2, docs/formulas.md)
- Workbook §2 EDA: key figures from eda.ipynb, explore_play and explore_engagement (playtime distribution, never-played share, popularity, ownership vs play)

**[Task 2]: related work + consistency check**
- Workbook §5 Related work: a paragraph each on SASRec (same Steam data and 100-candidate protocol), BPR, implicit-feedback CF (Hu et al.), item-based CF, latent factor models, sampled-metric caveats (Krichene & Rendle). References are in docs/plan.md
- Run the workbook top to bottom on a clean kernel and flag anything broken or inconsistent

**[Task 3]: video**
- Slide outline for ~20 min following the 5 sections, with figures pulled from the notebooks
- Schedule recording for Sat/Sun; each of us records our own section; edit, upload, put the link in submit/video_url.txt

**~Timeline**
- Wed: final test numbers in the repo
- Sat noon: all workbook sections drafted (one branch/PR per section to avoid notebook merge conflicts)
- Sat–Sun: record video, everyone reviews the workbook
- Sun evening: submit (I have a midterm Monday, so not leaving it to the last day)

Everything you need to read first: README.md, docs/plan.md, docs/formulas.md. Shout if you want a different piece or can't make the timeline.