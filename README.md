# CSE 258 Assignment 2

Group project for CSE 258 (Recommender Systems & Web Mining), Fall 2026.

**Tasks:** game recommendation on Steam, from implicit feedback (ownership + playtime) in the
Steam v1 Australian users data.

- **A (main, ranking):** given a user's library, rank 100 candidates so the held-out game they
  **played for at least 2 hours** comes first. The other 99 are popularity-weighted games they
  don't own. Hit@10, NDCG@10, MRR, with bootstrap confidence intervals.
- **B (engagement):** given that a user owns a game, predict how long they play it, as
  log(1 + minutes). MSE, MAE, and within-user Spearman.

Why these tasks, the protocol, and the baselines to beat: [`docs/plan.md`](docs/plan.md).

Deliverables (see [`docs/`](docs/) for the full spec):

1. `workbook.html`: the notebook, exported with outputs. Graders read it without running it.
2. `video_url.txt`: a link to a ~20-minute presentation (Google Drive mp4 or YouTube).
3. Peer grading report, due about a week after submission.

The presentation and notebook are graded together on five sections: **predictive task,
exploratory analysis, modeling, evaluation, related work** (5 marks each, plus 4 for peer
grading). The notebook follows that order.

---

## Getting started

### GitHub Codespaces (recommended)

**Code → Codespaces → Create codespace on main.** On creation the devcontainer runs
`uv sync --locked && make setup`, which downloads the Steam data and builds the prepped
cache, so `load_prepped(get_config())` works as soon as it opens.

### Local devcontainer

Open the repo in VS Code and choose **Dev Containers: Reopen in Container**. Same setup as
Codespaces.

### Without a container

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make setup
```

Select the `.venv` interpreter as the notebook kernel.

---

## Common commands

Everything goes through `make`. `prep` reruns only when code in `src/`, `config.toml`, or the
downloaded data changes.

| Command          | What it does                                                        |
| ---------------- | ------------------------------------------------------------------- |
| `make setup`     | `data` + `prep`: everything a fresh environment needs               |
| `make data`      | Download the dataset named in `config.toml` into `data/`            |
| `make prep`      | Sample users, split, build candidate tables → `artifacts/steam/<tag>/` |
| `make train`     | Fit all registered models on task A → `artifacts/steam/<tag>/models/play/` |
| `make eval`      | Score models on validation → `results/metrics.json` *(not implemented yet)* |
| `make figures`   | Plots for the slides *(not implemented yet; figures come from notebooks)* |
| `make results`   | `prep` + `train` + `eval` + `figures`                               |
| `make final`     | Also score the test split; run once models are frozen               |
| `make workbook`  | Execute the notebook → `submit/workbook.html`                       |
| `make check`     | Run the autograder's checks, including the video download           |
| `make submit`    | `final` + `workbook` + `check`                                      |
| `make lint`      | Ruff lint and format check                                          |
| `make fmt`       | Auto-fix lint and formatting                                        |
| `make test`      | Run tests                                                           |

Force a rerun with `make -B prep`.

The CLI can also be called directly:

```bash
uv run assignment-2 --help
uv run assignment-2 models                     # list registered models
uv run assignment-2 train --model svd          # train one model (default: all) on task A
uv run assignment-2 train --task engagement    # task B; models go to models/engagement/
uv run assignment-2 prep --seed 1              # any setting can be overridden per run
```

---

## Repository layout

```
.
├── .devcontainer/            # Codespaces / devcontainer config
├── config.toml               # the group's settings: dataset, sample, split, evaluation, models
├── data/                     # raw datasets (gitignored, fetched by scripts/get_data)
├── artifacts/                # regenerable outputs: prepped cache, trained models (gitignored)
├── results/                  # committed outputs: metrics.json, figures/
├── docs/                     # assignment spec, plans
├── notebooks/
│   ├── workbook.ipynb            # the submission: one section per rubric item
│   ├── eda.ipynb                 # exploratory analysis feeding the EDA section
│   ├── explore_play.ipynb        # task A: threshold choice and baselines
│   ├── explore_engagement.ipynb  # task B: holdout checks, metric coverage, baselines
│   ├── task_a_play.ipynb         # task A models: BPR-MF, ablations, LFM on A
│   ├── task_b_engagement.ipynb   # task B models: LFM, hurdle, BPR on B
│   └── scratch/<name>/           # personal exploration; never imported from
├── scripts/
│   ├── get_data/             # dataset registry + downloader
│   └── check_submission/     # the spec's autograder checks
├── src/assignment_2/
│   ├── cli.py                # subcommands: prep, train, eval, figures, models
│   └── modules/
│       ├── config/           # settings (pydantic-settings), loaded from config.toml
│       ├── data/             # loading, sampling, splitting, candidate tables, cache
│       │   ├── steam.py      # Steam loader: raw files -> tidy tables
│       │   ├── split.py      # per-user split (leave-n-out or fraction)
│       │   ├── candidates.py # fixed ranking candidates (positives + sampled negatives)
│       │   ├── prep.py       # prep stage (both tasks), meta.json, load_prepped
│       │   └── tasks.py      # train_tables(): a task's training tables, leaks removed
│       ├── core/
│       │   ├── metrics.py    # per-user ranking metrics (Hit, NDCG, MRR), MSE, MAE
│       │   ├── evaluate.py   # bootstrap CIs and model comparison (in progress)
│       │   └── plots.py      # shared figure style, save(), log_hist()
│       └── models/
│           ├── registry.py   # MODELS and build()
│           └── ranking/      # ScoringModel interface, TrainData, Surprise models
├── tests/
├── Makefile
└── pyproject.toml
```

**Rule of thumb:** logic lives in `src/`, and notebooks only call it and display results.

---

## Data

Datasets are **not committed**. `config.toml` selects the dataset (`steam`), and
`make data` downloads it (about 80 MB):

- `australian_users_items`: ownership + playtime, one row per (user, game)
- `australian_user_reviews`: reviews with a recommend flag and post date
- `steam_games`: game metadata (genres, tags, price, release date)
- `bundle_data`: bundles and their games

```bash
uv run python scripts/get_data --list    # status of every dataset entry
```

### Prepped cache

`make prep` builds everything models and notebooks use, in `artifacts/steam/<tag>/`
(`<tag>` is `users_0.1` for the 10% dev sample, `full` for all users):

| Table                                  | Contents                                                     |
| -------------------------------------- | ------------------------------------------------------------ |
| `play_{train,valid,test}`              | task A: one played (≥ 120 min) game per user held out for valid and test; train also keeps owned games under 120 min |
| `play_{valid,test}_candidates`         | each held-out game + 99 sampled non-owned games (`label`)    |
| `own_{train,valid,test}`               | task B: 3 owned games per user held out for valid and test   |
| `owned`                                | every owned (user, game) with playtime, unsplit (for EDA)    |
| `reviews`, `items`, `bundles`          | supporting tables (reviews unsplit; filtered per task)       |
| `meta.json`                            | the settings that built the cache + dataset statistics       |

Read it with:

```python
from assignment_2.modules.config import get_config
from assignment_2.modules.data import load_prepped, read_meta, train_tables

cfg = get_config()
t = load_prepped(cfg)  # refuses a cache built with different settings
tables = train_tables(t, "play")  # or "engagement": training tables, held-out pairs removed
```

If `config.toml` changes (seed, `[sample]`, `[split.*]`, `[evaluation]`), `load_prepped` raises
"run `make prep`" rather than returning stale splits.

### Development sample

`[sample] user_frac = 0.1` keeps a fixed, hash-based 10% of users with their whole
libraries, so `prep` takes seconds. Figures saved from a sample get the tag in their file name
(`games_per_user.users_0.1.png`). **Set `user_frac = 1.0` for final results.**

---

## Evaluation protocol

Full details and the reasoning behind them: [`docs/plan.md`](docs/plan.md) §2.

**Task A (`[split.play]`)**
- **Split:** leave-one-out per user among games played ≥ 120 minutes (`min_minutes`): one for
  validation, one for test. Users need at least 3 played games. Owned games under 120 minutes
  stay in train and are never held out.
- **Candidates:** each held-out game is ranked against 99 games the user owns in no split,
  drawn from train games in proportion to popularity (`[evaluation]`) and sampled once in
  `prep`, so every model scores the same rows.
- **Metrics:** Hit@10, NDCG@10, MRR per user, averaged; 95% bootstrap intervals over users, and
  bootstrap intervals on the *difference* between a model and its baseline.

**Task B (`[split.engagement]`)**
- **Split:** 3 owned games per user held out for validation and 3 for test (users need at least
  7 owned games), including never-played games (target 0).
- **Metrics:** MSE and MAE on log(1 + minutes); within-user Spearman over each user's held-out
  games, averaged over users where it's defined (~94%).

**Both**
- Held-out games never seen in train are dropped (`drop_cold`).
- A review of a held-out game reveals the answer, so `train_tables` drops reviews of each
  task's held-out pairs, and lists the pairs in `exclude` so no model samples them as
  training negatives.
- Each model trains on its own task's split; tune on `valid`, score `test` once (`make final`).

---

## Working together

### Branches and PRs

- Branch from `main` as `<type>/<topic>` (e.g. `task/eda`, `chore/hot-fix`) and merge by PR.
- Run `make lint test` before opening a PR.
- If a PR changes `results/metrics.json`, include the `make diff` output in its description.

### Adding a model

1. Subclass `ScoringModel` (`modules/models/ranking/base.py`), or `SurpriseModel` for a
   Surprise algorithm:
   - `fit(self, data: TrainData) -> Self` receives one task's training tables from
     `train_tables` (train interactions, items, bundles, reviews with held-out pairs removed)
     plus `exclude`, the task's held-out pairs, never to use as training negatives.
   - `predict(self, rows) -> np.ndarray` returns one score per row, in order. For task A only
     the order matters (higher = more likely to be played); for task B it's the predicted
     log(1 + minutes).
   - The first line of the class docstring is shown by `assignment-2 models`.
2. Register it in `MODELS` in `modules/models/registry.py`.
3. Put hyperparameters in `config.toml` under `[models.<name>]`; they're passed to the
   constructor by name, plus `seed` if the constructor accepts one.
4. Run `uv run assignment-2 train --model <name>` (add `--task engagement` for task B).

Models never sample their own evaluation negatives or split the data: they read the prepped
cache, so every model is compared on identical rows.

### Notebooks

- **`workbook.ipynb` is shared.** Coordinate before editing it, and keep cells short:
  import, call, display.
- `notebooks/scratch/<name>/` is yours. Experiment freely; move anything worth keeping
  into `src/`. Clear outputs from scratch notebooks before committing.
- Start each notebook with `use_style()` from `core/plots.py`, and save figures with `save()`.
- `jupytext.toml` pairs notebooks with `py:percent` scripts. To build or rebuild a notebook
  from a `.py`: `uv run jupytext --to ipynb notebooks/<name>.py`.
- After editing `config.toml` while a kernel is running, call `get_config.cache_clear()`. After
  pulling changes to `src/` (especially `config/`), restart the kernel.

### Reproducibility

- Use `cfg.seed` for anything random.
- Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`) and commit `uv.lock`.

---

## Submitting

1. Set `[sample] user_frac = 1.0`, then `make submit` builds `submit/workbook.html` and runs
   the checks.
2. Upload the mp4 to Google Drive, set sharing to **Anyone with the link**, and paste the
   link as the only line of `submit/video_url.txt`.
3. Run `make check` again, then upload `workbook.html` and `video_url.txt` to Gradescope.

Late submissions aren't accepted, because submissions are redistributed for peer grading.
Check Gradescope for the exact deadline.

| Milestone            | Date                       |
| -------------------- | -------------------------- |
| Submission           | Week 10, Tue (Gradescope)  |
| Peer grading         | Week 11                    |