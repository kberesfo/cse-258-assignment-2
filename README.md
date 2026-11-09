# CSE 258 Assignment 2

Group project for CSE 258 (Recommender Systems & Web Mining), Fall 2026.

**Task:** _TBD — predictive task, dataset, and metric go here once chosen._

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

**Code → Codespaces → Create codespace on main.** The devcontainer runs `uv sync` and
`make data` on creation, so the environment and dataset are ready when it opens.

### Local devcontainer

Open the repo in VS Code and choose **Dev Containers: Reopen in Container**. Same setup as
Codespaces.

### Without a container

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make data
```

Select the `.venv` interpreter as the notebook kernel.

---

## Common commands

Everything goes through `make`. Each pipeline stage reruns only when code in `src/` or
`config.toml` changes.

| Command          | What it does                                                   |
| ---------------- | -------------------------------------------------------------- |
| `make data`      | Download the dataset named in `config.toml` into `data/`       |
| `make prep`      | Load, clean, and split the data → `artifacts/`                 |
| `make train`     | Fit all registered models → `artifacts/`                       |
| `make eval`      | Score every model on validation → `results/metrics.json`       |
| `make figures`   | Plots for the notebook and slides → `results/figures/`         |
| `make results`   | All of the above                                               |
| `make final`     | Also score the test split; run once models are frozen          |
| `make diff`      | Compare metrics with the previous run                          |
| `make workbook`  | Execute the notebook → `submit/workbook.html`                  |
| `make check`     | Run the autograder's checks, including the video download      |
| `make submit`    | `final` + `workbook` + `check`                                 |
| `make lint`      | Ruff lint and format check                                     |
| `make fmt`       | Auto-fix lint and formatting                                   |
| `make test`      | Run tests, including a full pipeline run on synthetic data     |

Force a rerun with `make -B train`, which also reruns the stages before it.

The CLI can also be called directly:

```bash
uv run assignment-2 --help
uv run assignment-2 models                          # list registered models
uv run assignment-2 train --model popularity        # train one model
uv run assignment-2 eval --split valid
```

---

## Repository layout

```
.
├── .devcontainer/            # Codespaces / devcontainer config
├── data/                     # raw datasets (gitignored, fetched by scripts/get_data)
├── artifacts/                # regenerable outputs: splits, trained models (gitignored)
├── results/                  # committed outputs: metrics.json, figures/
├── docs/                     # assignment spec
├── notebooks/
│   ├── workbook.ipynb        # the submission: one section per rubric item
│   ├── workbook.py           # jupytext pair of the notebook; review this in PRs
│   └── scratch/<name>/       # personal exploration; never imported from
├── scripts/
│   ├── get_data/             # dataset downloads
│   └── check_submission.py   # the spec's autograder checks
├── src/assignment_2/
│   ├── cli.py                # subcommands: prep, train, eval, figures, models
│   └── modules/
│       ├── config/           # settings, loaded from config.toml
│       ├── core/             # the pipeline every model shares
│       │   ├── datasets.py   # one loader per dataset → the standard table
│       │   ├── data.py       # cleaning, train/valid/test split, load_split
│       │   ├── features.py   # ID maps, sparse matrices, library adapters, negatives
│       │   ├── evaluate.py   # evaluation protocol: scores saved predictions
│       │   ├── metrics.py    # pure metric functions
│       │   ├── plots.py      # EDA and model comparison figures
│       │   └── notebook.py   # display helpers, including show(fn)
│       └── models/           # one file per model family
│           ├── base.py       # save_predictions: the contract with evaluation
│           ├── registry.py   # MODELS and train()
│           └── baselines.py  # trivial, popularity, bias
├── submit/                   # workbook.html (gitignored), video_url.txt
├── tests/
├── config.toml               # the group's dataset, task, split, and hyperparameters
├── Makefile
└── pyproject.toml
```

**Rule of thumb:** logic lives in `src/`, and notebooks only call it and display results.
Graders only see the exported HTML, so use `show(fn)` in the notebook to print the code you
walk through.

---

## Data

Datasets are **not committed**. Until the group picks one, `config.toml` uses `synthetic`, a
small fake dataset that needs no download. ModCloth is included as a worked example.

### Plugging in a dataset

1. Add a download entry to `DATASETS` in `scripts/get_data/__main__.py`
   ([dataset links](https://cseweb.ucsd.edu/~jmcauley/datasets.html)).
2. Write a loader in `modules/core/datasets.py`, using the same key, that returns the standard
   table, and add it to `LOADERS`. `load_modcloth` is the template.
3. Set `dataset` and `task` (`rating`, `binary`, or `ranking`) in `config.toml`.
4. Run `make data results`.

The standard table has one row per interaction, with `user` and `item`, plus `rating` or
`label` as the task requires, and an optional `timestamp` in Unix seconds. Extra columns are
kept for feature-based models. Without timestamps, the split falls back to random.

Set `A2_PATHS__DATA_DIR` to keep data somewhere other than `data/`.

---

## Working together

### Branches and PRs

- Branch from `main` as `<name>/<topic>` (e.g. `kiernan/bias-model`) and merge by PR.
- Run `make lint test` before opening a PR.
- If a PR changes `results/metrics.json`, include the `make diff` output in its description.

### Adding a model

1. Write a function `fit_<name>(cfg)` in a file under `modules/models/`. Train on
   `load_split(cfg, "train")`, then call `save_predictions(cfg, "<name>", predict)`, where
   `predict` takes a table and returns one score per row. The first line of its docstring is
   shown by `assignment-2 models`.
2. Register it in `MODELS` in `modules/models/registry.py`.
3. Run `uv run assignment-2 train --model <name>`, then `uv run assignment-2 eval`.

`modules/models/base.py` defines what a score means for each task. Every model is scored by
`modules/core/evaluate.py` on the same rows, so results are comparable. Don't write separate
evaluation code inside a model file.

Hyperparameters go in `config.toml` under `[models.<name>]`, and `cfg.params("<name>")` reads
them. `features.py` has adapters for `implicit` and `surprise`.

`make eval` only scores validation, so tune against that. Run `make final` once models are
frozen.

### Notebooks

- **`workbook.ipynb` is shared.** Coordinate before editing it, and keep cells short:
  import, call, display.
- `notebooks/scratch/<name>/` is yours. Experiment freely; move anything worth keeping
  into `src/`.
- Notebooks are paired with `.py` files via jupytext (`make sync`). Review the `.py` diff
  in PRs. Clear outputs from scratch notebooks before committing.

### Reproducibility

- Use `cfg.seed` for anything random.
- Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`) and commit `uv.lock`.

---

## Submitting

1. `make submit` builds `submit/workbook.html` and runs the checks.
2. Upload the mp4 to Google Drive, set sharing to **Anyone with the link**, and paste the
   link as the only line of `submit/video_url.txt`.
3. Run `make check` again, then upload `workbook.html` and `video_url.txt` to Gradescope.

Late submissions aren't accepted, because submissions are redistributed for peer grading.
Check Gradescope for the exact deadline.

| Milestone            | Date                       |
| -------------------- | -------------------------- |
| Submission           | Week 10, Tue (Gradescope)  |
| Peer grading         | Week 11                    |