# CSE 258 Assignment 2
#   make setup      download + prep: everything a fresh container needs
#   make data       download the dataset named in config.toml -> data/
#   make prep       load + clean + split data          -> artifacts/
#   make train      fit baselines and models           -> artifacts/
#   make eval       score models on the validation split -> results/metrics.json
#   make figures    plots for the notebook and slides  -> results/figures/
#   make results    prep + train + eval + figures (each step reruns only if inputs changed)
#   make final      also score the test split; run this once models are frozen
#   make diff       metrics now vs the latest snapshot in results_history/
#   make sync       pair/sync notebooks with jupytext
#   make workbook   execute notebooks/workbook.ipynb   -> submit/workbook.html
#   make check      verify the submission files the way the autograder does
#   make submit     final + workbook + check
#   make lint / fmt / test

RUN      ?= uv run
CLI      ?= $(RUN) assignment-2

NB       := notebooks/workbook.ipynb
DATA     := data
ART      := artifacts
RES      := results
HIST     := $(RES)_history
SUB      := submit
STAMP    := $(ART)/.stamps
SRC      := $(shell find src -name '*.py' 2>/dev/null) pyproject.toml uv.lock $(wildcard config.toml)

.PHONY: setup data prep train eval figures results final snapshot diff sync workbook check submit \
        lint fmt test clean clean-all

setup: data prep

data:
	$(RUN) python scripts/get_data

# ── pipeline ────────────────────────────────────────────────────────────────
# Stamp files make each stage rerun only when its inputs change, so `make workbook`
# doesn't retrain from scratch every time. Force a rerun with: make -B train
prep:    $(STAMP)/prep
train:   $(STAMP)/train
eval:    $(STAMP)/eval
figures: $(STAMP)/figures
results: figures

$(STAMP)/prep: $(SRC) $(wildcard $(DATA)/*/*.gz) | data
	$(CLI) prep
	@mkdir -p $(@D) && touch $@

$(STAMP)/train: $(STAMP)/prep
	$(CLI) train --task play
	$(CLI) train --task engagement
	@touch $@

$(STAMP)/eval: $(STAMP)/train
	@$(MAKE) --no-print-directory snapshot
	$(CLI) eval
	@touch $@

$(STAMP)/figures: $(STAMP)/eval
	$(CLI) figures
	@touch $@

# Test metrics are deliberately not part of `results`, so nobody tunes against them.
final: results
	$(CLI) eval --split test
	$(CLI) figures

# Copy current metrics to results_history/<timestamp>/ before eval overwrites them.
snapshot:
	@if [ -f $(RES)/metrics.json ]; then \
		d=$(HIST)/$$(date +%Y%m%d-%H%M%S); mkdir -p $$d; \
		cp $(RES)/metrics.json $$d/; echo "snapshot -> $$d"; \
	fi

# Compare metrics against the most recent snapshot.
diff:
	@last=$$(ls -1d $(HIST)/*/ 2>/dev/null | sort | tail -1); \
	if [ -z "$$last" ]; then echo "no snapshot yet"; exit 0; fi; \
	echo "vs $$last"; \
	git --no-pager diff --no-index --word-diff=color $${last}metrics.json $(RES)/metrics.json || true

# ── notebook ────────────────────────────────────────────────────────────────
sync:
	$(RUN) jupytext --sync notebooks/*.ipynb

# Run with the project's env so imports from src/ resolve; outputs are saved
# into the html so graders don't need to execute anything.
workbook: results
	@mkdir -p $(SUB)
	$(RUN) jupyter nbconvert --to html --execute $(NB) \
		--ExecutePreprocessor.kernel_name=python3 \
		--ExecutePreprocessor.timeout=1800 \
		--output-dir $(SUB) --output workbook

# ── submission ──────────────────────────────────────────────────────────────
# Runs the spec's autograder checks, including downloading the start of the video.
check:
	$(RUN) python scripts/check_submission $(SUB)

submit: final workbook check
	@echo "Upload from $(SUB)/:" && ls -1 $(SUB)/workbook.html $(SUB)/video_url.txt

# ── quality ─────────────────────────────────────────────────────────────────
lint:
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

fmt:
	$(RUN) ruff check --fix src tests scripts
	$(RUN) ruff format src tests scripts

test:
	$(RUN) pytest -q

# ── cleanup ─────────────────────────────────────────────────────────────────
# clean keeps results/ and submit/video_url.txt; clean-all wipes generated results too.
clean:
	rm -rf $(ART) $(SUB)/workbook.html

clean-all: clean
	rm -rf $(RES)/metrics.json $(RES)/figures $(HIST)