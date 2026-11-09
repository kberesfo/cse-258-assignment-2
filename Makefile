# CSE 258 Assignment 2
#   make prep       load + clean + split data          -> artifacts/
#   make train      fit baselines and models           -> artifacts/
#   make eval       score everything on the fixed split -> results/metrics.json
#   make figures    plots for the notebook and slides  -> results/figures/
#   make results    prep + train + eval + figures (each step reruns only if src/ changed)
#   make diff       metrics now vs the latest snapshot in results_history/
#   make sync       pair/sync notebooks with jupytext
#   make workbook   execute notebooks/workbook.ipynb   -> submit/workbook.html
#   make check      verify the submission files the way the autograder does
#   make submit     results + workbook + check
#   make lint / fmt / test

RUN      ?= uv run
# The CLI isn't settled yet, so everything goes through this one variable.
# Swap it for a console script later, e.g.  CLI ?= $(RUN) a2
CLI      ?= $(RUN) python -m a2
DATA_DIR ?= data   # set DATA_DIR in devcontainer.json to point at the mounted dataset

NB       := notebooks/workbook.ipynb
ART      := artifacts
RES      := results
HIST     := $(RES)_history
SUB      := submit
STAMP    := $(ART)/.stamps
SRC      := $(shell find src -name '*.py' 2>/dev/null) pyproject.toml uv.lock

.PHONY: prep train eval figures results snapshot diff sync workbook check submit \
        lint fmt test clean clean-all

# ── pipeline ────────────────────────────────────────────────────────────────
# Stamp files make each stage rerun only when code changes, so `make workbook`
# doesn't retrain from scratch every time. Force a stage with: make -B train
prep:    $(STAMP)/prep
train:   $(STAMP)/train
eval:    $(STAMP)/eval
figures: $(STAMP)/figures
results: figures

$(STAMP)/prep: $(SRC)
	$(CLI) prep --data-dir $(DATA_DIR) --out-dir $(ART)
	@mkdir -p $(@D) && touch $@

$(STAMP)/train: $(STAMP)/prep
	$(CLI) train --art-dir $(ART)
	@touch $@

$(STAMP)/eval: $(STAMP)/train
	@$(MAKE) --no-print-directory snapshot
	$(CLI) eval --art-dir $(ART) --out $(RES)/metrics.json
	@touch $@

$(STAMP)/figures: $(STAMP)/eval
	$(CLI) figures --art-dir $(ART) --out-dir $(RES)/figures
	@touch $@

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
# Mirrors the autograder's checks from the spec (html header, video link format).
check:
	@test -f $(SUB)/workbook.html || { echo "check: missing $(SUB)/workbook.html"; exit 1; }
	@head -c 15 $(SUB)/workbook.html | grep -q '^<!DOCTYPE html>' \
		&& echo "check: workbook.html is html" \
		|| { echo "check: workbook.html does not start with <!DOCTYPE html>"; exit 1; }
	@test -f $(SUB)/video_url.txt || { echo "check: missing $(SUB)/video_url.txt"; exit 1; }
	@[ $$(grep -c . $(SUB)/video_url.txt) -eq 1 ] || { echo "check: video_url.txt must be one line"; exit 1; }
	@grep -Eq '^https://(drive\.google\.com/(file/d/|uc\?)|(www\.)?youtube\.com/watch|youtu\.be/)' $(SUB)/video_url.txt \
		&& echo "check: video_url.txt looks like a drive/youtube link" \
		|| { echo "check: video_url.txt is not a drive/youtube link"; exit 1; }
	@if [ -f $(SUB)/check_submission.py ]; then $(RUN) python $(SUB)/check_submission.py; fi

submit: workbook check
	@echo "Upload from $(SUB)/:" && ls -1 $(SUB)/workbook.html $(SUB)/video_url.txt

# ── quality ─────────────────────────────────────────────────────────────────
lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

fmt:
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

test:
	$(RUN) pytest -q

# ── cleanup ─────────────────────────────────────────────────────────────────
# clean keeps results/ and submit/video_url.txt; clean-all wipes generated results too.
clean:
	rm -rf $(ART) $(SUB)/workbook.html

clean-all: clean
	rm -rf $(RES)/metrics.json $(RES)/figures