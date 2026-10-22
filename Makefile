# Common tasks. Run `make help` for the list.
NOTEBOOK := notebooks/Assignment3_Hyperparameter_Tuning_Anush_Goel.ipynb
PY ?= python
REPO ?=

.DEFAULT_GOAL := help
.PHONY: help install install-dev report app test lint clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the dashboard and report dependencies
	$(PY) -m pip install -r requirements.txt

install-dev: ## Install everything needed for tests and linting
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

report: ## Fill the notebook analysis and build REPORT.md from results/ (optional: REPO=owner/name)
	$(PY) scripts/fill_report.py $(NOTEBOOK)
	$(PY) scripts/build_report.py $(if $(REPO),--repo $(REPO),)

app: ## Launch the Streamlit dashboard
	streamlit run streamlit_app.py

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Lint the Python sources
	ruff check .

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
