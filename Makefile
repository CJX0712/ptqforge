# PTQForge — developer Makefile. Author: 晨星.
PY ?= python
VENV ?= .venv

.PHONY: help venv install lint format test ci demo clean

help:
	@echo "Targets: venv install lint format test ci demo clean"

venv:
	$(PY) -m venv $(VENV)

install:
	$(VENV)/Scripts/python -m pip install -U pip
	$(VENV)/Scripts/python -m pip install -r requirements.txt

lint:
	$(VENV)/Scripts/python -m ruff check .
	$(VENV)/Scripts/python -m ruff format --check .

format:
	$(VENV)/Scripts/python -m ruff format .
	$(VENV)/Scripts/python -m ruff check --fix .

test:
	$(VENV)/Scripts/python -m pytest -q -W ignore::UserWarning --cov=ptqforge --cov-report=term-missing

ci: lint test

demo:
	$(VENV)/Scripts/python ptqforge/examples/run_demo.py

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ .coverage benchmark*.json
