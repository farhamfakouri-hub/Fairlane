ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
BOOTSTRAP_PYTHON ?= python
else
VENV_PYTHON := .venv/bin/python
BOOTSTRAP_PYTHON ?= python3
endif

PYTHON ?= $(VENV_PYTHON)

.PHONY: setup test lint format smoke simulate score experiments ui api openapi

setup:
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

smoke:
	$(PYTHON) -m ree.cli smoke

simulate:
	$(PYTHON) -m ree.cli simulate

score:
	$(PYTHON) -m ree.cli score

experiments:
	$(PYTHON) -m ree.cli experiments

ui:
	$(PYTHON) -m ree.cli ui

api:
	$(PYTHON) -m ree.cli api

openapi:
	$(PYTHON) -m ree.cli export-openapi
