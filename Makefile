VENV_NAME ?= .venv

ifeq ($(OS),Windows_NT)
	USER_PYTHON ?= python
	VENV_BIN := $(VENV_NAME)/Scripts
	rmrf = rmdir /s /q
	rmf = del /q
else
	USER_PYTHON ?= python3
	VENV_BIN := $(VENV_NAME)/bin
	rmrf = rm -rf
	rmf = rm -f
endif

VENV_PYTHON := $(VENV_BIN)/python
VENV_UV := $(VENV_BIN)/uv
UV_LOC := $(shell $(USER_PYTHON) -c "import shutil; print(shutil.which('uv') if shutil.which('uv') else '$(VENV_UV)')")

.PHONY: prepare-venv install install-dev install-tests test lint format clean docs docs-serve docs-clean

.DEFAULT_GOAL := install-dev

prepare-venv: $(VENV_NAME)

$(VENV_NAME):
ifeq ($(UV_LOC),$(VENV_UV))
	@echo "Using .venv installed uv: $(UV_LOC)"
	$(MAKE) clean && $(USER_PYTHON) -m venv $(VENV_NAME)
	$(VENV_PYTHON) -m pip install -U pip
	$(VENV_PYTHON) -m pip install uv
else
	@echo "Using global uv: $(UV_LOC)"
	$(MAKE) clean && $(UV_LOC) venv $(VENV_NAME)
endif

install: prepare-venv
	$(UV_LOC) sync

install-dev: prepare-venv
	$(UV_LOC) sync --group dev

install-tests: prepare-venv
	$(UV_LOC) sync --group tests

install-docs: prepare-venv
	$(UV_LOC) sync --group docs

build-dev: install-dev
	$(UV_LOC) run maturin develop

test: install-tests
	$(UV_LOC) run pytest -vv -W error
	cargo test

docs: install-docs
	$(MAKE) -C docs html

docs-serve: install-docs
	$(MAKE) -C docs serve VENV_PYTHON=$(CURDIR)/$(VENV_PYTHON)

docs-clean:
	$(MAKE) -C docs clean

lint: install-dev
	$(UV_LOC) run ruff check
	$(UV_LOC) run ruff check --select I
	$(UV_LOC) run python -m pyright
	cargo clippy --all-targets --all-features

format: install-dev
	$(UV_LOC) run ruff check --select I --fix
	$(UV_LOC) run ruff format
	cargo fmt
	cargo clippy --all-features --fix --allow-staged

clean: docs-clean
ifeq ($(OS),Windows_NT)
	@if exist $(VENV_NAME) $(rmrf) $(VENV_NAME)
	@for %%d in (target dist medmodels.egg-info .ruff_cache .pytest_cache) do @if exist %%d $(rmrf) %%d
	@if exist medmodels\*.pyd $(rmf) medmodels\*.pyd
	@if exist .vscode\*.log $(rmf) .vscode\*.log
	@if exist .coverage $(rmf) .coverage
	@for /d /r %%i in (__pycache__) do @if exist "%%i" $(rmrf) "%%i"
else
	$(rmrf) $(VENV_NAME) target dist medmodels.egg-info .ruff_cache .pytest_cache
	$(rmf) medmodels/*.so
	$(rmf) .vscode/*.log
	$(rmf) .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
endif
