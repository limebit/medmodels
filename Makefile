VENV_NAME?=.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN:=$(VENV_NAME)/Scripts
else
	VENV_BIN:=$(VENV_NAME)/bin
endif

USER_PYTHON?=python3
VENV_PYTHON:=$(VENV_BIN)/python
VENV_UV=${VENV_BIN}/uv
UV_LOC:=$(shell $(USER_PYTHON) -c 'import shutil; print(shutil.which("uv") if shutil.which("uv") else "$(VENV_UV)")')

.PHONY = prepare-venv install install-dev install-tests test lint format clean

.DEFAULT_GOAL = install-dev

prepare-venv: $(VENV_NAME)

$(VENV_NAME):
ifeq ($(UV_LOC), $(VENV_UV))
	@echo "Using .venv installed uv: ${UV_LOC}"
	$(MAKE) clean && ${USER_PYTHON} -m venv $(VENV_NAME)
	${VENV_PYTHON} -m pip install -U pip
	${VENV_PYTHON} -m pip install uv
else
	@echo "Using global uv: ${UV_LOC}"
	$(MAKE) clean && ${UV_LOC} venv $(VENV_NAME)
	${UV_LOC} pip install -U pip
endif

install: prepare-venv
	${UV_LOC} sync
	${UV_LOC} pip install -e .

install-dev: prepare-venv
	${UV_LOC} sync  --extra dev
	${UV_LOC} pip install -e .

install-tests: prepare-venv
	${UV_LOC} sync --extra tests
	${UV_LOC} pip install -e .

install-docs: prepare-venv
	${UV_LOC} sync --extra docs
	${UV_LOC} pip install -e .

build-dev: install-dev
	${UV_LOC} run maturin develop

test: install-tests
	${UV_LOC} run pytest -W error
	cargo test

docs: install-docs
	$(MAKE) -C docs html

docs-serve: install-docs
	$(MAKE) -C docs serve VENV_PYTHON=$(CURDIR)/$(VENV_PYTHON)

docs-clean:
	$(MAKE) -C docs clean

lint: install-dev
	${UV_LOC} run ruff check
	${UV_LOC} run ruff check --select I
	${UV_LOC} run python -m pyright
	cargo clippy --all-targets --all-features

format: install-dev
	${UV_LOC} run ruff check --select I --fix
	${UV_LOC} run ruff format
	cargo fmt
	cargo clippy --all-features --fix --allow-staged

clean: docs-clean
	rm -rf $(VENV_NAME)
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf medmodels.egg-info
	rm -rf target
	rm -rf dist
	rm -f medmodels/*.so
	rm -f .vscode/*.log
	find . -name __pycache__ -type d -exec rm -r {} +
