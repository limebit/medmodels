VENV_NAME?=.venv

USER_PYTHON ?= python3
VENV_PYTHON=${VENV_NAME}/bin/python

.PHONY = prepare-venv install install-dev install-tests test lint format clean

.DEFAULT_GOAL = install-dev

prepare-venv: $(VENV_NAME)/bin/python

$(VENV_NAME)/bin/python:
	make clean && ${USER_PYTHON} -m venv $(VENV_NAME)

install: prepare-venv
	${VENV_PYTHON} -m pip install -U pip
	${VENV_PYTHON} -m pip install -e .

install-dev: prepare-venv
	${VENV_PYTHON} -m pip install -U pip
	${VENV_PYTHON} -m pip install -e .\[dev\]

install-tests: prepare-venv
	${VENV_PYTHON} -m pip install -U pip
	${VENV_PYTHON} -m pip install -e .\[tests\]

install-docs: prepare-venv
	${VENV_PYTHON} -m pip install -U pip
	${VENV_PYTHON} -m pip install -e .\[docs\]

build-dev: install-dev
	${VENV_PYTHON} -m maturin develop

test: install-tests
	${VENV_PYTHON} -m pytest -W error
	cargo test

docs: install-docs
	$(MAKE) -C docs html

docs-serve: install-docs
	$(MAKE) -C docs serve VENV_PYTHON=$(CURDIR)/$(VENV_PYTHON)

docs-clean:
	$(MAKE) -C docs clean

lint: install-dev
	${VENV_PYTHON} -m ruff check
	${VENV_PYTHON} -m ruff check --select I
	${VENV_PYTHON} -m pyright
	cargo clippy --all-targets --all-features

format: install-dev
	${VENV_PYTHON} -m ruff check --select I --fix
	${VENV_PYTHON} -m ruff format
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
