VENV_NAME?=.venv

USER_PYTHON ?= python3
VENV_PYTHON=${VENV_NAME}/bin/python
VENV_UV=${VENV_NAME}/bin/uv
GLOBAL_PYTHON := $(shell $(USER_PYTHON) -c "import sys, os; print(os.path.join(sys.base_prefix, 'bin', 'python3') if sys.base_prefix != sys.prefix else sys.executable)")
UV_LOC := $(shell $(GLOBAL_PYTHON) -c "import importlib.util; print('uv' if importlib.util.find_spec('uv') else '.venv/bin/uv')")

.PHONY = prepare-venv install install-dev install-tests test lint format clean

.DEFAULT_GOAL = install-dev

prepare-venv: $(VENV_NAME)

$(VENV_NAME):
ifeq ($(UV_LOC), uv)
	@echo "Using global uv: ${UV_LOC}"
	$(MAKE) clean && ${UV_LOC} venv $(VENV_NAME)
	${UV_LOC} pip install -U pip
else
	@echo "Using .venv installed uv: ${UV_LOC}"
	$(MAKE) clean && ${USER_PYTHON} -m venv $(VENV_NAME)
	${VENV_PYTHON} -m pip install -U pip
	${VENV_PYTHON} -m pip install uv
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
	${UV_LOC} tool run maturin develop

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
	${UV_LOC} tool run ruff check
	${UV_LOC} tool run ruff check --select I
	${UV_LOC} run python -m pyright
	cargo clippy --all-targets --all-features

format: install-dev
	${UV_LOC} tool run ruff check --select I --fix
	${UV_LOC} tool run ruff format
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
