ifeq ($(OS),Windows_NT)
	VENV_BIN := ../.venv/Scripts
	rmrf = rmdir /s /q
else
	VENV_BIN := ../.venv/bin
	rmrf = rm -rf
endif

VENV_PYTHON ?= $(VENV_BIN)/python
SPHINXOPTS  ?=
SPHINXBUILD ?= $(VENV_BIN)/sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help clean serve
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
ifeq ($(OS),Windows_NT)
	@if exist "$(BUILDDIR)" $(rmrf) "$(BUILDDIR)"
	@if exist "api/_autosummary" $(rmrf) "api/_autosummary"
else
	@$(rmrf) "$(BUILDDIR)"
	@$(rmrf) "api/_autosummary"
endif

serve:
	@$(VENV_PYTHON) serve_docs.py

%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
