# Sphinx documentation Makefile

# Command-line overridable variables
VENV_PYTHON 	?= ../.venv/bin/python
SPHINXOPTS    	?=
SPHINXBUILD   	?= sphinx-build
SOURCEDIR    	= .
BUILDDIR      	= _build


# Default target: show help
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean serve

# Clean build and autosummary directories
clean:
	@rm -rf "$(BUILDDIR)" "api/_autosummary"

# Serve docs using live reload server
serve:
	@$(VENV_PYTHON) serve_docs.py

# Catch-all target for Sphinx build
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
