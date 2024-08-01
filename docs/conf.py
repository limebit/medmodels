"""Sphinx configuration file."""

import sys
from datetime import date
from pathlib import Path

from myst_parser import __version__
from sphinx.application import Sphinx


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "medmodels"
author = "Limebit GmbH"
copyright = f"{date.today().year}, {author}"
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_pyscript",
    "sphinx_tippy",
    "sphinx_togglebutton",
    "sphinx_multiversion",
    "sphinx.ext.extlinks",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extlinks = {
    "doi": ("https://doi.org/%s", "DOI: %s"),
    "gh-issue": (
        "https://github.com/limebit/medmodels/issues/%s",
        "issue #%s",
    ),
    "gh-user": ("https://github.com/%s", "@%s"),
}

suppress_warnings = ["myst.strikethrough"]

# Hide overload type signatures
# sphinx_toolbox - Box of handy tools for Sphinx
# https://sphinx-toolbox.readthedocs.io/en/latest/
overloads_location = ["bottom"]

# -- Extension settings  -----------------------------------------------------

# # Sphinx-copybutton - add copy button to code blocks
# # https://sphinx-copybutton.readthedocs.io/en/latest/index.html
# # strip the '>>>' and '...' prompt/continuation prefixes.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Support for markdown
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Napoleon settings (Support for Google Style DocStrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_attr_annotations = True

# Auto Doc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}


autosummary_generate = True
autosummary_imported_members = False
add_module_names = False
autodoc_typehints = "signature"
autodoc_typehints_format = "short"

# -- MyST settings ---------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]
myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "ftp": None,
    "wiki": "https://en.wikipedia.org/wiki/{{path}}#{{fragment}}",
    "doi": "https://doi.org/{{path}}",
    "gh-pr": {
        "url": "https://github.com/limebit/medmodels/pull/{{path}}#{{fragment}}",
        "title": "PR #{{path}}",
        "classes": ["github"],
    },
    "gh-issue": {
        "url": "https://github.com/limebit/medmodels/issues/{{path}}#{{fragment}}",
        "title": "Issue #{{path}}",
        "classes": ["github"],
    },
    "gh-user": {
        "url": "https://github.com/{{path}}",
        "title": "@{{path}}",
        "classes": ["github"],
    },
}
myst_number_code_blocks = ["typescript"]
myst_heading_anchors = 3
myst_footnote_transition = True
myst_dmath_double_inline = True
myst_enable_checkboxes = True
myst_substitutions = {
    "role": "[role](#syntax/roles)",
    "directive": "[directive](#syntax/directives)",
}


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "https://raw.githubusercontent.com/limebit/medmodels-static/main/logos/logo_color.svg"
html_favicon = "https://raw.githubusercontent.com/limebit/medmodels-static/main/icons/favicon-32x32.png"
html_title = "MedModels Documentation"

html_theme_options = {
    "show_toc_level": 2,
    "github_url": "https://github.com/limebit/medmodels",
    "use_edit_page_button": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "show_nav_level": 2,
    "collapse_navigation": True,
    "external_links": [
        {"name": "MedModels Home", "url": "https://medmodels.de"},
    ],
    "primary_sidebar_end": ["indices.html"],
    "secondary_sidebar_items": ["page-toc"],
}

html_context = {
    "github_user": "limebit",
    "github_repo": "medmodels",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "dark",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["local.css"]

tippy_skip_anchor_classes = ("headerlink", "sd-stretched-link", "sd-rounded-pill")
tippy_anchor_parent_selector = "article.bd-article"

# -- LaTeX output -------------------------------------------------

latex_engine = "xelatex"


# -- Local Sphinx extensions -------------------------------------------------


def setup(app: Sphinx):
    """Add functions to the Sphinx setup."""
    from myst_parser._docs import (
        DirectiveDoc,
        DocutilsCliHelpDirective,
        MystAdmonitionDirective,
        MystConfigDirective,
        MystExampleDirective,
        MystLexer,
        MystToHTMLDirective,
        MystWarningsDirective,
        NumberSections,
        StripUnsupportedLatex,
    )

    app.add_directive("myst-config", MystConfigDirective)
    app.add_directive("docutils-cli-help", DocutilsCliHelpDirective)
    app.add_directive("doc-directive", DirectiveDoc)
    app.add_directive("myst-warnings", MystWarningsDirective)
    app.add_directive("myst-example", MystExampleDirective)
    app.add_directive("myst-admonitions", MystAdmonitionDirective)
    app.add_directive("myst-to-html", MystToHTMLDirective)
    app.add_post_transform(StripUnsupportedLatex)
    app.add_post_transform(NumberSections)
    app.add_lexer("myst", MystLexer)
