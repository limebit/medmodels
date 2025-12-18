# pyright: reportPrivateImportUsage=false
"""Sphinx configuration for MedModels documentation."""

import importlib.metadata
import sys
from datetime import date
from pathlib import Path

from sphinx.application import Sphinx

# Add current and parent directory to sys.path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


# -- Project information -----------------------------------------------------
project = "MedModels"
author = "Limebit GmbH"
copyright = f"{date.today().year}, {author}"
version = f"v{importlib.metadata.version(project)}"

# -- General configuration ---------------------------------------------------
language = "en"
root_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
templates_path = ["_templates"]

extensions = [
    # Sphinx built-in extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    # Third-party extensions
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_pyscript",
    "sphinx_tippy",
    "sphinx_togglebutton",
    # Local extensions
    "sphinx_extensions",
]

# -- Extension configuration -------------------------------------------------

# sphinx.ext.extlinks
extlinks = {
    "doi": ("https://doi.org/%s", "DOI: %s"),
    "gh-issue": ("https://github.com/limebit/medmodels/issues/%s", "issue #%s"),
    "gh-user": ("https://github.com/%s", "@%s"),
}

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

suppress_warnings = ["myst.strikethrough"]
overloads_location = ["bottom"]  # Hide overload type signatures

# sphinx_copybutton
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# sphinx.ext.napoleon
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

# sphinx.ext.autodoc and sphinx.ext.autosummary
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "inherited-members": True,
    "show-inheritance": True,
    "ignore-module-all": False,
}

autosummary_generate = True
autosummary_imported_members = False
add_module_names = False
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
maximum_signature_line_length = 88

# Auto-generate Type Aliases from medmodels modules (see sphinx_extensions.py)
autodoc_type_aliases = importlib.import_module(
    "sphinx_extensions"
).get_autodoc_type_aliases()

# sphinx_autodoc_typehints settings for cross-reference linking
typehints_fully_qualified = False
always_use_bars_union = False

# myst_parser
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

# -- HTML output -------------------------------------------------------------
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
    "switcher": {
        "json_url": "https://www.medmodels.de/docs/switcher.json",
        "version_match": version,
    },
    "show_version_warning_banner": False,
}

html_context = {
    "github_user": "limebit",
    "github_repo": "medmodels",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "dark",
}

html_static_path = ["_static"]
html_css_files = ["local.css"]

tippy_skip_anchor_classes = ("headerlink", "sd-stretched-link", "sd-rounded-pill")
tippy_anchor_parent_selector = "article.bd-article"

# -- Other output formats ----------------------------------------------------
latex_engine = "xelatex"


# -- Custom setup ------------------------------------------------------------
def setup(app: Sphinx) -> None:
    """Add custom directives and transformations to Sphinx."""
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
