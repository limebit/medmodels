# # Configuration file for the Sphinx documentation builder.
# #
# # For the full list of built-in configuration values, see the documentation:
# # https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any
import warnings


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, str(Path("../../medmodels").resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "medmodels"
author = "Limebit GmbH"
copyright = f"2024, {author}"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    # Third-party extensions
    "autodocsumm",
    "sphinx_autosummary_accessors",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_favicon",
    "sphinx_reredirects",
    "sphinx_toolbox.more_autodoc.overloads",
]

# Render docstring text in `single backticks` as code.
default_role = "code"

maximum_signature_line_length = 88

templates_path = ["_templates"]
exclude_patterns = ["Thumbs.db", ".DS_Store"]

# Hide overload type signatures
# sphinx_toolbox - Box of handy tools for Sphinx
# https://sphinx-toolbox.readthedocs.io/en/latest/
overloads_location = ["bottom"]

# -- Extension settings  -----------------------------------------------------

# Sphinx-copybutton - add copy button to code blocks
# https://sphinx-copybutton.readthedocs.io/en/latest/index.html
# strip the '>>>' and '...' prompt/continuation prefixes.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# redirect empty root to the actual landing page
redirects = {"index": "reference/index.html"}

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
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Auto Doc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_inherit_docstrings = False

# Autosummary Options
autosummary_generate = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../../images"]
html_css_files = ["css/custom.css"]  # relative to html_static_path
html_show_sourcelink = True

# key site root paths
static_assets_root = "https://raw.githubusercontent.com/pola-rs/polars-static/master"
github_root = "https://github.com/medmodels/medmodels"
web_root = "https://medmodels.de"

# # Specify version for version switcher dropdown menu
git_ref = os.environ.get("POLARS_VERSION", "main")
version_match = re.fullmatch(r"py-(\d+)\.\d+\.\d+.*", git_ref)
switcher_version = version_match.group(1) if version_match is not None else "dev"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": github_root,
            "icon": "fa-brands fa-github",
        },
        {
            "name": "LinkedIn",
            "url": "https://linkedin.com/company/medmodels",
            "icon": "fa-brands fa-linkedin",
        },
    ],
    "logo": {
        "image_light": f"{html_static_path}/medmodels_logo.svg",
        "image_dark": f"{html_static_path}/medmodels_logo.svg",
    },
    # "switcher": {
    #     "json_url": f"{web_root}/api/python/dev/_static/version_switcher.json",
    #     "version_match": switcher_version,
    # },
    "show_version_warning_banner": True,
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "check_switcher": True,
}

# sphinx-favicon - Add support for custom favicons
# https://github.com/tcmetzger/sphinx-favicon
favicons = [
    {
        "rel": "icon",
        "sizes": "32x32",
        "href": f"{static_assets_root}/icons/favicon-32x32.png",
    },
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "href": f"{static_assets_root}/icons/touchicon-180x180.png",
    },
]


# sphinx-ext-linkcode - Add external links to source code
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
def linkcode_resolve(domain: str, info: dict[str, Any]):
    """
    Determine the URL corresponding to Python object.

    Based on pandas equivalent:
    https://github.com/pandas-dev/pandas/blob/main/doc/source/conf.py#L629-L686
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            with warnings.catch_warnings():
                # Accessing deprecated objects will generate noisy warnings
                warnings.simplefilter("ignore", FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    conf_dir_path = Path(__file__).absolute().parent
    polars_root = (conf_dir_path.parent.parent / "polars").absolute()

    fn = os.path.relpath(fn, start=polars_root)
    return f"{github_root}/blob/{git_ref}/py-polars/polars/{fn}{linespec}"


def _minify_classpaths(s: str) -> str:
    # strip private polars classpaths, leaving the classname:
    # * "pl.Expr" -> "Expr"
    # * "polars.expr.expr.Expr" -> "Expr"
    # * "polars.lazyframe.frame.LazyFrame" -> "LazyFrame"
    # also:
    # * "datetime.date" => "date"
    s = s.replace("datetime.", "")
    return re.sub(
        pattern=r"""
        ~?
        (
          (?:pl|
            (?:polars\.
              (?:_reexport|datatypes)
            )
          )
          (?:\.[a-z.]+)?\.
          ([A-Z][\w.]+)
        )
        """,
        repl=r"\2",
        string=s,
        flags=re.VERBOSE,
    )


def process_signature(  # noqa: D103
    app: object,
    what: object,
    name: object,
    obj: object,
    opts: object,
    sig: str,
    ret: str,
) -> tuple[str, str]:
    return (
        _minify_classpaths(sig) if sig else sig,
        _minify_classpaths(ret) if ret else ret,
    )


def setup(app: Any) -> None:  # noqa: D103
    # TODO: a handful of methods do not seem to trigger the event for
    #  some reason (possibly @overloads?) - investigate further...
    app.connect("autodoc-process-signature", process_signature)
