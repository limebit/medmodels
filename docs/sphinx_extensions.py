"""Sphinx extension to execute included code snippets and display their output.

This module defines a custom Sphinx directive `ExecLiteralInclude` that allows
including code from external files, executing it, and displaying both the code
and its output in the documentation. In case the user wants to show an expected error
message, they can specify the error message using the `expect-error` option.

Example:
    ```{exec-literalinclude} path/to/your_script.py
    ---
    language: python
    setup-lines: 1-4
    lines: 6-10
    expect-error: ValueError
    ---
    ```
"""

import contextlib
import io
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import parselinenos
from sphinx.util.docutils import SphinxDirective

# Modules to scan for type alias discovery
MODULES_TO_SCAN = [
    "medmodels.medrecord.types",
    "medmodels.medrecord.querying",
    "medmodels.medrecord.schema",
    "medmodels.medrecord.datatype",
]

# Mutable cache for discovered type names (populated on first use)
_type_names: Dict[str, str] = {}
_type_names_initialized = False


def _get_typing_links() -> Dict[str, str]:
    """Generate links to Python docs for typing module types.

    Returns:
        Dict[str, str]: Mapping of type name to Python docs URL.
    """
    import typing

    base_url = "https://docs.python.org/3/library/typing.html#typing."
    links: Dict[str, str] = {}

    # Skip internal type variables and constants
    skip_names = {
        # Type variables used internally
        "T", "T_co", "T_contra", "KT", "VT", "VT_co", "V_co", "CT_co", "AnyStr",
        # Constants
        "TYPE_CHECKING", "EXCLUDED_ATTRIBUTES",
        # Internal/meta classes
        "ABCMeta", "GenericAlias", "NamedTupleMeta", "ForwardRef",
        # Param spec internals
        "ParamSpecArgs", "ParamSpecKwargs",
    }

    for name in dir(typing):
        # Skip private names
        if name.startswith("_"):
            continue

        # Skip known internal names
        if name in skip_names:
            continue

        # Only include PascalCase names (skip ALL_CAPS constants)
        if not name[0].isupper() or name.isupper():
            continue

        obj = getattr(typing, name, None)
        if obj is None:
            continue

        # Skip basic types that are just re-exports (like int, str)
        if hasattr(obj, "__module__") and obj.__module__ == "builtins":
            continue

        links[name] = f"{base_url}{name}"

    return links


TYPING_LINKS = _get_typing_links()


class ErrorMessageNode(nodes.General, nodes.Element):
    """A custom node to represent a formatted error message."""

    def __init__(self, message: str) -> None:
        """Initialize the ErrorMessageNode.

        Args:
            message (str): The error message to display.
        """
        super().__init__()
        self.message = message


def visit_error_message_node(
    visitor: nodes.NodeVisitor, node: ErrorMessageNode
) -> None:
    """Visitor function to generate HTML for ErrorMessageNode.

    Args:
        visitor (nodes.NodeVisitor): The visitor instance.
        node (ErrorMessageNode): The ErrorMessageNode instance.
    """
    visitor.body.append(  # pyright: ignore[reportAttributeAccessIssue]
        f'<div class="admonition error"><p class="admonition-title error-title">Error:</p><p>{node.message}</p></div>'
    )


def depart_error_message_node(
    visitor: nodes.NodeVisitor, node: ErrorMessageNode
) -> None:
    """Departure function for ErrorMessageNode. Placeholder function.

    Args:
        visitor (nodes.NodeVisitor): The visitor instance.
        node (ErrorMessageNode): The ErrorMessageNode instance.
    """
    pass


class ExecLiteralInclude(SphinxDirective):
    """Directive to include, execute, and display code from external files."""

    required_arguments = 1  # The file path is the only required argument
    optional_arguments = 0
    option_spec: ClassVar[Dict[str, Callable[[str], Any]]] = {  # pyright: ignore[reportIncompatibleVariableOverride]
        "lines": lambda x: x,
        "setup-lines": lambda x: x,
        "language": lambda x: x,
        "expect-error": lambda x: x,
    }
    has_content = False

    def run(self) -> List[nodes.Node]:  # noqa: C901
        """Process the directive and return nodes to be inserted into the document.

        Returns:
            List[nodes.Node]: A list of docutils nodes representing the code block
                and its output.

        Raises:
            KeyboardInterrupt: If the code raises a KeyboardInterrupt exception.
            SystemExit: If the code raises a SystemExit exception.
            RuntimeError: If an expected error does not occur.
        """
        environment = self.state.document.settings.env
        _, filename = environment.relfn2path(self.arguments[0])

        try:
            with Path(filename).open("r") as file:
                code_lines = file.readlines()
        except FileNotFoundError:
            error = self.state_machine.reporter.error(
                f"File not found: {filename}", line=self.lineno
            )
            return [error]

        total_lines = len(code_lines)
        expected_error = self.options.get("expect-error")

        # Extract setup code
        setup_code = ""
        if "setup-lines" in self.options:
            setup_line_numbers = parselinenos(self.options["setup-lines"], total_lines)
            setup_code = "".join([code_lines[i] for i in setup_line_numbers])

        # Extract main code
        if "lines" in self.options:
            main_line_numbers = parselinenos(self.options["lines"], total_lines)
            main_code = "".join([code_lines[i] for i in main_line_numbers])
        else:
            main_code = "".join(code_lines)

        # Create a literal block node for the main code
        code_node = nodes.literal_block(main_code, main_code)
        code_node["language"] = self.options.get("language", "python")

        # Prepare code for execution
        main_code_lines = main_code.rstrip().split("\n")

        # Remove trailing empty lines
        while main_code_lines and not main_code_lines[-1].strip():
            main_code_lines.pop()
        if main_code_lines:
            last_line = main_code_lines.pop()
            code_before_last_line = "\n".join(main_code_lines)
        else:
            last_line = ""
            code_before_last_line = ""

        # Execute code and capture output
        output_io = io.StringIO()
        exec_globals = {}

        try:
            with (
                contextlib.redirect_stdout(output_io),
                contextlib.redirect_stderr(output_io),
            ):
                if setup_code:
                    exec(setup_code, exec_globals)
                if code_before_last_line:
                    exec(code_before_last_line, exec_globals)
                if last_line:
                    if self._is_expression(last_line):
                        result = eval(last_line, exec_globals)
                        if result is not None:
                            print(repr(result))  # noqa: T201
                    else:
                        exec(last_line, exec_globals)

        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            if expected_error and e.__class__.__name__ == expected_error:
                return [code_node, ErrorMessageNode(f"{expected_error}: {e}")]
            raise

        if expected_error:
            msg = f"Expected error '{expected_error}' did not occur."
            raise RuntimeError(msg)

        output_text = output_io.getvalue()

        # Create a literal block for the output
        output_node = nodes.literal_block(output_text, output_text)
        output_node["language"] = "none"

        return [code_node, output_node]

    def _is_expression(self, code_line: str) -> bool:
        """Determine if a line of code is an expression.

        Args:
            code_line (str): The line of code to check.

        Returns:
            bool: True if the line is an expression, False otherwise.
        """
        try:
            compile(code_line, "<string>", "eval")
            return True
        except SyntaxError:
            return False

    def _parse_line_range(self, line_range_str: str, total_lines: int) -> List[int]:
        """Parse a line range string into a list of line indices.

        Args:
            line_range_str (str): The line range string (e.g., "1-3,5").
            total_lines (int): The total number of lines in the source file.

        Returns:
            List[int]: A list of line indices (0-based).
        """
        return parselinenos(line_range_str, total_lines)


def discover_type_names() -> Dict[str, str]:
    """Discover type alias and class names from medmodels modules.

    Only includes types that are unique to medmodels (not in typing module).

    Returns:
        Dict[str, str]: Mapping of type name to module name.
    """
    import importlib
    import inspect
    import typing

    types: Dict[str, str] = {}

    # Get all names from the typing module to avoid conflicts
    typing_names = set(dir(typing))

    for module_name in MODULES_TO_SCAN:
        module = importlib.import_module(module_name)

        for name in dir(module):
            if name.startswith("_"):
                continue

            # Skip names that exist in the typing module to avoid conflicts
            if name in typing_names:
                continue

            obj = getattr(module, name, None)
            if obj is None:
                continue

            obj_module = getattr(obj, "__module__", None)

            # Classes defined in this module
            if inspect.isclass(obj) and obj_module == module_name:
                types[name] = module_name
                continue

            # Type aliases (annotated names)
            annotations = getattr(module, "__annotations__", {})
            if name in annotations:
                types[name] = module_name

    return types


def get_autodoc_type_aliases() -> Dict[str, str]:
    """Get autodoc_type_aliases config from discovered types.

    Returns:
        Dict[str, str]: Mapping of type name to itself for autodoc.
    """
    return {name: name for name in discover_type_names()}


def _fix_typing_references(doctree: nodes.document) -> None:
    """Fix references to typing module types that were incorrectly resolved.

    sphinx_autodoc_typehints sometimes resolves typing.Union, typing.Optional, etc.
    to local classes with the same name. This function fixes those references
    to point to Python docs instead.
    """
    # Mapping of incorrectly resolved refs to correct Python docs URLs
    typing_fixes = {
        "medmodels.medrecord.datatype.Union": "https://docs.python.org/3/library/typing.html#typing.Union",
        "medmodels.medrecord.datatype.Optional": "https://docs.python.org/3/library/typing.html#typing.Optional",
    }

    for ref in list(doctree.findall(nodes.reference)):
        refuri = ref.get("refuri", "")
        reftitle = ref.get("reftitle", "")

        # Check if this reference points to a wrongly resolved typing type
        for wrong_target, correct_url in typing_fixes.items():
            if wrong_target in refuri or wrong_target in reftitle:
                # Convert to external reference pointing to Python docs
                ref["refuri"] = correct_url
                ref["internal"] = False
                ref["classes"] = ["reference", "external"]
                if "reftitle" in ref.attributes:
                    del ref.attributes["reftitle"]
                break


def _get_doc_target(app: Sphinx, type_name: str, module_name: str) -> str:
    """Get the documentation target for a type, checking if the page exists.

    Args:
        app: The Sphinx application instance.
        type_name: The name of the type.
        module_name: The module where the type is defined.

    Returns:
        The documentation path (without .html extension).
    """
    # Check the autosummary RST source files (generated before doctree processing)
    srcdir = Path(app.srcdir) / "api" / "_autosummary"

    # First try the specific type page
    specific_path = f"{module_name}.{type_name}"
    if (srcdir / f"{specific_path}.rst").exists():
        return specific_path

    # Fall back to module page
    if (srcdir / f"{module_name}.rst").exists():
        return module_name

    # Last resort: return the module path (safer than broken specific link)
    return module_name


def link_type_aliases(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Link type alias names to their documentation pages.

    This function processes the doctree and replaces type alias names
    with cross-reference links to their documentation. Also links typing
    module types (List, Optional, etc.) to Python docs.

    Args:
        app: The Sphinx application instance.
        doctree: The document tree.
        docname: The name of the document being processed.
    """
    import re

    global _type_names, _type_names_initialized

    # Initialize type names on first call
    if not _type_names_initialized:
        _type_names.update(discover_type_names())
        _type_names_initialized = True

    # Fix incorrectly resolved typing module references
    # sphinx_autodoc_typehints sometimes resolves typing.Union to local Union classes
    _fix_typing_references(doctree)

    # Combine medmodels types and typing module types for pattern matching
    all_type_names = set(_type_names.keys()) | set(TYPING_LINKS.keys())

    if not all_type_names:
        return

    # Calculate the relative path prefix based on document depth
    depth = docname.count("/")
    prefix = "../" * depth + "api/_autosummary/"

    # Build regex pattern to match type aliases (whole words only)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in all_type_names) + r")\b"
    )

    # Find all Text nodes and check for type aliases
    for node in list(doctree.findall(nodes.Text)):
        parent = node.parent
        if parent is None:
            continue

        # Skip if already inside a reference (check all ancestors)
        ancestor = parent
        skip = False
        while ancestor is not None:
            if isinstance(ancestor, nodes.reference):
                skip = True
                break
            if isinstance(ancestor, (nodes.literal_block, nodes.raw, nodes.comment)):
                skip = True
                break
            ancestor = ancestor.parent
        if skip:
            continue

        text = str(node)
        matches = list(pattern.finditer(text))

        if not matches:
            continue

        # Build new nodes with links
        new_nodes: List[nodes.Node] = []
        last_end = 0

        for match in matches:
            # Add text before match
            if match.start() > last_end:
                new_nodes.append(nodes.Text(text[last_end : match.start()]))

            # Create reference node for the type
            type_name = match.group(1)

            if type_name in TYPING_LINKS:
                # Link to Python docs for typing module types
                ref_node = nodes.reference(
                    "",
                    type_name,
                    internal=False,
                    refuri=TYPING_LINKS[type_name],
                    classes=["reference", "external"],
                )
            else:
                # Link to medmodels docs for custom types
                module_name = _type_names[type_name]
                target = _get_doc_target(app, type_name, module_name)
                ref_uri = f"{prefix}{target}.html"

                ref_node = nodes.reference(
                    "",
                    type_name,
                    internal=True,
                    refuri=ref_uri,
                    classes=["reference", "internal"],
                )
            new_nodes.append(ref_node)
            last_end = match.end()

        # Add remaining text
        if last_end < len(text):
            new_nodes.append(nodes.Text(text[last_end:]))

        # Replace the original text node with new nodes
        if new_nodes:
            parent_index = parent.index(node)
            parent.remove(node)
            for i, new_node in enumerate(new_nodes):
                parent.insert(parent_index + i, new_node)


def setup(app: Sphinx) -> None:
    """Set up the Sphinx extension.

    Args:
        app: The Sphinx application instance.
    """
    app.add_directive("exec-literalinclude", ExecLiteralInclude)
    app.add_node(
        ErrorMessageNode, html=(visit_error_message_node, depart_error_message_node)
    )
    app.add_css_file("exec_literalinclude.css")

    # Register the type alias linker
    app.connect("doctree-resolved", link_type_aliases)
