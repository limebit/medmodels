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
# ruff: noqa: D100, D103

import contextlib
import io
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import parselinenos
from sphinx.util.docutils import SphinxDirective


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


def setup(app: Sphinx) -> None:
    """Set up the Sphinx extension.

    Args:
        app (Sphinx): The Sphinx application instance.
    """
    app.add_directive("exec-literalinclude", ExecLiteralInclude)
    app.add_node(
        ErrorMessageNode, html=(visit_error_message_node, depart_error_message_node)
    )
    app.add_css_file("exec_literalinclude.css")
