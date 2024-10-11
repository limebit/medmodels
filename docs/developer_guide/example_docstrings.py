from __future__ import annotations

from typing import Any, Dict, Iterator


def example_function_args(
    param1: int,
    param2: str | int,
    optional_param: list[str] | None = None,
    *args: float | str,
    **kwargs: Dict[str, Any],
) -> tuple[bool, list[str]]:
    """Example function with PEP 484 type annotations and PEP 563 future annotations.

    This function shows how to define and document typing for different kinds of
    arguments, including positional, optional, variable-length args, and keyword args.

    Args:
        param1 (int): A required integer parameter.
        param2 (str | int): A parameter that can be either a string or an integer.
        optional_param (list[str] | None, optional): An optional parameter that accepts
            a list of strings. Defaults to None if not provided.
        *args (float | str): Variable length argument list that accepts floats or
            strings.
        **kwargs (Dict[str, Any]): Arbitrary keyword arguments as a dictionary of string
            keys and values of any type.

    Returns:
        tuple[bool, list[str]]: A tuple containing:
            - bool: Always True for this example.
            - list[str]: A list with a single string describing the received arguments.
    """
    result = (
        f"Received: param1={param1}, param2={param2}, optional_param={optional_param}, "
        f"args={args}, kwargs={kwargs}"
    )

    return True, [result]


def example_generator(n: int) -> Iterator[int]:
    """Generators have a ``Yields`` section instead of a ``Returns`` section.

    Args:
        n (int): The upper limit of the range to generate, from 0 to `n` - 1.

    Yields:
        int: The next number in the range of 0 to `n` - 1.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> print([i for i in example_generator(4)])
        [0, 1, 2, 3]

    """
    yield from range(n)
