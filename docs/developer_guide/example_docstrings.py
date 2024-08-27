"""This module provides functionality for analyzing something awesome.

This module contains functions and classes for performing awesome analysis
of example effects. It includes tools for finding nodes
within specific time windows, calculating time-dependent metrics, and
visualizing patterns in outcomes.

Attributes:
    DEFAULT_TIME_FORMAT (str): The default time format string used across the module.
        It follows the format 'YYYY-MM-DD HH:MM:SS'.
    MAX_TIME_WINDOW (int): The maximum allowed time window in days for analysis.
    EFFECT_THRESHOLD (float): The threshold value for considering an effect significant.

Example:
    Basic usage of the module:

    .. code-block:: python
        :linenos:
        :caption: example.py

        from medmodels.treatment_effect import temporal_analysis

        # Find nodes within a specific time window
        nodes = temporal_analysis.find_node_in_time_window(
            graph, start_time="2023-01-01", end_time="2023-12-31"
        )

        # Calculate time-dependent treatment effect
        effect = temporal_analysis.calculate_time_dependent_effect(
            treatment_data, outcome_data, time_points
        )

        # Check if effect is significant
        is_significant = effect > temporal_analysis.EFFECT_THRESHOLD

    .. code-block:: console
        :caption: console output

        [Node1, Node2, Node3]
        0.75
        True

.. note::
    This module assumes that input data is properly formatted and validated.
    Please refer to individual function docstrings for specific input requirements.

See Also:
    - :mod:`medmodels.example_module`: For data preparation utilities
    - :mod:`medmodels.example_module2`: For additional visualization tools
"""

# Module-level variables
DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_TIME_WINDOW = 365
EFFECT_THRESHOLD = 0.5


def function(param1: int, param2: int) -> bool:
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    return param1 == param2


def module_level_function(param1: int, param2: int, *args, **kwargs) -> bool:
    """This is an example of a module level function.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    If ``*args`` or ``**kwargs`` are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name (type): description
            The description may span multiple lines. Following
            lines should be indented. The "(type)" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.

        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {"param1": param1, "param2": param2}

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    if param1 == param2:
        msg = "param1 may not be equal to param2"
        raise ValueError(msg)
    return True


def example_generator(n):
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


class ExampleError(Exception):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        msg (str): Human readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """

    def __init__(self, msg, code) -> None:
        self.msg = msg
        self.code = code


class ExampleClass:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, param1, param2, param3) -> None:
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (list(str)): Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list(str): Doc comment *before* attribute, with type specified
        self.attr4 = ["attr4"]

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

    @property
    def readonly_property(self) -> str:
        """str: Properties should be documented in their getter method."""
        return "readonly_property"

    @property
    def readwrite_property(self):
        """list(str): Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return ["readwrite_property"]

    @readwrite_property.setter
    def readwrite_property(self, value) -> None:
        value

    def example_method(self, param1, param2) -> bool:
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        return True

    def __special__(self):
        """By default special members with docstrings are not included.

        Special members are any methods or attributes that start with and
        end with a double underscore. Any special member with a docstring
        will be included in the output, if
        ``napoleon_include_special_with_doc`` is set to True.

        This behavior can be enabled by changing the following setting in
        Sphinx's conf.py::

            napoleon_include_special_with_doc = True

        """
        pass

    def __special_without_docstring__(self):
        pass

    def _private(self) -> None:
        """By default private members are not included.

        Private members are any methods or attributes that start with an
        underscore and are *not* special. By default they are not included
        in the output.

        This behavior can be changed such that private members *are* included
        by changing the following setting in Sphinx's conf.py::

            napoleon_include_private_with_doc = True

        """
        pass

    def _private_without_docstring(self) -> None:
        pass


class ExamplePEP526Class:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. If ``napoleon_attr_annotations``
    is True, types can be specified in the class body using ``PEP 526``
    annotations.

    Attributes:
        attr1: Description of `attr1`.
        attr2: Description of `attr2`.

    """

    attr1: str
    attr2: int
