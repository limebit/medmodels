import pandas as pd
import numpy as np


def df_to_nodes(
    input_data: pd.DataFrame,
    index_col: str,
    attributes: list = [],
    drop_duplicates: bool = True,
) -> np.array:
    """Transform a given dataframe into the dataclass format for nodes:
    np.array((identifier, {attr: value, ..}))

    :param input_data: Dataset
    :type input_data: pd.DataFrame
    :param index_col: identifier column
    :type index_col: str
    :param attributes: columns to keep, defaults to []
    :type attributes: list, optional
    :param drop_duplicates: drop non-unique entries, keep first, defaults to True
    :type drop_duplicates: bool, optional
    :return: Data Class Format for nodes
    :rtype: _type_: np.array
    """
    assert (
        drop_duplicates or input_data[index_col].is_unique
    ), "Found duplicate values in index. Either solve or set drop_duplicates=True"
    assert set(attributes).issubset(
        input_data.columns
    ), "Found attributes that are not in dataset"

    if drop_duplicates:
        input_data = input_data.drop_duplicates(subset=[index_col])

    # drop nan values from identifier
    data = input_data.copy()
    data = data.dropna(subset=[index_col])
    data = data.set_index(index_col)
    data.index = data.index.astype("str")
    data = data[attributes].to_dict(orient="index")
    data = np.array([(key, value) for key, value in data.items()])
    return data


def df_to_edges(
    data: pd.DataFrame,
    identifier1: str,
    identifier2: str,
    attributes: list = [],
) -> np.array:
    """Transform a given dataframe into the dataclass format for edges:
    np.array((identifier1, identifier2, {attr: value, ..}))

    :param data: Dataset
    :type data: pd.DataFrame
    :param identifier1: identifier column if node 1
    :type identifier1: str
    :param identifier2: identifier column if node 2
    :type identifier2: str
    :param attributes: columns to keep, defaults to []
    :type attributes: list, optional
    :return: Data Class format for edges
    :rtype: np.array
    """

    data = data[[identifier1] + [identifier2] + attributes]
    data = data.astype({identifier1: "str", identifier2: "str"})
    data = data.to_dict(orient="index")
    data = np.array(
        [
            (
                value[identifier1],
                value[identifier2],
                {a: value[a] for a in attributes},
            )
            for key, value in data.items()
        ]
    )
    return data


def align_types(func):
    """Wrapper for comparison functions to cast some x value
    into the same type as the y value

    :param func: Function to be wrapped
    :type func: _type_
    """

    def wrapper(x, y):
        types = [str, float, int]
        for t in types:
            if type(x) is t:
                y = t(y)
        return func(x, y)

    return wrapper


@align_types
def _larger(x, y):
    return True if x > y else False


@align_types
def _larger_equals(x, y):
    return True if x >= y else False


@align_types
def _smaller(x, y):
    return True if x < y else False


@align_types
def _smaller_equals(x, y):
    return True if x <= y else False


@align_types
def _not_equals(x, y):
    return True if x != y else False


@align_types
def _equals(x, y):
    return True if x == y else False


@align_types
def _anyof(x, y):
    y = y[1:-1].split(",")
    return True if x in y else False


@align_types
def _noneof(x, y):
    y = y[1:-1].split(",")
    return False if x in y else True


@align_types
def _startwith(x, y):
    return True if x[: len(y)] == y else False


@align_types
def _startwithany(x, y):
    y = y[1:-1].split(",")
    for start in y:
        if _startwith(x, start):
            return True
    return False


@align_types
def _not_startwith(x, y):
    return False if x[: len(y)] == y else True


def parse_criteria(queries):
    f = {
        ">": _larger,
        ">=": _larger_equals,
        "<": _smaller,
        "<=": _smaller_equals,
        "==": _equals,
        "!=": _not_equals,
        "anyof": _anyof,
        "noneof": _noneof,
        "startwith": _startwith,
        "not_startwith": _not_startwith,
        "startwithany": _startwithany,
    }

    return [
        [dim, attr, f[func], param]
        for dim, attr, func, param in [query.split(" ") for query in queries]
    ]
