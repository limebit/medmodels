"""Module containing functions for displaying statistics."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Union

from medmodels.medrecord.types import Attributes, AttributeSummary, EdgeIndex, NodeIndex

if TYPE_CHECKING:
    from typing import TypeAlias

    from medmodels.medrecord.types import Group

AttributeDictionary: TypeAlias = Union[
    Dict[EdgeIndex, Attributes], Dict[NodeIndex, Attributes]
]


def prettify_table(
    data: Dict[Group, AttributeSummary], header: List[str], decimal: int
) -> List[str]:
    """Takes a DataFrame and turns it into a list for displaying a pretty table.

    Args:
        data (Dict[Group, AttributeSummary]): Table information
            stored in a dictionary.
        header (List[str]): Header line consisting of column names for the table.
        decimal (int): Decimal point to round the float values to.

    Returns:
        List[str]: List of lines for printing the table.
    """
    lengths = [len(title) for title in header]

    rows = []

    info_order = [
        "type",
        "min",
        "max",
        "mean",
        "median",
        "Q1",
        "Q3",
        "values",
        "count",
        "top",
        "freq",
    ]

    for group in data:
        # determine longest group name and count
        lengths[0] = max(len(str(group)), lengths[0])

        lengths[1] = max(len(str(data[group]["count"])), lengths[1])

        row = [str(group), str(data[group]["count"]), "-", "-", "-"]

        # in case of no attribute info, just keep Group name and count
        if not data[group]["attribute"]:
            rows.append(row)
            continue

        for attribute, info in data[group]["attribute"].items():
            lengths[2] = max(len(str(attribute)), lengths[2])

            # display attribute name only once
            first_line = True

            for key in sorted(info.keys(), key=lambda x: info_order.index(x)):
                if key == "type":
                    continue

                if not first_line:
                    row[0], row[1] = "", ""

                row[2] = str(attribute) if first_line else ""

                row[3] = info["type"] if first_line else ""

                # displaying information based on the type
                if "values" in info:
                    row[4] = info[key]
                else:
                    if isinstance(info[key], float):
                        row[4] = f"{key}: {info[key]:.{decimal}f}"
                    elif isinstance(info[key], datetime):
                        row[4] = f"{key}: {info[key].strftime('%Y-%m-%d %H:%M:%S')}"
                    else:
                        row[4] = f"{key}: {info[key]}"

                lengths[3] = max(len(row[3]), lengths[3])
                lengths[4] = max(len(row[4]), lengths[4])

                rows.append(copy.deepcopy(row))

                first_line = False

    table = [
        "-" * (sum(lengths) + len(lengths)),
        "".join([f"{head.title():<{lengths[i]}} " for i, head in enumerate(header)]),
        "-" * (sum(lengths) + len(lengths)),
    ]

    table.extend(
        [
            "".join(f"{row[x]: <{lengths[x]}} " for x in range(len(lengths)))
            for row in rows
        ]
    )

    table.append("-" * (sum(lengths) + len(lengths)))

    return table
