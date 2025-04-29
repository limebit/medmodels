"""Module for extracting and displaying an overview of the data in a MedRecord."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

from medmodels.medrecord.schema import AttributesSchema, AttributeType
from medmodels.medrecord.types import Attributes, EdgeIndex, NodeIndex

if TYPE_CHECKING:
    from typing import TypeAlias

    from medmodels.medrecord.types import (
        AttributeInfo,
        Group,
        MedRecordAttribute,
        NumericAttributeInfo,
        StringAttributeInfo,
        TemporalAttributeInfo,
    )

AttributeDictionary: TypeAlias = Union[
    Dict[EdgeIndex, Attributes], Dict[NodeIndex, Attributes]
]


def extract_attribute_summary(
    attribute_dictionary: AttributeDictionary,
    schema: Optional[AttributesSchema] = None,
) -> Dict[
    MedRecordAttribute,
    Union[TemporalAttributeInfo, NumericAttributeInfo, StringAttributeInfo],
]:
    """Extracts a summary from a node or edge attribute dictionary.

    Args:
        attribute_dictionary (AttributeDictionary): Edges or Nodes and their attributes
            and values.
        schema (Optional[AttributesSchema], optional): Attribute Schema for the group
            nodes or edges. Defaults to None.
        decimal (int): Decimal points to round the numeric values to. Defaults to 2.

    Returns:
        Dict[MedRecordAttribute, Union[TemporalAttributeInfo, NumericAttributeInfo,
            StringAttributeInfo]: Summary of node or edge attributes.
    """
    data = {}

    for dictionary in attribute_dictionary.values():
        for key, value in dictionary.items():
            data.setdefault(key, []).append(value)

    data_dict = {}

    for attribute in sorted(data):
        attribute_values = [value for value in data[attribute] if value is not None]

        if len(attribute_values) == 0:
            attribute_info = {"type": "-", "values": "-"}

        # check if the attribute has as an attribute type defined in the schema
        elif schema and attribute in schema and schema[attribute][1]:
            if schema[attribute][1] == AttributeType.Continuous:
                attribute_info = _extract_numeric_attribute_info(attribute_values)
            elif schema[attribute][1] == AttributeType.Temporal:
                attribute_info = _extract_temporal_attribute_info(attribute_values)
            else:
                attribute_info = _extract_string_attribute_info(
                    attribute_values=attribute_values,
                    long_string_suffix="categories",
                    short_string_prefix="Categories",
                )

        # Without Schema
        else:
            if all(isinstance(value, (int, float)) for value in attribute_values):
                attribute_info = _extract_numeric_attribute_info(attribute_values)
            elif all(isinstance(value, datetime) for value in attribute_values):
                attribute_info = _extract_temporal_attribute_info(attribute_values)
            else:
                attribute_info = _extract_string_attribute_info(
                    attribute_values=[str(value) for value in attribute_values]
                )

        data_dict[attribute] = attribute_info

    return data_dict


def _extract_numeric_attribute_info(
    attribute_values: List[Union[int, float]],
) -> NumericAttributeInfo:
    """Extracts info about attributes with numeric format.

    Args:
        attribute_values (List[Union[int, float]]): List containing attribute values.

    Returns:
        NumericAttributeInfo: Dictionary containg attribute metrics.
    """
    min_value = min(attribute_values)
    max_value = max(attribute_values)
    mean_value = sum(attribute_values) / len(attribute_values)

    # assertion to ensure correct typing
    # never fails, because the series never contains None values and is always numeric
    assert isinstance(min_value, (int, float))
    assert isinstance(max_value, (int, float))
    assert isinstance(mean_value, (int, float))

    return {
        "type": "Continuous",
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
    }


def _extract_temporal_attribute_info(
    attribute_values: List[datetime],
) -> TemporalAttributeInfo:
    """Extracts info about attributes with temporal format.

    Args:
        attribute_values (List[datetime]): List containing temporal attribute values.

    Returns:
        TemporalAttributeInfo: Dictionary containg attribute metrics.
    """
    return {
        "type": "Temporal",
        "min": min(attribute_values),
        "max": max(attribute_values),
    }


def _extract_string_attribute_info(
    attribute_values: List[str],
    short_string_prefix: Literal["Values", "Categories"] = "Values",
    long_string_suffix: str = "unique values",
    max_number_values: int = 5,
    max_line_length: int = 100,
) -> StringAttributeInfo:
    """Extracts info about attributes with string format.

    Args:
        attribute_values (List[str]): List containing attribute values.
        short_string_prefix (Literal["Values", "Categories"], optional): Prefix for
            information string in case of listing all the values. Defaults to "Values".
        long_string_suffix (str, optional): Suffix for attribute information in case of
            too many values to list. Here only the count will be displayed. Defaults to
            "unique values".
        max_number_values (int, optional): Maximum values that should be listed in the
            information string. Defaults to 5.
        max_line_length (int, optional): Maximum line length for the information string.
            Defaults to 100.

    Returns:
        StringAttributeInfo: Dictionary containg attribute metrics.
    """
    values = sorted(set(attribute_values))

    values_string = f"{short_string_prefix}: {', '.join(list(values))}"

    if (len(values) > max_number_values) | (len(values_string) > max_line_length):
        values_string = f"{len(values)} {long_string_suffix}"

    return {
        "type": "Categorical",
        "values": values_string,
    }


def prettify_table(
    data: Dict[Group, AttributeInfo], header: List[str], decimal: int
) -> List[str]:
    """Takes a DataFrame and turns it into a list for displaying a pretty table.

    Args:
        data (Dict[Group, AttributeInfo]): Table information
            stored in a dictionary.
        header (List[str]): Header line consisting of column names for the table.
        decimal (int): Decimal point to round the float values to.

    Returns:
        List[str]: List of lines for printing the table.
    """
    lengths = [len(title) for title in header]

    rows = []

    info_order = ["type", "min", "max", "mean", "values"]

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
