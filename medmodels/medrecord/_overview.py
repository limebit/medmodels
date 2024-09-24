import copy
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Union

import polars as pl

from medmodels.medrecord.schema import AttributesSchema, AttributeType
from medmodels.medrecord.types import (
    AttributeInfo,
    Attributes,
    EdgeIndex,
    Group,
    MedRecordAttribute,
    MedRecordValue,
    NodeIndex,
    is_medrecord_value,
)


def extract_attribute_summary(
    attribute_dictionary: Union[
        Dict[EdgeIndex, Attributes], Dict[NodeIndex, Attributes]
    ],
    schema: Optional[AttributesSchema] = None,
) -> Dict[
    MedRecordAttribute,
    OrderedDict[
        Literal["min", "max", "mean", "values"],
        MedRecordValue,
    ],
]:
    """Extracts a summary from a node or edge attribute dictionary.


    Example:

     {"diagnosis_time": ["min: 1962-10-21 00:00:00", "max: 2024-04-12 00:00:00"],
      "duration_days": ["min: 0.0", "max: 3416.0", "mean: 405.02"]}


    Args:
        attribute_dict (Union[Dict[EdgeIndex, Attributes], Dict[NodeIndex, Attributes]]):
            Edges or Nodes and their attributes and values.
        schema (Optional[AttributesSchema], optional): Attribute Schema for the group
            nodes or edges. Defaults to None.
        decimal (int): Decimal points to round the numeric values to. Defaults to 2.

    Returns:
        Dict[MedRecordAttribute, List[str]]: Summary of node or edge attributes.
    """
    data = pl.DataFrame(data=[{"id": k, **v} for k, v in attribute_dictionary.items()])

    data_dict = {}

    attributes = sorted([col for col in data.columns if col != "id"])

    if not attributes:
        return {}

    for attribute in attributes:
        attribute_values = data[attribute].drop_nulls()

        if len(attribute_values) == 0:
            attribute_info = {"values": "-"}
        elif schema and attribute in schema and schema[attribute][1]:
            if schema[attribute][1] == AttributeType.Continuous:
                attribute_info = _extract_numeric_attribute_info(attribute_values)
            elif schema[attribute][1] == AttributeType.Temporal:
                attribute_info = _extract_temporal_attribute_info(attribute_values)
            else:
                attribute_info = _extract_string_attribute_info(
                    attribute_series=attribute_values,
                    long_string_suffix="categories",
                    short_string_prefix="Categories",
                )
        # Without Schema
        else:
            if attribute_values.dtype.is_numeric():
                attribute_info = _extract_numeric_attribute_info(attribute_values)
            elif attribute_values.dtype.is_temporal():
                attribute_info = _extract_temporal_attribute_info(attribute_values)
            else:
                attribute_info = _extract_string_attribute_info(
                    attribute_series=attribute_values
                )

        data_dict[str(attribute)] = attribute_info

    return data_dict


def _extract_numeric_attribute_info(
    attribute_series: pl.Series,
) -> OrderedDict[str, MedRecordValue]:
    """Extracts info about attributes with numeric format.

    Args:
        attribute_series (pl.Series): Series containing attribute values.

    Returns:
        List[str]: attribute_info_list
    """
    min = attribute_series.min()
    max = attribute_series.max()
    mean = attribute_series.mean()

    # assertion to ensure correct typing
    assert is_medrecord_value(min)
    assert is_medrecord_value(max)
    assert is_medrecord_value(mean)

    attribute_info = OrderedDict(
        {
            "min": min,
            "max": max,
            "mean": mean,
        }
    )

    return attribute_info


def _extract_temporal_attribute_info(
    attribute_series: pl.Series,
) -> OrderedDict[str, MedRecordValue]:
    """Extracts info about attributes with temporal format.

    Args:
        attribute_series (pl.Series): Series containing attribute values.

    Returns:
        List[str]: attribute_info_list
    """
    if not attribute_series.dtype.is_temporal():
        if attribute_series.dtype.is_numeric():
            attribute_series = attribute_series.cast(pl.Datetime)
        else:
            attribute_series = attribute_series.str.to_datetime()

    attribute_info = OrderedDict(
        {
            "min": min(attribute_series).strftime("%Y-%m-%d %H:%M:%S"),
            "max": max(attribute_series).strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    return attribute_info


def _extract_string_attribute_info(
    attribute_series: pl.Series,
    short_string_prefix: str = "Values",
    long_string_suffix: str = "unique values",
    max_number_values: int = 5,
    max_line_length: int = 100,
) -> OrderedDict[str, MedRecordValue]:
    """Extracts info about attributes with string format.

    Args:
        attribute_series (pl.Series): Series containing attribute values.
        short_string_prefix (str, optional): Prefix for Info string in case of listing
            all the values. Defaults to "Values".
        long_string_suffix (str, optional): Suffix for attribute info in case of too
            many values to list. Here only the count will be displayed.
            Defaults to "unique values".
        max_number_values (int, optional): Maximum values that should be listed in the
            info string. Defaults to 5.
        max_line_length (int, optional): Maximum line length for the info string.
            Defaults to 100.

    Returns:
        str: Attribute info string.
    """
    values = attribute_series.unique().sort()

    values_string = f"{short_string_prefix}: {', '.join(list(values))}"

    if (len(values) > max_number_values) | (len(values_string) > max_line_length):
        values_string = f"{len(values)} {long_string_suffix}"

    return OrderedDict({"values": values_string})


def prettify_table(
    data: Dict[Group, AttributeInfo], header: List[str], decimal: int
) -> List[str]:
    """Takes a DataFrame and turns it into a list for displaying a pretty table.

    Args:
        data (Dict[Group, AttributeInfo]): Table info
            stored in a dictionary.
        header (List[str]): Header line consisting of column names for the table.
        decimal (int): Decimal point to round the float values to.

    Returns:
        List[str]: List of lines for printing the table.
    """
    lengths = [len(title) for title in header]

    rows = []

    for group in data.keys():
        # determine longest group name and count
        lengths[0] = max(len(str(group)), lengths[0])

        lengths[1] = max(len(str(data[group]["count"])), lengths[1])

        row = [str(group), str(data[group]["count"]), "-", "-"]

        # in case of no attribute info, just keep Group name and count
        if not data[group]["attribute"]:
            rows.append(row)
            continue

        for attribute, info in data[group]["attribute"].items():
            lengths[2] = max(len(str(attribute)), lengths[2])

            # display attribute name only once
            first_line = True

            for key in info.keys():
                if not first_line:
                    row[0], row[1] = "", ""

                row[2] = str(attribute) if first_line else ""

                # displaying info based on the type
                if key == "values":
                    row[3] = str(info[key])
                else:
                    if isinstance(info[key], float):
                        row[3] = f"{key}: {info[key]:.{decimal}f}"
                    else:
                        row[3] = f"{key}: {info[key]}"

                lengths[3] = max(len(row[3]), lengths[3])

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
