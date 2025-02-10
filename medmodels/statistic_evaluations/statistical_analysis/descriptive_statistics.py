"""Module containing descriptive statistic functions for different attribute types."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import numpy as np

from medmodels.medrecord.schema import AttributesSchema, AttributeType
from medmodels.medrecord.types import Attributes, EdgeIndex, NodeIndex

if TYPE_CHECKING:
    from typing import TypeAlias

    from medmodels.medrecord.types import (
        AttributeInfo,
        MedRecordAttribute,
        MedRecordValue,
        NumericAttributeInfo,
        NumericAttributeStatistics,
        StringAttributeInfo,
        StringAttributeStatistics,
        TemporalAttributeInfo,
        TemporalAttributeStatistics,
    )

AttributeDictionary: TypeAlias = Union[
    Dict[EdgeIndex, Attributes], Dict[NodeIndex, Attributes]
]


def determine_attribute_type(
    attribute_values: List[MedRecordValue],
) -> Optional[AttributeType]:
    """Determines the type of the attribute based on their data types.

    Args:
        attribute_values (List[MedRecordValue]): Extracted values in a list.

    Returns:
        Optional[AttributeType]: Type of the attribute or None if no attribute type can
            be determined.
    """
    if all(isinstance(value, (int, float)) for value in attribute_values):
        return AttributeType.Continuous

    if all(isinstance(value, datetime) for value in attribute_values):
        return AttributeType.Temporal

    # TODO @Laura: add new string type after PR #325
    return None


def extract_attribute_summary(
    attribute_dictionary: AttributeDictionary,
    schema: Optional[AttributesSchema] = None,
) -> Dict[
    MedRecordAttribute,
    AttributeInfo,
]:
    """Extracts a summary from a node or edge attribute dictionary.

    Args:
        attribute_dictionary (AttributeDictionary): Edges or Nodes and their attributes
            and values.
        schema (Optional[AttributesSchema], optional): Attribute Schema for the group
            nodes or edges. Defaults to None.
        decimal (int): Decimal points to round the numeric values to. Defaults to 2.

    Returns:
        Dict[MedRecordAttribute, AttributeInfo]: Summary of node or edge attributes.
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

        else:
            # check if the attribute has as an attribute type defined in the schema
            if schema and attribute in schema and schema[attribute][1]:
                attribute_type = schema[attribute][1]
            else:
                attribute_type = determine_attribute_type(
                    attribute_values=attribute_values
                )

            # handle different attribute types
            if attribute_type == AttributeType.Continuous:
                attribute_info = extract_numeric_attribute_info(attribute_values)

            elif attribute_type == AttributeType.Temporal:
                attribute_info = extract_temporal_attribute_info(attribute_values)

            elif attribute_type == AttributeType.Categorical:
                attribute_info = extract_string_attribute_info(
                    attribute_values=attribute_values,
                    long_string_suffix="categories",
                    short_string_prefix="Categories",
                )
            else:
                attribute_info = extract_string_attribute_info(
                    attribute_values=attribute_values,
                )

        data_dict[attribute] = attribute_info

    return data_dict


def extract_numeric_attribute_info(
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


def extract_numeric_attribute_statistics(
    attribute_values: List[Union[int, float]],
) -> NumericAttributeStatistics:
    """Extracts full statitics summary for numeric attribute values.

    Args:
        attribute_values (List[Union[int, float]]): _description_

    Returns:
        NumericAttributeInfo: _description_
    """
    min_value = min(attribute_values)
    max_value = max(attribute_values)
    mean_value = sum(attribute_values) / len(attribute_values)
    median = np.quantile(attribute_values, 0.5)
    first_quartile = np.quantile(attribute_values, 0.25)
    third_quartile = np.quantile(attribute_values, 0.75)

    assert isinstance(min_value, (int, float))
    assert isinstance(max_value, (int, float))
    assert isinstance(mean_value, (int, float))
    assert isinstance(median, (int, float))
    assert isinstance(first_quartile, (int, float))
    assert isinstance(third_quartile, (int, float))

    return {
        "type": "Continuous",
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "median": median,
        "Q1": first_quartile,
        "Q3": third_quartile,
    }


def calculate_datetime_mean(attribute_values: List[datetime]) -> datetime:
    """Calculate mean of a list of datetimes.

    Args:
        attribute_values (List[datetime]): List containing datetime values.

    Returns:
        datetime: Mean of the datetime list.
    """
    min_value = np.datetime64(min(attribute_values))
    timedeltas = [(np.datetime64(value) - min_value) for value in attribute_values]
    return (min_value + np.mean(timedeltas)).astype(datetime)


def extract_temporal_attribute_info(
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
        "mean": calculate_datetime_mean(attribute_values),
    }


def extract_temporal_attribute_statistics(
    attribute_values: List[datetime],
) -> TemporalAttributeStatistics:
    """Extracts full statistics summary for attributes with temporal format.

    Args:
        attribute_values (List[datetime]): List of datetime values.

    Returns:
        TemporalAttributeStatistics: Dictionary containing detailed attribute metrics.
    """
    median = np.quantile(np.array(attribute_values), 0.5)
    q1 = np.quantile(np.array(attribute_values), 0.25)
    q3 = np.quantile(np.array(attribute_values), 0.75)

    assert isinstance(median, datetime)
    assert isinstance(q1, datetime)
    assert isinstance(q3, datetime)

    return {
        "type": "Temporal",
        "min": min(attribute_values),
        "max": max(attribute_values),
        "mean": calculate_datetime_mean(attribute_values),
        "median": median,
        "Q1": q1,
        "Q3": q3,
    }


def extract_string_attribute_info(
    attribute_values: List[MedRecordAttribute],
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
    values = sorted({str(value) for value in attribute_values})

    values_string = f"{short_string_prefix}: {', '.join(list(values))}"

    if (len(values) > max_number_values) | (len(values_string) > max_line_length):
        values_string = f"{len(values)} {long_string_suffix}"

    return {
        "type": "Categorical",
        "values": values_string,
    }


def extract_categorical_attribute_statistics(
    attribute_values: List[MedRecordAttribute],
) -> StringAttributeStatistics:
    """Extract detailed statistics about categorical attribute values.

    Args:
        attribute_values (List[MedRecordAttribute]): List containing attribute values.

    Returns:
        StringAttributeStatistics: Dictionary with detailed information.
    """
    values = [str(value) for value in attribute_values]
    top = max(values, key=attribute_values.count)
    freq = values.count(top)

    return {
        "type": "Categorical",
        "count": len(set(values)),
        "top": top,
        "freq": freq,
    }
