from typing import Dict, List, Optional, Union

import polars as pl

from medmodels.medrecord.schema import AttributesSchema, AttributeType
from medmodels.medrecord.types import Attributes, EdgeIndex, NodeIndex


def extract_attribute_summary(
    attribute_dict: Union[Dict[EdgeIndex, Attributes], Dict[NodeIndex, Attributes]],
    schema: Optional[AttributesSchema] = None,
    decimal: int = 2,
) -> pl.DataFrame:
    """Extracts a summary from a node or edge attribute dictionary.


    Example:
    ┌────────────────┬──────────────────────────┐
    │ Attribute      ┆ Info                     │
    │ ---            ┆ ---                      │
    │ str            ┆ str                      │
    ╞════════════════╪══════════════════════════╡
    │ diagnosis_time ┆ min: 1962-10-21 00:00:00 │
    │ diagnosis_time ┆ max: 2024-04-12 00:00:00 │
    │ duration_days  ┆ min: 0.0                 │
    │ duration_days  ┆ max: 3416.0              │
    │ duration_days  ┆ mean: 405.02             │
    └────────────────┴──────────────────────────┘


    Args:
        attribute_dict (Union[Dict[EdgeIndex, Attributes], Dict[NodeIndex, Attributes]]):
            Edges or Nodes and their attributes and values.
        schema (Optional[AttributesSchema], optional): Attribute Schema for the group
            nodes or edges. Defaults to None.
        decimal (int): Decimal points to round the numeric values. Defaults to 2.

    Returns:
        pl.DataFrame: Summary of node or edge attributes.
    """
    data = pl.DataFrame(data=[{"id": k, **v} for k, v in attribute_dict.items()])

    data_dict = {
        "Attribute": [],
        "Info": [],
    }

    attributes = [col for col in data.columns if col != "id"]
    attributes.sort()

    if not attributes:
        return pl.DataFrame({"Attribute": ["-"], "Info": ["-"]})

    for attribute in attributes:
        if schema and attribute in schema:
            if schema[attribute][1] == AttributeType.Continuous:
                attribute_info = [
                    f"min: {data[attribute].min()}",
                    f"max: {data[attribute].max()}",
                    f"mean: {data[attribute].mean():.{decimal}f}",
                ]

            elif schema[attribute][1] == AttributeType.Temporal:
                time_attribute = data[attribute].str.to_datetime()
                attribute_info = [
                    f"min: {min(time_attribute).strftime('%Y-%m-%d %H:%M:%S')}",
                    f"max: {max(time_attribute).strftime('%Y-%m-%d %H:%M:%S')}",
                ]

            elif schema[attribute][1] == AttributeType.Categorical:
                attribute_info = [
                    _extract_string_attribute_info(
                        attribute_series=data[attribute],
                        long_string_suffix="categories",
                        short_string_prefix="Categories",
                    )
                ]
            else:
                attribute_info = [
                    _extract_string_attribute_info(attribute_series=data[attribute])
                ]

        ## Without Schema
        else:
            if data[attribute].dtype.is_numeric():
                attribute_info = [
                    f"min: {data[attribute].min()}",
                    f"max: {data[attribute].max()}",
                    f"mean: {data[attribute].mean():.{decimal}f}",
                ]
            elif data[attribute].dtype.is_temporal():
                attribute_info = [
                    f"min: {min(data[attribute]).strftime('%Y-%m-%d %H:%M:%S')}",
                    f"max: {max(data[attribute]).strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            else:
                attribute_info = [
                    _extract_string_attribute_info(attribute_series=data[attribute])
                ]

        data_dict["Attribute"].extend([attribute] * len(attribute_info))
        data_dict["Info"].extend(attribute_info)

    return pl.DataFrame(data_dict)


def _extract_string_attribute_info(
    attribute_series: pl.Series,
    short_string_prefix: str = "Values",
    long_string_suffix: str = "unique values",
    max_number_values: int = 5,
    max_line_length: int = 100,
) -> str:
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
    values = attribute_series.drop_nulls()

    if len(values) == 0:
        return "-"

    values = values.unique().sort()
    values_string = f"{short_string_prefix}: {', '.join(list(values))}"
    if (len(values) > 5) | (len(values_string) > 100):
        return f"{len(values)} {long_string_suffix}"
    else:
        return values_string


def prettify_table(table_info: pl.DataFrame) -> List[str]:
    """Takes a DataFrame and turns it into a list for printing a pretty table.

    Args:
        table_info (pl.DataFrame): Table in DataFrame format.

    Returns:
        List[str]: List of lines for printing the table.
    """
    lengths = []
    table_info = table_info.with_columns(pl.exclude(pl.Utf8).cast(str))
    for col in table_info.columns:
        lengths.append(max(len(max(table_info[col], key=len)), len(col)))

    print_table = []

    print_table.append("-" * (sum(lengths) + len(lengths) + 1))

    print_table.append(
        " ".join([f"{head:<{lengths[i]}}" for i, head in enumerate(table_info.columns)])
    )

    print_table.append("-" * (sum(lengths) + len(lengths) + 1))

    old_row = [""] * len(table_info.columns)
    for row in table_info.rows():
        print_row = ""
        for i, elem in enumerate(row):
            if (elem == old_row[i]) & (row[:i] == old_row[:i]):
                elem = ""
            print_row += f"{elem: <{lengths[i]}} "
        print_table.append(print_row)
        old_row = row

    print_table.append("-" * (sum(lengths) + len(lengths) + 1))

    return print_table
