"""Module for extracting and displaying an overview of the data in a MedRecord."""

from __future__ import annotations

import re
from enum import Enum, auto
from io import StringIO
from typing import TYPE_CHECKING, Dict, List, Literal, Set, Union

from rich.console import Console
from rich.table import Table
from rich.text import Text

from medmodels.medrecord.types import (
    AnyAttributeInfo,
    Attributes,
    EdgeIndex,
    MedRecordAttribute,
    MedRecordValue,
    NodeIndex,
)

if TYPE_CHECKING:
    from typing import TypeAlias

    from medmodels.medrecord import MedRecord
    from medmodels.medrecord.querying import (
        EdgeOperand,
        EdgeQuery,
        NodeOperand,
        NodeQuery,
    )
    from medmodels.medrecord.types import AttributeInfo, Group, MedRecordAttribute


AttributeDictionary: TypeAlias = Union[
    Dict[EdgeIndex, Attributes], Dict[NodeIndex, Attributes]
]


class Metric(Enum):
    """Enumeration of possible metrics."""

    min = "is_min"
    max = "is_max"


class TypeTable(Enum):
    """Enumeration of possible types for the table."""

    MedRecord = auto()
    Schema = auto()


def style_datatype(raw: str) -> Text:
    """Style the datatype string.

    In case the datatype is "Option", we show it in yellow.
    Otherwise, we return the raw datatype string.

    Args:
        raw (str): The raw datatype string.

    Returns:
        Text: The styled datatype string.
    """
    match = re.match(r"(Option\()(.+)(\))", raw)
    if match:
        styled = Text()
        styled.append(match.group(1), style="yellow")
        styled.append(match.group(2))
        styled.append(match.group(3), style="yellow")
        return styled
    return Text(raw)


def prettify_table(  # noqa: C901
    data: Dict[Group, AttributeInfo],
    headers: List[str],
    decimal: int,
    type_table: TypeTable,
) -> Table:
    """Takes a DataFrame and turns it into a list for displaying a pretty table.

    Args:
        data (Dict[Group, AttributeInfo]): Table information
            stored in a dictionary.
        headers (List[str]): Header line consisting of column names for the table.
        decimal (int): Decimal point to round the float values to.
        type_table (TypeTable): Type of the table to be displayed.
            It can be either MedRecord or Schema.

    Returns:
        Table: The formatted table.
    """

    def format_detail(label: str, value: str) -> str:
        """Format the detail for the table.

        In case the label is "values", return the value as is.
        If the value is a float (or float-like), round it.
        Otherwise, return the label and value as a string.

        Args:
            label (str): The label of the detail.
            value (str|float|int): The value of the detail.

        Returns:
            str: The formatted detail.
        """
        if isinstance(value, float):
            rounded_val = round(value, decimal)
            value_str = str(rounded_val)
        else:
            value_str = str(value)

        return value_str if label == "values" else f"{label}: {value_str}"

    def get_detailed_rows(attribute_info: AnyAttributeInfo) -> List[str]:
        """Returns formatted value rows for MedRecord table.

        Args:
            attribute_info (AnyAttributeInfo]): The attribute information.

        Returns:
            List[str]: The formatted rows.
        """
        rows = []
        detailed_values = [
            (label, attribute_info[label])
            for label in info_order
            if label in attribute_info
        ]

        if not detailed_values:
            rows.append("-")
        else:
            for label, value in detailed_values:
                rows.append(format_detail(label, value))
        return rows

    table = Table(show_lines=False)
    for header in headers:
        table.add_column(header, no_wrap=(header.strip().lower() == "data"))

    if not data:
        table.add_row("No data")
        return table

    info_order = ["min", "max", "values"]
    type_colors = {
        "Continuous": "cyan",
        "Temporal": "magenta",
        "Categorical": "green",
        "Unstructured": "red",
    }

    for group_idx, (group, attributes) in enumerate(sorted(data.items())):
        group = str(group)
        count = str(attributes.get("count", ""))
        attribute_dictionary = attributes.get("attribute", {})

        for i, (attribute_name, attribute_info) in enumerate(
            sorted(attribute_dictionary.items())
        ):
            attribute_name = str(attribute_name)
            attribute_type = attribute_info.get("type", "")
            attribute_type = Text(
                attribute_type, style=type_colors.get(attribute_type, "white")
            )
            group = group if i == 0 else ""
            count = count if i == 0 else ""

            datatype = style_datatype(attribute_info.get("datatype", ""))

            if type_table == TypeTable.MedRecord:
                rows = get_detailed_rows(attribute_info)
                table.add_row(
                    group, count, attribute_name, attribute_type, datatype, rows[0]
                )
                for row in rows[1:]:
                    table.add_row("", "", "", "", "", row)
            else:
                table.add_row(group, attribute_name, attribute_type, datatype)

        if group_idx < len(data) - 1:
            table.add_row(*([""] * len(headers)))

    return table


def join_tables_with_titles(
    title1: str, table1: Table, title2: str, table2: Table
) -> str:
    """Render two rich tables into a single string with section titles.

    Args:
        title1 (str): Title for the first table.
        table1 (Table): First rich Table.
        title2 (str): Title for the second table.
        table2 (Table): Second rich Table.

    Returns:
        str: The combined string representation of the two tables.
    """

    def table_width(table: Table) -> int:
        """Calculate the width of a table.

        Args:
            table (Table): The rich Table.

        Returns:
            int: The width of the table.
        """
        padding = 12
        return (
            sum(len(col.header) for col in table.columns) + len(table.columns) * padding  # pyright: ignore[reportArgumentType]
        )

    # Calculate needed width for both tables
    width1 = table_width(table1)
    width2 = table_width(table2)
    width = max(width1, width2, 80)

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=width)

    console.rule(f"[bold green]{title1}")
    console.print(table1)
    console.rule(f"[bold green]{title2}")
    console.print(table2)

    return buffer.getvalue()


def get_values_from_attribute(
    medrecord: MedRecord,
    query: Union[NodeQuery, EdgeQuery],
    attribute: MedRecordAttribute,
    type: Literal["nodes", "edges"],
) -> Set[MedRecordValue]:
    """Returns all values of a specific attribute from nodes or edges.

    Args:
        medrecord (MedRecord): The MedRecord object.
        query (Union[NodeQuery, EdgeQuery]): The query to get the
            nodes or edges from.
        attribute (MedRecordAttribute): The attribute to search for.
        type (Literal["nodes", "edges"]): The type of the attribute.

    Returns:
        Set[MedRecordValue]: The values of the attribute in the group.
    """

    def query_node(node: NodeOperand) -> None:
        query(node)  # pyright: ignore[reportArgumentType]
        node.has_attribute(attribute)

    def query_edge(edge: EdgeOperand) -> None:
        query(edge)  # pyright: ignore[reportArgumentType]
        edge.has_attribute(attribute)

    return (
        set(medrecord.node[query_node, attribute].values())
        if type == "nodes"
        else set(medrecord.edge[query_edge, attribute].values())
    )


def get_attribute_metric(
    medrecord: MedRecord,
    group_query: Union[NodeQuery, EdgeQuery],
    attribute: MedRecordAttribute,
    metric: Metric,
    type: Literal["nodes", "edges"],
) -> MedRecordValue:
    """Get the attribute metrics for a group.

    Args:
        medrecord (MedRecord): The MedRecord object.
        group_query (Union[NodeQuery, EdgeQuery]): The query to search for the group.
        attribute (MedRecordAttribute): The attribute to search for.
        metric (Metric): The metric to get.
        type (Literal["nodes", "edges"]): The type of the attribute.

    Returns:
        AttributeInfo: The attribute metrics.
    """

    def query_node(node: NodeOperand) -> None:
        group_query(node)  # pyright: ignore[reportArgumentType]
        node.exclude(lambda node: node.attribute(attribute).is_null())
        getattr(node.attribute(attribute), metric.value)()

    def query_edge(edge: EdgeOperand) -> None:
        group_query(edge)  # pyright: ignore[reportArgumentType]
        edge.exclude(lambda edge: edge.attribute(attribute).is_null())
        getattr(edge.attribute(attribute), metric.value)()

    return (
        next(iter(medrecord.node[query_node, attribute].values()))
        if type == "nodes"
        else next(iter(medrecord.edge[query_edge, attribute].values()))
    )
