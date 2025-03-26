"""Module for extracting and displaying an overview of the data in a MedRecord."""

from __future__ import annotations

from enum import Enum
from io import StringIO
from typing import TYPE_CHECKING, Dict, List, Literal, Set, Union

from rich.console import Console
from rich.table import Table

from medmodels.medrecord.types import (
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


def prettify_table(
    data: Dict[Group, AttributeInfo], headers: List[str], decimal: int
) -> Table:
    """Takes a DataFrame and turns it into a list for displaying a pretty table.

    Args:
        data (Dict[Group, AttributeInfo]): Table information
            stored in a dictionary.
        headers (List[str]): Header line consisting of column names for the table.
        decimal (int): Decimal point to round the float values to.

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

    table = Table(show_lines=False)
    for header in headers:
        table.add_column(header)

    info_order = ["min", "max", "values"]

    for group, attributes in data.items():
        group = str(group)
        count = str(attributes.get("count", ""))
        attr_dict = attributes.get("attribute", {})

        for i, (attribute_name, attribute_info) in enumerate(attr_dict.items()):
            attribute_name = str(attribute_name)
            attr_type = attribute_info.get("type", "")
            datatype = attribute_info.get("datatype", "")

            detailed_values = [
                (label, attribute_info[label])
                for label in info_order
                if label in attribute_info
            ]

            if detailed_values:
                label, value = detailed_values[0]
                table.add_row(
                    group if i == 0 else "",
                    count if i == 0 else "",
                    attribute_name,
                    attr_type,
                    datatype,
                    format_detail(label, value),
                )
                for label, value in detailed_values[1:]:
                    table.add_row("", "", "", "", "", format_detail(label, value))
            else:
                table.add_row(
                    group if i == 0 else "",
                    count if i == 0 else "",
                    attribute_name,
                    attr_type,
                    datatype,
                    "-",
                )

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
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)

    console.rule(f"[bold blue]{title1}")
    console.print(table1)
    console.rule(f"[bold green]{title2}")
    console.print(table2)

    return buffer.getvalue()


def values_in_group_attribute(
    medrecord: MedRecord,
    group_query: Union[NodeQuery, EdgeQuery],
    attribute: MedRecordAttribute,
    type: Literal["nodes", "edges"],
) -> Set[MedRecordValue]:
    """Returns all values of a specific attribute in a group.

    Args:
        medrecord (MedRecord): The MedRecord object.
        group_query (Union[NodeQuery, EdgeQuery]): The query to search for the group.
        attribute (MedRecordAttribute): The attribute to search for.
        type (Literal["nodes", "edges"]): The type of the attribute.

    Returns:
        Set[MedRecordValue]: The values of the attribute in the group.
    """

    def query_node(node: NodeOperand) -> None:
        group_query(node)  # pyright: ignore[reportArgumentType]
        node.has_attribute(attribute)

    def query_edge(edge: EdgeOperand) -> None:
        group_query(edge)  # pyright: ignore[reportArgumentType]
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
