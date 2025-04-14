"""Builder class for constructing MedRecord instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import medmodels as mm
from medmodels.medrecord.types import (
    EdgeTuple,
    Group,
    GroupInfo,
    NodeIndex,
    NodeTuple,
    PandasEdgeDataFrameInput,
    PandasNodeDataFrameInput,
    PolarsEdgeDataFrameInput,
    PolarsNodeDataFrameInput,
    is_edge_tuple,
    is_edge_tuple_list,
    is_node_tuple,
    is_node_tuple_list,
    is_pandas_edge_dataframe_input,
    is_pandas_edge_dataframe_input_list,
    is_pandas_node_dataframe_input,
    is_pandas_node_dataframe_input_list,
    is_polars_edge_dataframe_input,
    is_polars_edge_dataframe_input_list,
    is_polars_node_dataframe_input,
    is_polars_node_dataframe_input_list,
)

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from medmodels.medrecord.schema import Schema

NodeInputBuilder = Union[
    NodeTuple,
    List[NodeTuple],
    PandasNodeDataFrameInput,
    List[PandasNodeDataFrameInput],
    PolarsNodeDataFrameInput,
    List[PolarsNodeDataFrameInput],
]


def is_node_input_builder(value: object) -> TypeIs[NodeInputBuilder]:
    """Check if a value is a valid node input.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[NodeInput]: True if the value is a valid node input, otherwise False.
    """
    return (
        is_node_tuple(value)
        or is_node_tuple_list(value)
        or is_pandas_node_dataframe_input(value)
        or is_pandas_node_dataframe_input_list(value)
        or is_polars_node_dataframe_input(value)
        or is_polars_node_dataframe_input_list(value)
    )


EdgeInputBuilder = Union[
    EdgeTuple,
    List[EdgeTuple],
    PandasEdgeDataFrameInput,
    List[PandasEdgeDataFrameInput],
    PolarsEdgeDataFrameInput,
    List[PolarsEdgeDataFrameInput],
]


def is_edge_input_builder(value: object) -> TypeIs[EdgeInputBuilder]:
    """Check if a value is a valid edge input.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[EdgeInput]: True if the value is a valid edge input, otherwise False.
    """
    return (
        is_edge_tuple(value)
        or is_edge_tuple_list(value)
        or is_pandas_edge_dataframe_input(value)
        or is_pandas_edge_dataframe_input_list(value)
        or is_polars_edge_dataframe_input(value)
        or is_polars_edge_dataframe_input_list(value)
    )


NodeInputWithGroup = Tuple[NodeInputBuilder, Group]
EdgeInputWithGroup = Tuple[EdgeInputBuilder, Group]


class MedRecordBuilder:
    """A builder class for constructing MedRecord instances.

    Allows for adding nodes, edges, and groups incrementally, and optionally
    specifying a schema.
    """

    _nodes: List[Union[NodeInputBuilder, NodeInputWithGroup]]
    _edges: List[Union[EdgeInputBuilder, EdgeInputWithGroup]]
    _groups: Dict[Group, GroupInfo]
    _schema: Optional[Schema]

    def __init__(self) -> None:
        """Initializes a new MedRecordBuilder instance."""
        self._nodes = []
        self._edges = []
        self._groups = {}
        self._schema = None

    def add_nodes(
        self,
        nodes: NodeInputBuilder,
        *,
        group: Optional[Group] = None,
    ) -> MedRecordBuilder:
        """Adds nodes to the builder.

        Args:
            nodes (NodeInput): Nodes to add.
            group (Optional[Group], optional): Group to associate with the nodes.

        Returns:
            MedRecordBuilder: The current instance of the builder.
        """
        if group is not None:
            self._nodes.append((nodes, group))
        else:
            self._nodes.append(nodes)

        return self

    def add_edges(
        self,
        edges: EdgeInputBuilder,
        *,
        group: Optional[Group] = None,
    ) -> MedRecordBuilder:
        """Adds edges to the builder.

        Args:
            edges (EdgeInput): Edges to add.
            group (Optional[Group], optional): Group to associate with the edges.

        Returns:
            MedRecordBuilder: The current instance of the builder.
        """
        if group is not None:
            self._edges.append((edges, group))
        else:
            self._edges.append(edges)

        return self

    def add_group(
        self, group: Group, *, nodes: Optional[List[NodeIndex]] = None
    ) -> MedRecordBuilder:
        """Adds a group to the builder with an optional list of nodes.

        Args:
            group (Group): The name of the group to add.
            nodes (List[NodeIndex], optional): Node indices to add to the group.

        Returns:
            MedRecordBuilder: The current instance of the builder.
        """
        if nodes is None:
            nodes = []
        self._groups[group] = {"nodes": nodes, "edges": []}
        return self

    def with_schema(self, schema: Schema) -> MedRecordBuilder:
        """Specifies a schema for the MedRecord.

        Args:
            schema (Schema): The schema to apply.

        Returns:
            MedRecordBuilder: The current instance of the builder.
        """
        self._schema = schema
        return self

    def build(self) -> mm.MedRecord:
        """Constructs a MedRecord instance from the builder's configuration.

        Returns:
            MedRecord: The constructed MedRecord instance.
        """
        medrecord = mm.MedRecord()

        for node in self._nodes:
            if is_node_input_builder(node):
                medrecord.add_nodes(node)
                continue

            group = node[1]
            node = node[0]

            medrecord.add_nodes(node, group)

        for edge in self._edges:
            if is_edge_input_builder(edge):
                medrecord.add_edges(edge)
                continue

            group = edge[1]
            edge = edge[0]

            medrecord.add_edges(edge, group)

        for group in self._groups:
            medrecord.add_group(
                group, self._groups[group]["nodes"], self._groups[group]["edges"]
            )

        if self._schema is not None:
            medrecord.set_schema(self._schema)

        return medrecord
