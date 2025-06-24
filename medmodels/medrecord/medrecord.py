"""MedRecord class for managing medical records using nodes and edges.

The `MedRecord` class is the core component of the `medmodels` package, providing
methods to create, manage, and query medical records represented through node and
edge data structures. It allows for the dynamic addition and removal of nodes and
edges, with the capability to attach, remove, and query attributes on both.

The class supports instantiation from various data formats, enhancing flexibility and
interoperability. Additionally, it offers mechanisms to group nodes and edges for
simplified management and efficient querying.
"""

from __future__ import annotations

import sys
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Sequence, Union, overload

import polars as pl

from medmodels._medmodels import PyEdgeOperand, PyMedRecord, PyNodeOperand
from medmodels.medrecord._overview import extract_attribute_summary, prettify_table
from medmodels.medrecord.builder import MedRecordBuilder
from medmodels.medrecord.indexers import EdgeIndexer, NodeIndexer
from medmodels.medrecord.querying import (
    EdgeAttributesTreeGroupOperand,
    EdgeAttributesTreeGroupQueryResult,
    EdgeAttributesTreeOperand,
    EdgeAttributesTreeQueryResult,
    EdgeIndexGroupOperand,
    EdgeIndexGroupQueryResult,
    EdgeIndexOperand,
    EdgeIndexQuery,
    EdgeIndexQueryResult,
    EdgeIndicesGroupOperand,
    EdgeIndicesGroupQueryResult,
    EdgeIndicesOperand,
    EdgeIndicesQuery,
    EdgeIndicesQueryResult,
    EdgeMultipleAttributesWithIndexGroupOperand,
    EdgeMultipleAttributesWithIndexGroupQueryResult,
    EdgeMultipleAttributesWithIndexOperand,
    EdgeMultipleAttributesWithIndexQueryResult,
    EdgeMultipleAttributesWithoutIndexOperand,
    EdgeMultipleAttributesWithoutIndexQueryResult,
    EdgeMultipleValuesWithIndexGroupOperand,
    EdgeMultipleValuesWithIndexGroupQueryResult,
    EdgeMultipleValuesWithIndexOperand,
    EdgeMultipleValuesWithIndexQueryResult,
    EdgeMultipleValuesWithoutIndexOperand,
    EdgeMultipleValuesWithoutIndexQueryResult,
    EdgeOperand,
    EdgeQuery,
    EdgeSingleAttributeWithIndexGroupOperand,
    EdgeSingleAttributeWithIndexGroupQueryResult,
    EdgeSingleAttributeWithIndexOperand,
    EdgeSingleAttributeWithIndexQueryResult,
    EdgeSingleAttributeWithoutIndexGroupOperand,
    EdgeSingleAttributeWithoutIndexGroupQueryResult,
    EdgeSingleAttributeWithoutIndexOperand,
    EdgeSingleAttributeWithoutIndexQueryResult,
    EdgeSingleValueWithIndexGroupOperand,
    EdgeSingleValueWithIndexGroupQueryResult,
    EdgeSingleValueWithIndexOperand,
    EdgeSingleValueWithIndexQueryResult,
    EdgeSingleValueWithoutIndexGroupOperand,
    EdgeSingleValueWithoutIndexGroupQueryResult,
    EdgeSingleValueWithoutIndexOperand,
    EdgeSingleValueWithoutIndexQueryResult,
    NodeAttributesTreeGroupOperand,
    NodeAttributesTreeGroupQueryResult,
    NodeAttributesTreeOperand,
    NodeAttributesTreeQueryResult,
    NodeIndexGroupOperand,
    NodeIndexGroupQueryResult,
    NodeIndexOperand,
    NodeIndexQuery,
    NodeIndexQueryResult,
    NodeIndicesGroupOperand,
    NodeIndicesGroupQueryResult,
    NodeIndicesOperand,
    NodeIndicesQuery,
    NodeIndicesQueryResult,
    NodeMultipleAttributesWithIndexGroupOperand,
    NodeMultipleAttributesWithIndexGroupQueryResult,
    NodeMultipleAttributesWithIndexOperand,
    NodeMultipleAttributesWithIndexQueryResult,
    NodeMultipleAttributesWithoutIndexOperand,
    NodeMultipleAttributesWithoutIndexQueryResult,
    NodeMultipleValuesWithIndexGroupOperand,
    NodeMultipleValuesWithIndexGroupQueryResult,
    NodeMultipleValuesWithIndexOperand,
    NodeMultipleValuesWithIndexQueryResult,
    NodeMultipleValuesWithoutIndexOperand,
    NodeMultipleValuesWithoutIndexQueryResult,
    NodeOperand,
    NodeQuery,
    NodeSingleAttributeWithIndexGroupOperand,
    NodeSingleAttributeWithIndexGroupQueryResult,
    NodeSingleAttributeWithIndexOperand,
    NodeSingleAttributeWithIndexQueryResult,
    NodeSingleAttributeWithoutIndexGroupOperand,
    NodeSingleAttributeWithoutIndexGroupQueryResult,
    NodeSingleAttributeWithoutIndexOperand,
    NodeSingleAttributeWithoutIndexQueryResult,
    NodeSingleValueWithIndexGroupOperand,
    NodeSingleValueWithIndexGroupQueryResult,
    NodeSingleValueWithIndexOperand,
    NodeSingleValueWithIndexQueryResult,
    NodeSingleValueWithoutIndexGroupOperand,
    NodeSingleValueWithoutIndexGroupQueryResult,
    NodeSingleValueWithoutIndexOperand,
    NodeSingleValueWithoutIndexQueryResult,
    PyQueryReturnOperand,
    QueryResult,
    QueryReturnOperand,
)
from medmodels.medrecord.schema import Schema
from medmodels.medrecord.types import (
    AttributeInfo,
    Attributes,
    EdgeIndex,
    EdgeIndexInputList,
    EdgeInput,
    EdgeTuple,
    Group,
    GroupInfo,
    GroupInputList,
    NodeIndex,
    NodeIndexInputList,
    NodeInput,
    NodeTuple,
    PandasEdgeDataFrameInput,
    PandasNodeDataFrameInput,
    PolarsEdgeDataFrameInput,
    PolarsNodeDataFrameInput,
    is_edge_tuple,
    is_node_tuple,
    is_pandas_edge_dataframe_input,
    is_pandas_edge_dataframe_input_list,
    is_pandas_node_dataframe_input,
    is_pandas_node_dataframe_input_list,
    is_polars_edge_dataframe_input,
    is_polars_edge_dataframe_input_list,
    is_polars_node_dataframe_input,
    is_polars_node_dataframe_input_list,
)


def process_nodes_dataframe(
    nodes: PandasNodeDataFrameInput,
) -> PolarsNodeDataFrameInput:
    """Converts a PandasNodeDataFrameInput to a PolarsNodeDataFrameInput.

    Args:
        nodes (PandasNodeDataFrameInput): A tuple of the Pandas DataFrame and index
            index column name.

    Returns:
        PolarsNodeDataFrameInput: A tuple of the Polars DataFrame and index column name.
    """
    nodes_polars = pl.from_pandas(nodes[0])
    return nodes_polars, nodes[1]


def process_edges_dataframe(
    edges: PandasEdgeDataFrameInput,
) -> PolarsEdgeDataFrameInput:
    """Converts a PandasEdgeDataFrameInput to a PolarsEdgeDataFrameInput.

    Args:
        edges (PandasEdgeDataFrameInput): A tuple of the Pandas DataFrame,
            source index, and target index column names.

    Returns:
        PolarsEdgeDataFrameInput: A tuple of the Polars DataFrame, source index, and
            target index column names.
    """
    edges_polars = pl.from_pandas(edges[0])
    return edges_polars, edges[1], edges[2]


def _convert_queryreturnoperand_to_pyqueryreturnoperand(
    operand: QueryReturnOperand,
) -> PyQueryReturnOperand:
    if isinstance(
        operand,
        (
            NodeAttributesTreeOperand,
            NodeAttributesTreeGroupOperand,
            EdgeAttributesTreeOperand,
            EdgeAttributesTreeGroupOperand,
        ),
    ):
        return operand._attributes_tree_operand
    if isinstance(
        operand,
        (
            NodeMultipleAttributesWithIndexOperand,
            NodeMultipleAttributesWithIndexGroupOperand,
            NodeMultipleAttributesWithoutIndexOperand,
            EdgeMultipleAttributesWithIndexOperand,
            EdgeMultipleAttributesWithIndexGroupOperand,
            EdgeMultipleAttributesWithoutIndexOperand,
        ),
    ):
        return operand._multiple_attributes_operand
    if isinstance(
        operand,
        (
            NodeSingleAttributeWithIndexOperand,
            NodeSingleAttributeWithIndexGroupOperand,
            NodeSingleAttributeWithoutIndexOperand,
            NodeSingleAttributeWithoutIndexGroupOperand,
            EdgeSingleAttributeWithIndexOperand,
            EdgeSingleAttributeWithIndexGroupOperand,
            EdgeSingleAttributeWithoutIndexOperand,
            EdgeSingleAttributeWithoutIndexGroupOperand,
        ),
    ):
        return operand._single_attribute_operand
    if isinstance(operand, (EdgeIndicesOperand, EdgeIndicesGroupOperand)):
        return operand._edge_indices_operand
    if isinstance(operand, (EdgeIndexOperand, EdgeIndexGroupOperand)):
        return operand._edge_index_operand
    if isinstance(operand, (NodeIndicesOperand, NodeIndicesGroupOperand)):
        return operand._node_indices_operand
    if isinstance(operand, (NodeIndexOperand, NodeIndexGroupOperand)):
        return operand._node_index_operand
    if isinstance(
        operand,
        (
            NodeMultipleValuesWithIndexOperand,
            NodeMultipleValuesWithIndexGroupOperand,
            NodeMultipleValuesWithoutIndexOperand,
            EdgeMultipleValuesWithIndexOperand,
            EdgeMultipleValuesWithIndexGroupOperand,
            EdgeMultipleValuesWithoutIndexOperand,
        ),
    ):
        return operand._multiple_values_operand
    if isinstance(operand, Sequence):
        return [
            _convert_queryreturnoperand_to_pyqueryreturnoperand(operand)
            for operand in operand
        ]

    return operand._single_value_operand


class OverviewTable:
    """Class for the node/edge group overview table."""

    data: Dict[Group, AttributeInfo]
    group_header: str
    decimal: int

    def __init__(
        self,
        data: Dict[Group, AttributeInfo],
        group_header: str,
        decimal: int,
    ) -> None:
        """Initializes the OverviewTable class.

        Args:
            data (Dict[Group, AttributeInfo]): Dictionary containing attribute info for
                edges/nodes.
            group_header (str): Header for group column, i.e. 'Group Nodes'.
            decimal (int): Decimal point to round the float values to.
        """
        self.data = data
        self.group_header = group_header
        self.decimal = decimal

    def __repr__(self) -> str:
        """Returns a string representation of the group nodes/ edges overview."""
        header = [self.group_header, "count", "attribute", "type", "data"]

        return "\n".join(prettify_table(self.data, header=header, decimal=self.decimal))


class EdgesDirected(Enum):
    """Enum for specifying whether edges are considered directed or undirected."""

    DIRECTED = auto()
    UNDIRECTED = auto()


class MedRecord:
    """A class to manage medical records with node and edge data structures.

    Provides methods to create instances from different data formats, manage node and
    edge attributes, and perform operations like adding or removing nodes and edges.
    """

    _medrecord: PyMedRecord

    def __init__(self) -> None:
        """Initializes a MedRecord instance with an underlying PyMedRecord object."""
        self._medrecord = PyMedRecord()

    @staticmethod
    def builder() -> MedRecordBuilder:
        """Creates a MedRecordBuilder instance to build a MedRecord.

        Returns:
            MedRecordBuilder: A new builder instance.
        """
        return MedRecordBuilder()

    @classmethod
    def with_schema(cls, schema: Schema) -> MedRecord:
        """Creates a MedRecord instance with the specified schema.

        Args:
            schema (Schema): The schema to apply to the MedRecord.

        Returns:
            MedRecord: A new instance with the provided schema.
        """
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.with_schema(schema._schema)
        return medrecord

    @classmethod
    def from_tuples(
        cls,
        nodes: Sequence[NodeTuple],
        edges: Optional[Sequence[EdgeTuple]] = None,
    ) -> MedRecord:
        """Creates a MedRecord instance from lists of node and edge tuples.

        Nodes and edges are specified as lists of tuples. Each node tuple contains a
        node index and attributes. Each edge tuple includes indices of the source and
        target nodes and edge attributes.

        Args:
            nodes (Sequence[NodeTuple]): Sequence of node tuples.
            edges (Optional[Sequence[EdgeTuple]]): Sequence of edge tuples.

        Returns:
            MedRecord: A new instance created from the provided tuples.
        """
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_tuples(nodes, edges)
        return medrecord

    @classmethod
    def from_pandas(
        cls,
        nodes: Union[PandasNodeDataFrameInput, List[PandasNodeDataFrameInput]],
        edges: Optional[
            Union[PandasEdgeDataFrameInput, List[PandasEdgeDataFrameInput]]
        ] = None,
    ) -> MedRecord:
        """Creates a MedRecord from Pandas DataFrames of nodes and optionally edges.

        Accepts a tuple or a list of tuples for nodes and edges. Each node tuple
        consists of a Pandas DataFrame and an index column. Edge tuples include
        a DataFrame and index columns for source and target nodes.

        Args:
            nodes (Union[PolarsNodeDataFrameInput, List[PolarsNodeDataFrameInput]]):
                Node DataFrame(s).
            edges (Optional[Union[PolarsEdgeDataFrameInput, List[PolarsEdgeDataFrameInput]]]):
                Edge DataFrame(s), optional.

        Returns:
            MedRecord: A new instance from the provided DataFrames.
        """  # noqa: W505
        if edges is None:
            medrecord = cls.__new__(cls)
            medrecord._medrecord = PyMedRecord.from_nodes_dataframes(
                [process_nodes_dataframe(nodes_df) for nodes_df in nodes]
                if isinstance(nodes, list)
                else [process_nodes_dataframe(nodes)]
            )
            return medrecord

        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_dataframes(
            (
                [process_nodes_dataframe(nodes_df) for nodes_df in nodes]
                if isinstance(nodes, list)
                else [process_nodes_dataframe(nodes)]
            ),
            (
                [process_edges_dataframe(edges_df) for edges_df in edges]
                if isinstance(edges, list)
                else [process_edges_dataframe(edges)]
            ),
        )
        return medrecord

    @classmethod
    def from_polars(
        cls,
        nodes: Union[PolarsNodeDataFrameInput, List[PolarsNodeDataFrameInput]],
        edges: Optional[
            Union[PolarsEdgeDataFrameInput, List[PolarsEdgeDataFrameInput]]
        ] = None,
    ) -> MedRecord:
        """Creates a MedRecord from Polars DataFrames of nodes and optionally edges.

        Accepts a tuple or a list of tuples for nodes and edges. Each node tuple
        consists of a Polars DataFrame and an index column. Edge tuples include
        a DataFrame and index columns for source and target nodes.

        Args:
            nodes (Union[PolarsNodeDataFrameInput, List[PolarsNodeDataFrameInput]]):
                Node data.
            edges (Optional[Union[PolarsEdgeDataFrameInput, List[PolarsEdgeDataFrameInput]]]):
                Edge data, optional.

        Returns:
            MedRecord: A new instance from the provided Polars DataFrames.
        """  # noqa: W505
        if edges is None:
            medrecord = cls.__new__(cls)
            medrecord._medrecord = PyMedRecord.from_nodes_dataframes(
                nodes if isinstance(nodes, list) else [nodes]
            )
            return medrecord

        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_dataframes(
            nodes if isinstance(nodes, list) else [nodes],
            edges if isinstance(edges, list) else [edges],
        )
        return medrecord

    @classmethod
    def from_simple_example_dataset(cls) -> MedRecord:
        """Creates a MedRecord instance from a simple example dataset.

        The example dataset was created using the Synthea™ Patient Generator:
        https://github.com/synthetichealth/synthea, and it includes 5 patients with
        their diagnoses, prescriptions and procedures.

        Returns:
            MedRecord: A new instance created from the example dataset.
        """
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_simple_example_dataset()
        return medrecord

    @classmethod
    def from_advanced_example_dataset(cls) -> MedRecord:
        """Creates a MedRecord instance from an advanced example dataset.

        The example dataset was created using the Synthea™ Patient Generator:
        https://github.com/synthetichealth/synthea, and it includes 600 patients with
        their diagnoses, prescriptions, procedures, and death events.

        Returns:
            MedRecord: A new instance created from the advanced example dataset.
        """
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_advanced_example_dataset()
        return medrecord

    @classmethod
    def from_admissions_example_dataset(cls) -> MedRecord:
        """Creates a MedRecord instance from an admissions example dataset.

        The example dataset was created using the Synthea™ Patient Generator:
        https://github.com/synthetichealth/synthea, and it includes 100 patients with
        their admissions, diagnoses, prescriptions, procedures, and death events.

        Returns:
            MedRecord: A new instance created from the admissions example dataset.
        """
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_admissions_example_dataset()
        return medrecord

    @classmethod
    def from_ron(cls, path: str) -> MedRecord:
        """Creates a MedRecord instance from a RON file.

        Reads node and edge data from a RON file specified by the path and creates a new
        MedRecord instance using this data.

        Args:
            path (str): Path to the RON file.

        Returns:
            MedRecord: A new instance created from the RON file.
        """
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_ron(path)
        return medrecord

    def to_ron(self, path: str) -> None:
        """Writes the MedRecord instance to a RON file.

        Serializes the MedRecord instance to a RON file at the specified path.

        Args:
            path (str): Path where the RON file will be written.
        """
        self._medrecord.to_ron(path)

    def get_schema(self) -> Schema:
        """Returns a copy of the MedRecords schema.

        Returns:
            Schema: The schema of the MedRecord.
        """
        return Schema._from_py_schema(self._medrecord.get_schema())

    def set_schema(self, schema: Schema) -> None:
        """Sets the schema of the MedRecord instance.

        Args:
            schema (Schema): The new schema to apply.
        """
        self._medrecord.set_schema(schema._schema)

    def freeze_schema(self) -> None:
        """Freezes the schema. No changes are automatically inferred."""
        self._medrecord.freeze_schema()

    def unfreeze_schema(self) -> None:
        """Unfreezes the schema. Changes are automatically inferred."""
        self._medrecord.unfreeze_schema()

    @property
    def nodes(self) -> List[NodeIndex]:
        """Lists the node indices in the MedRecord instance.

        Returns a list of all node indices currently managed by the MedRecord instance.

        Returns:
            List[NodeIndex]: A list of node indices.
        """
        return self._medrecord.nodes

    @property
    def node(self) -> NodeIndexer:
        """Provides access to node attributes within the MedRecord via an indexer.

        Facilitates querying, accessing, manipulating, and setting node attributes using
        various indexing methods. Supports conditions and ranges for more
        complex queries.

        Returns:
            NodeIndexer: An object for manipulating and querying node attributes.
        """
        return NodeIndexer(self)

    @property
    def edges(self) -> List[EdgeIndex]:
        """Lists the edge indices in the MedRecord instance.

        Returns a list of all edge indices currently managed by the MedRecord instance.

        Returns:
            List[EdgeIndex]: A list of edge indices.
        """
        return self._medrecord.edges

    @property
    def edge(self) -> EdgeIndexer:
        """Provides access to edge attributes within the MedRecord via an indexer.

        Facilitates querying, accessing, manipulating, and setting edge attributes using
        various indexing methods. Supports conditions and ranges for more
        complex queries.

        Returns:
            EdgeIndexer: An object for manipulating and querying edge attributes.
        """
        return EdgeIndexer(self)

    @property
    def groups(self) -> List[Group]:
        """Lists the groups in the MedRecord instance.

        Returns a list of all groups currently defined within the MedRecord instance.

        Returns:
            List[Group]: A list of groups.
        """
        return self._medrecord.groups

    @overload
    def group(self, group: Group) -> GroupInfo: ...

    @overload
    def group(self, group: GroupInputList) -> Dict[Group, GroupInfo]: ...

    def group(
        self, group: Union[Group, GroupInputList]
    ) -> Union[GroupInfo, Dict[Group, GroupInfo]]:
        """Returns the node and edge indices associated with the specified group/s.

        If a single group is specified, returns a list of node and edge indices
        for that group.
        If multiple groups are specified, returns a dictionary with each group name
        mapping to its list of node and edge indices indices.

        Args:
            group (Union[Group, List[Group]]): One or more group names.

        Returns:
            Union[GroupInfo, Dict[Group, GroupInfo]]: Node and edge indices for
                the specified group(s).
        """
        if isinstance(group, list):
            nodes_in_group = self._medrecord.nodes_in_group(group)
            edges_in_group = self._medrecord.edges_in_group(group)

            return {
                group: {"nodes": nodes_in_group[group], "edges": edges_in_group[group]}
                for group in group
            }

        nodes_in_group = self._medrecord.nodes_in_group([group])
        edges_in_group = self._medrecord.edges_in_group([group])

        return {"nodes": nodes_in_group[group], "edges": edges_in_group[group]}

    @overload
    def outgoing_edges(
        self, node: Union[NodeIndex, NodeIndexQuery]
    ) -> List[EdgeIndex]: ...

    @overload
    def outgoing_edges(
        self, node: Union[NodeIndexInputList, NodeIndicesQuery]
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...

    def outgoing_edges(
        self,
        node: Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery],
    ) -> Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]:
        """Lists the outgoing edges of the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of its outgoing edge indices.
        If multiple nodes are specified, returns a dictionary mapping each node index to
        its list of outgoing edge indices.

        Args:
            node (Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]):
                One or more node indices or a node query.

        Returns:
            Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]: Outgoing
                edge indices for each specified node.
        """  # noqa: W505
        if isinstance(node, Callable):
            query_result = self.query_nodes(node)

            if isinstance(query_result, list):
                return self._medrecord.outgoing_edges(query_result)
            if query_result is not None:
                return self._medrecord.outgoing_edges([query_result])[query_result]

            return []

        indices = self._medrecord.outgoing_edges(
            node if isinstance(node, list) else [node]
        )

        if isinstance(node, list):
            return indices

        return indices[node]

    @overload
    def incoming_edges(
        self, node: Union[NodeIndex, NodeIndexQuery]
    ) -> List[EdgeIndex]: ...

    @overload
    def incoming_edges(
        self, node: Union[NodeIndexInputList, NodeIndicesQuery]
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...

    def incoming_edges(
        self,
        node: Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery],
    ) -> Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]:
        """Lists the incoming edges of the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of its incoming edge indices.
        If multiple nodes are specified, returns a dictionary mapping each node index to
        its list of incoming edge indices.

        Args:
            node (Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]):
                One or more node indices or a node query.

        Returns:
            Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]: Incoming
                edge indices for each specified node.
        """  # noqa: W505
        if isinstance(node, Callable):
            query_result = self.query_nodes(node)

            if isinstance(query_result, list):
                return self._medrecord.incoming_edges(query_result)
            if query_result is not None:
                return self._medrecord.incoming_edges([query_result])[query_result]

            return []

        indices = self._medrecord.incoming_edges(
            node if isinstance(node, list) else [node]
        )

        if isinstance(node, list):
            return indices

        return indices[node]

    @overload
    def edge_endpoints(
        self, edge: Union[EdgeIndex, EdgeIndexQuery]
    ) -> tuple[NodeIndex, NodeIndex]: ...

    @overload
    def edge_endpoints(
        self, edge: Union[EdgeIndexInputList, EdgeIndicesQuery]
    ) -> Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]: ...

    def edge_endpoints(
        self,
        edge: Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery],
    ) -> Union[
        tuple[NodeIndex, NodeIndex], Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]
    ]:
        """Retrieves the source and target nodes of the specified edge(s).

        If a single edge index is provided, returns a tuple of
        node indices (source, target). If multiple edges are specified, returns
        a dictionary mapping each edge index to its tuple of node indices.

        Args:
            edge (Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery]):
                One or more edge indices or an edge query.

        Returns:
            Union[tuple[NodeIndex, NodeIndex],
                Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]]:
                Tuple of node indices or a dictionary mapping each edge to its
                node indices.

        Raises:
            IndexError: If the query returned no results.
        """  # noqa: W505
        if isinstance(edge, Callable):
            query_result = self.query_edges(edge)

            if isinstance(query_result, list):
                return self._medrecord.edge_endpoints(query_result)
            if query_result is not None:
                return self._medrecord.edge_endpoints([query_result])[query_result]

            msg = "The query returned no results"
            raise IndexError(msg)

        endpoints = self._medrecord.edge_endpoints(
            edge if isinstance(edge, list) else [edge]
        )

        if isinstance(edge, list):
            return endpoints

        return endpoints[edge]

    def edges_connecting(
        self,
        source_node: Union[
            NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery
        ],
        target_node: Union[
            NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery
        ],
        directed: EdgesDirected = EdgesDirected.DIRECTED,
    ) -> List[EdgeIndex]:
        """Retrieves the edges connecting the specified source and target nodes.

        If a NodeOperation is provided for either the source or target nodes, it is
        first evaluated to obtain the corresponding node indices. The method then
        returns a list of edge indices that connect the specified source and
        target nodes.

        Args:
            source_node (Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]):
                The index or indices of the source node(s), or a node query to
                select source nodes.
            target_node (Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]):
                The index or indices of the target node(s), or a node query to
                select target nodes.
            directed (EdgesDirected, optional): Whether to consider edges as directed.
                Defaults to EdgesDirected.DIRECTED.

        Returns:
            List[EdgeIndex]: A list of edge indices connecting the specified source and
                target nodes.
        """  # noqa: W505
        if isinstance(source_node, Callable):
            query_result = self.query_nodes(source_node)

            if query_result is None:
                return []

            source_node = query_result

        if isinstance(target_node, Callable):
            query_result = self.query_nodes(target_node)

            if query_result is None:
                return []

            target_node = query_result

        if directed == EdgesDirected.DIRECTED:
            return self._medrecord.edges_connecting(
                (source_node if isinstance(source_node, list) else [source_node]),
                (target_node if isinstance(target_node, list) else [target_node]),
            )
        return self._medrecord.edges_connecting_undirected(
            (source_node if isinstance(source_node, list) else [source_node]),
            (target_node if isinstance(target_node, list) else [target_node]),
        )

    @overload
    def remove_nodes(self, nodes: Union[NodeIndex, NodeIndexQuery]) -> Attributes: ...

    @overload
    def remove_nodes(
        self, nodes: Union[NodeIndexInputList, NodeIndicesQuery]
    ) -> Dict[NodeIndex, Attributes]: ...

    def remove_nodes(
        self,
        nodes: Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery],
    ) -> Union[Attributes, Dict[NodeIndex, Attributes]]:
        """Removes nodes from the MedRecord and returns their attributes.

        If a single node index is provided, returns the attributes of the removed node.
        If multiple node indices are specified, returns a dictionary mapping each node
        index to its attributes.

        Args:
            nodes (Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]):
                One or more node indices or a node query.

        Returns:
            Union[Attributes, Dict[NodeIndex, Attributes]]: Attributes of the
                removed node(s).
        """  # noqa: W505
        if isinstance(nodes, Callable):
            query_result = self.query_nodes(nodes)

            if isinstance(query_result, list):
                return self._medrecord.remove_nodes(query_result)
            if query_result is not None:
                return self._medrecord.remove_nodes([query_result])[query_result]

            return {}

        attributes = self._medrecord.remove_nodes(
            nodes if isinstance(nodes, list) else [nodes]
        )

        if isinstance(nodes, list):
            return attributes

        return attributes[nodes]

    def add_nodes(
        self,
        nodes: NodeInput,
        group: Optional[Group] = None,
    ) -> None:
        """Adds nodes to the MedRecord from different data formats.

        Accepts a node tuple (single node added), a list of tuples, DataFrame(s), or
        PolarsNodeDataFrameInput(s) to add nodes. If a DataFrame or list of DataFrames
        is used, the add_nodes_pandas method is called. If PolarsNodeDataFrameInput(s)
        are provided, each tuple must include a DataFrame and the index column. If a
        group is specified, the nodes are added to the group.

        Args:
            nodes (NodeInput): Data representing nodes in various formats.
            group (Optional[Group]): The name of the group to add the nodes to. If not
                specified, the nodes are added to the MedRecord without a group.
        """
        if is_pandas_node_dataframe_input(nodes) or is_pandas_node_dataframe_input_list(
            nodes
        ):
            self.add_nodes_pandas(nodes, group)
        elif is_polars_node_dataframe_input(
            nodes
        ) or is_polars_node_dataframe_input_list(nodes):
            self.add_nodes_polars(nodes, group)
        else:
            if is_node_tuple(nodes):
                nodes = [nodes]

            self._medrecord.add_nodes(nodes)

            if group is None:
                return

            if not self.contains_group(group):
                self.add_group(group)

            self.add_nodes_to_group(group, [node[0] for node in nodes])

    def add_nodes_pandas(
        self,
        nodes: Union[PandasNodeDataFrameInput, List[PandasNodeDataFrameInput]],
        group: Optional[Group] = None,
    ) -> None:
        """Adds nodes to the MedRecord instance from one or more Pandas DataFrames.

        This method accepts either a single tuple or a list of tuples, where each tuple
        consists of a Pandas DataFrame and an index column string. If a group is
        specified, the nodes are added to the group.

        Args:
            nodes (Union[PandasNodeDataFrameInput, List[PandasNodeDataFrameInput]]):
                A tuple or list of tuples, each with a DataFrame and index column.
            group (Optional[Group]): The name of the group to add the nodes to. If not
                specified, the nodes are added to the MedRecord without a group.
        """
        self.add_nodes_polars(
            [process_nodes_dataframe(nodes_df) for nodes_df in nodes]
            if isinstance(nodes, list)
            else [process_nodes_dataframe(nodes)],
            group,
        )

    def add_nodes_polars(
        self,
        nodes: Union[PolarsNodeDataFrameInput, List[PolarsNodeDataFrameInput]],
        group: Optional[Group] = None,
    ) -> None:
        """Adds nodes to the MedRecord instance from one or more Polars DataFrames.

        This method accepts either a single tuple or a list of tuples, where each tuple
        consists of a Polars DataFrame and an index column string. If a group is
        specified, the nodes are added to the group.

        Args:
            nodes (Union[PolarsNodeDataFrameInput, List[PolarsNodeDataFrameInput]]):
                A tuple or list of tuples, each with a DataFrame and index column.
            group (Optional[Group]): The name of the group to add the nodes to. If not
                specified, the nodes are added to the MedRecord without a group.
        """
        self._medrecord.add_nodes_dataframes(
            nodes if isinstance(nodes, list) else [nodes]
        )

        if group is None:
            return

        if not self.contains_group(group):
            self.add_group(group)

        if isinstance(nodes, list):
            node_indices = [
                nodes for node in nodes for nodes in node[0][node[1]].to_list()
            ]
        else:
            node_indices = nodes[0][nodes[1]].to_list()

        self.add_nodes_to_group(group, node_indices)

    @overload
    def remove_edges(self, edges: Union[EdgeIndex, EdgeIndexQuery]) -> Attributes: ...

    @overload
    def remove_edges(
        self, edges: Union[EdgeIndexInputList, EdgeIndicesQuery]
    ) -> Dict[EdgeIndex, Attributes]: ...

    def remove_edges(
        self,
        edges: Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery],
    ) -> Union[Attributes, Dict[EdgeIndex, Attributes]]:
        """Removes edges from the MedRecord and returns their attributes.

        If a single edge index is provided, returns the attributes of the removed edge.
        If multiple edge indices are specified, returns a dictionary mapping each edge
        index to its attributes.

        Args:
            edges (Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery]):
                One or more edge indices or an edge query.

        Returns:
            Union[Attributes, Dict[EdgeIndex, Attributes]]: Attributes of the
                removed edge(s).
        """  # noqa: W505
        if isinstance(edges, Callable):
            query_result = self.query_edges(edges)

            if isinstance(query_result, list):
                return self._medrecord.remove_edges(query_result)
            if query_result is not None:
                return self._medrecord.remove_edges([query_result])[query_result]

            return {}

        attributes = self._medrecord.remove_edges(
            edges if isinstance(edges, list) else [edges]
        )

        if isinstance(edges, list):
            return attributes

        return attributes[edges]

    def add_edges(
        self,
        edges: EdgeInput,
        group: Optional[Group] = None,
    ) -> List[EdgeIndex]:
        """Adds edges to the MedRecord instance from various data formats.

        Accepts edge tuple, lists of tuples, DataFrame(s), or EdgeDataFrameInput(s) to
        add edges. Each tuple must have indices for source and target nodes and a
        dictionary of attributes. If a DataFrame or list of DataFrames is used, the
        add_edges_dataframe method is invoked. If PolarsEdgeDataFrameInput(s) are
        provided, each tuple must include a DataFrame and index columns for source and
        target nodes. If a group is specified, the edges are added to the group.

        Args:
            edges (EdgeInput): Data representing edges in several formats.
            group (Optional[Group]): The name of the group to add the edges to. If not
                specified, the edges are added to the MedRecord without a group.

        Returns:
            List[EdgeIndex]: A list of edge indices that were added.
        """
        if is_pandas_edge_dataframe_input(edges) or is_pandas_edge_dataframe_input_list(
            edges
        ):
            return self.add_edges_pandas(edges, group)
        if is_polars_edge_dataframe_input(edges) or is_polars_edge_dataframe_input_list(
            edges
        ):
            return self.add_edges_polars(edges, group)
        if is_edge_tuple(edges):
            edges = [edges]

        edge_indices = self._medrecord.add_edges(edges)

        if group is None:
            return edge_indices

        if not self.contains_group(group):
            self.add_group(group)

        self.add_edges_to_group(group, edge_indices)

        return edge_indices

    def add_edges_pandas(
        self,
        edges: Union[PandasEdgeDataFrameInput, List[PandasEdgeDataFrameInput]],
        group: Optional[Group] = None,
    ) -> List[EdgeIndex]:
        """Adds edges to the MedRecord from one or more Pandas DataFrames.

        This method accepts either a single PandasEdgeDataFrameInput tuple or a list of
        such tuples, each including a DataFrame and index columns for the source and
        target nodes. If a group is specified, the edges are added to the group.

        Args:
            edges (Union[PandasEdgeDataFrameInput, List[PandasEdgeDataFrameInput]]):
                A tuple or list of tuples, each including a DataFrame and index columns
                for source and target nodes.
            group (Optional[Group]): The name of the group to add the edges to. If not
                specified, the edges are added to the MedRecord without a group.

        Returns:
            List[EdgeIndex]: A list of the edge indices added.
        """
        return self.add_edges_polars(
            [process_edges_dataframe(edges_df) for edges_df in edges]
            if isinstance(edges, list)
            else [process_edges_dataframe(edges)],
            group,
        )

    def add_edges_polars(
        self,
        edges: Union[PolarsEdgeDataFrameInput, List[PolarsEdgeDataFrameInput]],
        group: Optional[Group] = None,
    ) -> List[EdgeIndex]:
        """Adds edges to the MedRecord from one or more Polars DataFrames.

        This method accepts either a single PolarsEdgeDataFrameInput tuple or a list of
        such tuples, each including a DataFrame and index columns for the source and
        target nodes. If a group is specified, the edges are added to the group.

        Args:
            edges (Union[PolarsEdgeDataFrameInput, List[PolarsEdgeDataFrameInput]]):
                A tuple or list of tuples, each including a DataFrame and index columns
                for source and target nodes.
            group (Optional[Group]): The name of the group to add the edges to. If not
                specified, the edges are added to the MedRecord without a group.

        Returns:
            List[EdgeIndex]: A list of the edge indices added.
        """
        edge_indices = self._medrecord.add_edges_dataframes(
            edges if isinstance(edges, list) else [edges]
        )

        if group is None:
            return edge_indices

        if not self.contains_group(group):
            self.add_group(group)

        self.add_edges_to_group(group, edge_indices)

        return edge_indices

    def add_group(
        self,
        group: Group,
        nodes: Optional[
            Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]
        ] = None,
        edges: Optional[
            Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery]
        ] = None,
    ) -> None:
        """Adds a group to the MedRecord, optionally with node and edge indices.

        If node indices are specified, they are added to the group. If no nodes are
        specified, the group is created without any nodes.

        Args:
            group (Group): The name of the group to add.
            nodes (Optional[Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]]):
                One or more node indices or a node query to add
                to the group, optional.
            edges (Optional[Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery]]):
                One or more edge indices or an edge query to add
                to the group, optional.

        Returns:
            None
        """  # noqa: W505
        if isinstance(nodes, Callable):
            nodes = self.query_nodes(nodes)

        if isinstance(edges, Callable):
            edges = self.query_edges(edges)

        if nodes is not None and edges is not None:
            return self._medrecord.add_group(
                group,
                nodes if isinstance(nodes, list) else [nodes],
                edges if isinstance(edges, list) else [edges],
            )
        if nodes is not None:
            return self._medrecord.add_group(
                group, nodes if isinstance(nodes, list) else [nodes], None
            )
        if edges is not None:
            return self._medrecord.add_group(
                group, None, edges if isinstance(edges, list) else [edges]
            )
        return self._medrecord.add_group(group, None, None)

    def remove_groups(self, groups: Union[Group, GroupInputList]) -> None:
        """Removes one or more groups from the MedRecord instance.

        Args:
            groups (Union[Group, GroupInputList]): One or more group names to remove.

        Returns:
            None
        """
        return self._medrecord.remove_groups(
            groups if isinstance(groups, list) else [groups]
        )

    def add_nodes_to_group(
        self,
        group: Group,
        nodes: Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery],
    ) -> None:
        """Adds one or more nodes to a specified group in the MedRecord.

        Args:
            group (Group): The name of the group to add nodes to.
            nodes (Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]):
                One or more node indices or a node query to add to the group.

        Returns:
            None
        """  # noqa: W505
        if isinstance(nodes, Callable):
            query_result = self.query_nodes(nodes)

            if isinstance(query_result, list):
                return self._medrecord.add_nodes_to_group(group, query_result)
            if query_result is not None:
                return self._medrecord.add_nodes_to_group(group, [query_result])

            return None

        return self._medrecord.add_nodes_to_group(
            group, nodes if isinstance(nodes, list) else [nodes]
        )

    def add_edges_to_group(
        self,
        group: Group,
        edges: Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery],
    ) -> None:
        """Adds one or more edges to a specified group in the MedRecord.

        Args:
            group (Group): The name of the group to add edges to.
            edges (Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery]):
                One or more edge indices or an edge query to add to the group.

        Returns:
            None
        """  # noqa: W505
        if isinstance(edges, Callable):
            query_result = self.query_edges(edges)

            if isinstance(query_result, list):
                return self._medrecord.add_edges_to_group(group, query_result)
            if query_result is not None:
                return self._medrecord.add_edges_to_group(group, [query_result])

            return None

        return self._medrecord.add_edges_to_group(
            group, edges if isinstance(edges, list) else [edges]
        )

    def remove_nodes_from_group(
        self,
        group: Group,
        nodes: Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery],
    ) -> None:
        """Removes one or more nodes from a specified group in the MedRecord.

        Args:
            group (Group): The name of the group from which to remove nodes.
            nodes (Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]):
                One or more node indices or a node query to remove from the group.

        Returns:
            None
        """  # noqa: W505
        if isinstance(nodes, Callable):
            query_result = self.query_nodes(nodes)

            if isinstance(query_result, list):
                return self._medrecord.remove_nodes_from_group(group, query_result)
            if query_result is not None:
                return self._medrecord.remove_nodes_from_group(group, [query_result])

            return None

        return self._medrecord.remove_nodes_from_group(
            group, nodes if isinstance(nodes, list) else [nodes]
        )

    def remove_edges_from_group(
        self,
        group: Group,
        edges: Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery],
    ) -> None:
        """Removes one or more edges from a specified group in the MedRecord.

        Args:
            group (Group): The name of the group from which to remove edges.
            edges (Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery]):
                One or more edge indices or an edge query to remove from the group.

        Returns:
            None
        """  # noqa: W505
        if isinstance(edges, Callable):
            query_result = self.query_edges(edges)

            if isinstance(query_result, list):
                return self._medrecord.remove_edges_from_group(group, query_result)
            if query_result is not None:
                return self._medrecord.remove_edges_from_group(group, [query_result])

            return None

        return self._medrecord.remove_edges_from_group(
            group, edges if isinstance(edges, list) else [edges]
        )

    @overload
    def nodes_in_group(self, group: Group) -> List[NodeIndex]: ...

    @overload
    def nodes_in_group(self, group: GroupInputList) -> Dict[Group, List[NodeIndex]]: ...

    def nodes_in_group(
        self, group: Union[Group, GroupInputList]
    ) -> Union[List[NodeIndex], Dict[Group, List[NodeIndex]]]:
        """Retrieves the node indices associated with the specified group/s.

        If a single group is specified, returns a list of node indices for that group.
        If multiple groups are specified, returns a dictionary mapping each group name
        to its list of node indices.

        Args:
            group (GroupInputList): One or more group names.

        Returns:
            Union[List[NodeIndex], Dict[Group, List[NodeIndex]]]: Node indices
                associated with the specified group(s).
        """
        nodes = self._medrecord.nodes_in_group(
            group if isinstance(group, list) else [group]
        )

        if isinstance(group, list):
            return nodes

        return nodes[group]

    @overload
    def edges_in_group(self, group: Group) -> List[EdgeIndex]: ...

    @overload
    def edges_in_group(self, group: GroupInputList) -> Dict[Group, List[EdgeIndex]]: ...

    def edges_in_group(
        self, group: Union[Group, GroupInputList]
    ) -> Union[List[EdgeIndex], Dict[Group, List[EdgeIndex]]]:
        """Retrieves the edge indices associated with the specified group(s).

        If a single group is specified, returns a list of edge indices for that group.
        If multiple groups are specified, returns a dictionary mapping each group name
        to its list of edge indices.

        Args:
            group (GroupInputList): One or more group names.

        Returns:
            Union[List[EdgeIndex], Dict[Group, List[EdgeIndex]]]: Edge indices
                associated with the specified group(s).
        """
        edges = self._medrecord.edges_in_group(
            group if isinstance(group, list) else [group]
        )

        if isinstance(group, list):
            return edges

        return edges[group]

    @overload
    def groups_of_node(self, node: Union[NodeIndex, NodeIndexQuery]) -> List[Group]: ...

    @overload
    def groups_of_node(
        self, node: Union[NodeIndexInputList, NodeIndicesQuery]
    ) -> Dict[NodeIndex, List[Group]]: ...

    def groups_of_node(
        self,
        node: Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery],
    ) -> Union[List[Group], Dict[NodeIndex, List[Group]]]:
        """Retrieves the groups associated with the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of groups for that node.
        If multiple nodes are specified, returns a dictionary mapping each node index to
        its list of groups.

        Args:
            node (Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]):
                One or more node indices or a node query.

        Returns:
            Union[List[Group], Dict[NodeIndex, List[Group]]]: Groups associated with
                each node.
        """  # noqa: W505
        if isinstance(node, Callable):
            query_result = self.query_nodes(node)

            if isinstance(query_result, list):
                return self._medrecord.groups_of_node(query_result)
            if query_result is not None:
                return self._medrecord.groups_of_node([query_result])[query_result]

            return []

        groups = self._medrecord.groups_of_node(
            node if isinstance(node, list) else [node]
        )

        if isinstance(node, list):
            return groups

        return groups[node]

    @overload
    def groups_of_edge(self, edge: Union[EdgeIndex, EdgeIndexQuery]) -> List[Group]: ...

    @overload
    def groups_of_edge(
        self, edge: Union[EdgeIndexInputList, EdgeIndicesQuery]
    ) -> Dict[EdgeIndex, List[Group]]: ...

    def groups_of_edge(
        self,
        edge: Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery],
    ) -> Union[List[Group], Dict[EdgeIndex, List[Group]]]:
        """Retrieves the groups associated with the specified edge(s) in the MedRecord.

        If a single edge index is provided, returns a list of groups for that edge.
        If multiple edges are specified, returns a dictionary mapping each edge index to
        its list of groups.

        Args:
            edge (Union[EdgeIndex, EdgeIndexInputList, EdgeIndexQuery, EdgeIndicesQuery]):
                One or more edge indices or an edge query.

        Returns:
            Union[List[Group], Dict[EdgeIndex, List[Group]]]: Groups associated with
                each edge.
        """  # noqa: W505
        if isinstance(edge, Callable):
            query_result = self.query_edges(edge)

            if isinstance(query_result, list):
                return self._medrecord.groups_of_edge(query_result)
            if query_result is not None:
                return self._medrecord.groups_of_edge([query_result])[query_result]

            return []

        groups = self._medrecord.groups_of_edge(
            edge if isinstance(edge, list) else [edge]
        )

        if isinstance(edge, list):
            return groups

        return groups[edge]

    def node_count(self) -> int:
        """Returns the total number of nodes currently managed by the MedRecord.

        Returns:
            int: The total number of nodes.
        """
        return self._medrecord.node_count()

    def edge_count(self) -> int:
        """Returns the total number of edges currently managed by the MedRecord.

        Returns:
            int: The total number of edges.
        """
        return self._medrecord.edge_count()

    def group_count(self) -> int:
        """Returns the total number of groups currently defined within the MedRecord.

        Returns:
            int: The total number of groups.
        """
        return self._medrecord.group_count()

    def contains_node(self, node: NodeIndex) -> bool:
        """Checks whether a specific node exists in the MedRecord.

        Args:
            node (NodeIndex): The index of the node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        return self._medrecord.contains_node(node)

    def contains_edge(self, edge: EdgeIndex) -> bool:
        """Checks whether a specific edge exists in the MedRecord.

        Args:
            edge (EdgeIndex): The index of the edge to check.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        return self._medrecord.contains_edge(edge)

    def contains_group(self, group: Group) -> bool:
        """Checks whether a specific group exists in the MedRecord.

        Args:
            group (Group): The name of the group to check.

        Returns:
            bool: True if the group exists, False otherwise.
        """
        return self._medrecord.contains_group(group)

    @overload
    def neighbors(
        self,
        node: Union[NodeIndex, NodeIndexQuery],
        directed: EdgesDirected = EdgesDirected.DIRECTED,
    ) -> List[NodeIndex]: ...

    @overload
    def neighbors(
        self,
        node: Union[NodeIndexInputList, NodeIndicesQuery],
        directed: EdgesDirected = EdgesDirected.DIRECTED,
    ) -> Dict[NodeIndex, List[NodeIndex]]: ...

    def neighbors(
        self,
        node: Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery],
        directed: EdgesDirected = EdgesDirected.DIRECTED,
    ) -> Union[List[NodeIndex], Dict[NodeIndex, List[NodeIndex]]]:
        """Retrieves the neighbors of the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of its neighboring
        node indices. If multiple nodes are specified, returns a dictionary mapping
        each node index to its list of neighboring nodes.

        Args:
            node (Union[NodeIndex, NodeIndexInputList, NodeIndexQuery, NodeIndicesQuery]):
                One or more node indices or a query that returns node indices.
            directed (EdgesDirected, optional): Whether to consider edges as directed.
                Defaults to EdgesDirected.DIRECTED.

        Returns:
            Union[List[NodeIndex], Dict[NodeIndex, List[NodeIndex]]]: Neighboring nodes.
        """  # noqa: W505
        if isinstance(node, Callable):
            query_result = self.query_nodes(node)

            if query_result is None:
                return []

            node = query_result

        if directed == EdgesDirected.DIRECTED:
            neighbors = self._medrecord.neighbors(
                node if isinstance(node, list) else [node]
            )
        else:
            neighbors = self._medrecord.neighbors_undirected(
                node if isinstance(node, list) else [node]
            )

        if isinstance(node, list):
            return neighbors

        return neighbors[node]

    def clear(self) -> None:
        """Clears all data from the MedRecord instance.

        Removes all nodes, edges, and groups, effectively resetting the instance.

        Returns:
            None
        """
        return self._medrecord.clear()

    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeAttributesTreeOperand]
    ) -> NodeAttributesTreeQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeAttributesTreeGroupOperand]
    ) -> NodeAttributesTreeGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeAttributesTreeOperand]
    ) -> EdgeAttributesTreeQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeAttributesTreeGroupOperand]
    ) -> EdgeAttributesTreeGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeMultipleAttributesWithIndexOperand]
    ) -> NodeMultipleAttributesWithIndexQueryResult: ...
    @overload
    def query_nodes(
        self,
        query: Callable[[NodeOperand], NodeMultipleAttributesWithIndexGroupOperand],
    ) -> NodeMultipleAttributesWithIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeMultipleAttributesWithoutIndexOperand]
    ) -> NodeMultipleAttributesWithoutIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeMultipleAttributesWithIndexOperand]
    ) -> EdgeMultipleAttributesWithIndexQueryResult: ...
    @overload
    def query_nodes(
        self,
        query: Callable[[NodeOperand], EdgeMultipleAttributesWithIndexGroupOperand],
    ) -> EdgeMultipleAttributesWithIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeMultipleAttributesWithoutIndexOperand]
    ) -> EdgeMultipleAttributesWithoutIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeSingleAttributeWithIndexOperand]
    ) -> NodeSingleAttributeWithIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeSingleAttributeWithIndexGroupOperand]
    ) -> NodeSingleAttributeWithIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeSingleAttributeWithoutIndexOperand]
    ) -> NodeSingleAttributeWithoutIndexQueryResult: ...
    @overload
    def query_nodes(
        self,
        query: Callable[[NodeOperand], NodeSingleAttributeWithoutIndexGroupOperand],
    ) -> NodeSingleAttributeWithoutIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeSingleAttributeWithIndexOperand]
    ) -> EdgeSingleAttributeWithIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeSingleAttributeWithIndexGroupOperand]
    ) -> EdgeSingleAttributeWithIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeSingleAttributeWithoutIndexOperand]
    ) -> EdgeSingleAttributeWithoutIndexQueryResult: ...
    @overload
    def query_nodes(
        self,
        query: Callable[[NodeOperand], EdgeSingleAttributeWithoutIndexGroupOperand],
    ) -> EdgeSingleAttributeWithoutIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeIndicesOperand]
    ) -> EdgeIndicesQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeIndicesGroupOperand]
    ) -> EdgeIndicesGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeIndexOperand]
    ) -> EdgeIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeIndexGroupOperand]
    ) -> EdgeIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeIndicesOperand]
    ) -> NodeIndicesQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeIndicesGroupOperand]
    ) -> NodeIndicesGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeIndexOperand]
    ) -> NodeIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeIndexGroupOperand]
    ) -> NodeIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeMultipleValuesWithIndexOperand]
    ) -> NodeMultipleValuesWithIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeMultipleValuesWithIndexGroupOperand]
    ) -> NodeMultipleValuesWithIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeMultipleValuesWithoutIndexOperand]
    ) -> NodeMultipleValuesWithoutIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeMultipleValuesWithIndexOperand]
    ) -> EdgeMultipleValuesWithIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeMultipleValuesWithIndexGroupOperand]
    ) -> EdgeMultipleValuesWithIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeMultipleValuesWithoutIndexOperand]
    ) -> EdgeMultipleValuesWithoutIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeSingleValueWithIndexOperand]
    ) -> NodeSingleValueWithIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeSingleValueWithIndexGroupOperand]
    ) -> NodeSingleValueWithIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeSingleValueWithoutIndexOperand]
    ) -> NodeSingleValueWithoutIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], NodeSingleValueWithoutIndexGroupOperand]
    ) -> NodeSingleValueWithoutIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeSingleValueWithIndexOperand]
    ) -> EdgeSingleValueWithIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeSingleValueWithIndexGroupOperand]
    ) -> EdgeSingleValueWithIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeSingleValueWithoutIndexOperand]
    ) -> EdgeSingleValueWithoutIndexQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], EdgeSingleValueWithoutIndexGroupOperand]
    ) -> EdgeSingleValueWithoutIndexGroupQueryResult: ...
    @overload
    def query_nodes(
        self, query: Callable[[NodeOperand], Sequence[QueryReturnOperand]]
    ) -> List[QueryResult]: ...

    def query_nodes(self, query: NodeQuery) -> QueryResult:
        """Retrieves information on the nodes from the MedRecord given the query.

        Args:
            query (NodeQuery): A query to define the information to be retrieved.
                The query should be a callable that takes a NodeOperand and returns
                a QueryReturnOperand.

        Returns:
            QueryResult: The result of the query, which can be a list of node indices
                or a dictionary of node attributes, among others.
        """

        def _query(node: PyNodeOperand) -> PyQueryReturnOperand:
            result = query(NodeOperand._from_py_node_operand(node))

            return _convert_queryreturnoperand_to_pyqueryreturnoperand(result)

        return self._medrecord.query_nodes(_query)

    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeAttributesTreeOperand]
    ) -> NodeAttributesTreeQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeAttributesTreeGroupOperand]
    ) -> NodeAttributesTreeGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeAttributesTreeOperand]
    ) -> EdgeAttributesTreeQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeAttributesTreeGroupOperand]
    ) -> EdgeAttributesTreeGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeMultipleAttributesWithIndexOperand]
    ) -> NodeMultipleAttributesWithIndexQueryResult: ...
    @overload
    def query_edges(
        self,
        query: Callable[[EdgeOperand], NodeMultipleAttributesWithIndexGroupOperand],
    ) -> NodeMultipleAttributesWithIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeMultipleAttributesWithoutIndexOperand]
    ) -> NodeMultipleAttributesWithoutIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeMultipleAttributesWithIndexOperand]
    ) -> EdgeMultipleAttributesWithIndexQueryResult: ...
    @overload
    def query_edges(
        self,
        query: Callable[[EdgeOperand], EdgeMultipleAttributesWithIndexGroupOperand],
    ) -> EdgeMultipleAttributesWithIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeMultipleAttributesWithoutIndexOperand]
    ) -> EdgeMultipleAttributesWithoutIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeSingleAttributeWithIndexOperand]
    ) -> NodeSingleAttributeWithIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeSingleAttributeWithIndexGroupOperand]
    ) -> NodeSingleAttributeWithIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeSingleAttributeWithoutIndexOperand]
    ) -> NodeSingleAttributeWithoutIndexQueryResult: ...
    @overload
    def query_edges(
        self,
        query: Callable[[EdgeOperand], NodeSingleAttributeWithoutIndexGroupOperand],
    ) -> NodeSingleAttributeWithoutIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeSingleAttributeWithIndexOperand]
    ) -> EdgeSingleAttributeWithIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeSingleAttributeWithIndexGroupOperand]
    ) -> EdgeSingleAttributeWithIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeSingleAttributeWithoutIndexOperand]
    ) -> EdgeSingleAttributeWithoutIndexQueryResult: ...
    @overload
    def query_edges(
        self,
        query: Callable[[EdgeOperand], EdgeSingleAttributeWithoutIndexGroupOperand],
    ) -> EdgeSingleAttributeWithoutIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeIndicesOperand]
    ) -> EdgeIndicesQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeIndicesGroupOperand]
    ) -> EdgeIndicesGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeIndexOperand]
    ) -> EdgeIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeIndexGroupOperand]
    ) -> EdgeIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeIndicesOperand]
    ) -> NodeIndicesQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeIndicesGroupOperand]
    ) -> NodeIndicesGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeIndexOperand]
    ) -> NodeIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeIndexGroupOperand]
    ) -> NodeIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeMultipleValuesWithIndexOperand]
    ) -> NodeMultipleValuesWithIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeMultipleValuesWithIndexGroupOperand]
    ) -> NodeMultipleValuesWithIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeMultipleValuesWithoutIndexOperand]
    ) -> NodeMultipleValuesWithoutIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeMultipleValuesWithIndexOperand]
    ) -> EdgeMultipleValuesWithIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeMultipleValuesWithIndexGroupOperand]
    ) -> EdgeMultipleValuesWithIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeMultipleValuesWithoutIndexOperand]
    ) -> EdgeMultipleValuesWithoutIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeSingleValueWithIndexOperand]
    ) -> NodeSingleValueWithIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeSingleValueWithIndexGroupOperand]
    ) -> NodeSingleValueWithIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeSingleValueWithoutIndexOperand]
    ) -> NodeSingleValueWithoutIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], NodeSingleValueWithoutIndexGroupOperand]
    ) -> NodeSingleValueWithoutIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeSingleValueWithIndexOperand]
    ) -> EdgeSingleValueWithIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeSingleValueWithIndexGroupOperand]
    ) -> EdgeSingleValueWithIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeSingleValueWithoutIndexOperand]
    ) -> EdgeSingleValueWithoutIndexQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], EdgeSingleValueWithoutIndexGroupOperand]
    ) -> EdgeSingleValueWithoutIndexGroupQueryResult: ...
    @overload
    def query_edges(
        self, query: Callable[[EdgeOperand], Sequence[QueryReturnOperand]]
    ) -> List[QueryResult]: ...

    def query_edges(self, query: EdgeQuery) -> QueryResult:
        """Retrieves information on the edges from the MedRecord given the query.

        Args:
            query (EdgeQuery): A query to define the information to be retrieved.
                The query should be a callable that takes an EdgeOperand and returns
                a QueryReturnOperand.

        Returns:
            QueryResult: The result of the query, which can be a list of edge indices or
                a dictionary of edge attributes, among others.
        """

        def _query(edge: PyEdgeOperand) -> PyQueryReturnOperand:
            result = query(EdgeOperand._from_py_edge_operand(edge))

            return _convert_queryreturnoperand_to_pyqueryreturnoperand(result)

        return self._medrecord.query_edges(_query)

    def clone(self) -> MedRecord:
        """Clones the MedRecord instance.

        Returns:
            MedRecord: A clone of the MedRecord instance.
        """
        medrecord = MedRecord.__new__(MedRecord)
        medrecord._medrecord = self._medrecord.clone()

        return medrecord

    def _describe_group_nodes(
        self, groups: Optional[GroupInputList] = None
    ) -> Dict[Group, AttributeInfo]:
        """Creates a summary of group nodes and their attributes.

        Returns:
            pl.DataFrame: Dataframe with all nodes in medrecord groups and their
                attributes.
        """
        nodes_info = {}
        grouped_nodes = []

        if not groups:
            groups = sorted(self.groups, key=lambda x: str(x))
            add_ungrouped = True
        else:
            add_ungrouped = False

        for group in groups:
            nodes = self.group(group)["nodes"]
            grouped_nodes.extend(nodes)

            if (len(nodes) == 0) and (self.group(group)["edges"]):
                continue

            schema = self.get_schema()

            attributes_schema = (
                schema.group(group).nodes if group in schema.groups else None
            )

            nodes_info[group] = {
                "count": len(nodes),
                "attribute": extract_attribute_summary(
                    self.node[nodes], schema=attributes_schema
                ),
            }

        if not add_ungrouped:
            return nodes_info

        ungrouped_count = self.node_count() - len(set(grouped_nodes))

        if ungrouped_count > 0:
            nodes_info["Ungrouped Nodes"] = {
                "count": ungrouped_count,
                "attribute": {},
            }

        return nodes_info

    def _describe_group_edges(
        self, groups: Optional[GroupInputList] = None
    ) -> Dict[Group, AttributeInfo]:
        """Creates a summary of group edges and their attributes.

        Args:
            groups (Optional[GroupInputList], optional): List of groups that should be
                considered. If no groups are given, all groups containing edges will be
                summarized. Defaults to None.

        Returns:
            pl.DataFrame: DataFrame with an overview of edges connecting group nodes.
        """
        edges_info = {}
        grouped_edges = []

        if not groups:
            groups = sorted(self.groups, key=lambda x: str(x))
            add_ungrouped = True
        else:
            add_ungrouped = False

        for group in groups:
            edges = self.group(group)["edges"]
            grouped_edges.extend(edges)

            if not edges:
                continue

            schema = self.get_schema()

            attributes_schema = (
                schema.group(group).edges if group in schema.groups else None
            )

            edges_info[group] = {
                "count": len(edges),
                "attribute": extract_attribute_summary(
                    self.edge[edges], schema=attributes_schema
                ),
            }

        if not add_ungrouped:
            return edges_info

        ungrouped_count = self.edge_count() - len(set(grouped_edges))

        if ungrouped_count > 0:
            edges_info["Ungrouped Edges"] = {
                "count": ungrouped_count,
                "attribute": {},
            }

        return edges_info

    def __repr__(self) -> str:
        """Returns a string representation of the MedRecord instance."""
        # If in debugging mode, avoid computing the whole representation
        if sys.gettrace() is not None:
            return f"<MedRecord: {self.node_count()} nodes, {self.edge_count()} edges>"

        return "\n".join(
            [str(self.overview_nodes()), "", str(self.overview_edges())]
        )  # pragma: no cover  # coverage tests always have a trace, so this line cannot be reached by them

    def overview_nodes(
        self, groups: Optional[Union[Group, GroupInputList]] = None, decimal: int = 2
    ) -> OverviewTable:
        """Gets a summary for all nodes in groups and their attributes.

        Args:
            groups (Optional[Union[Group, GroupInputList]], optional): Group or list of
                node groups to display. If no groups are given, all groups containing
                nodes are shown. Defaults to None.
            decimal (int, optional): Decimal point to round the float values to.
                Defaults to 2.

        Returns:
            OverviewTable: Display of edge groups and their attributes.

        Example:
            .. code-block:: text

                --------------------------------------------------------------
                Nodes Group     Count Attribute   Type        Data
                --------------------------------------------------------------
                diagnosis       25    description Categorical 25 unique values
                patient         5     age         Continuous  min: 19
                                                              max: 96
                                                              mean: 43.20
                                      gender      Categorical Categories: F, M
                Ungrouped Nodes 10    -           -           -
                --------------------------------------------------------------
        """
        if groups:
            nodes_data = self._describe_group_nodes(
                groups if isinstance(groups, list) else [groups]
            )
        else:
            nodes_data = self._describe_group_nodes()

        return OverviewTable(
            data=nodes_data, group_header="Nodes Group", decimal=decimal
        )

    def overview_edges(
        self, groups: Optional[Union[Group, GroupInputList]] = None, decimal: int = 2
    ) -> OverviewTable:
        """Gets a summary for all edges in groups and their attributes.

        Args:
            groups (Optional[Union[Group, GroupInputList]], optional): Group or list of
                edge groups to display. If no groups are given, all groups containing
                nodes are shown. Defaults to None.
            decimal (int, optional): Decimal point to round the float values to.
                Defaults to 2.

        Returns:
            OverviewTable: Display of edge groups and their attributes.

        Example:
            .. code-block:: text

                --------------------------------------------------------------------------
                Edges Group       Count Attribute      Type       Data
                --------------------------------------------------------------------------
                Patient-Diagnosis 60    diagnosis_time Temporal   min: 1962-10-21 00:00:00
                                                                  max: 2024-04-12 00:00:00
                                        duration_days  Continuous min: 0
                                                                  max: 3416
                                                                  mean: 405.02
                --------------------------------------------------------------------------

        """  # noqa: W505
        if groups:
            edges_data = self._describe_group_edges(
                groups if isinstance(groups, list) else [groups]
            )
        else:
            edges_data = self._describe_group_edges()

        return OverviewTable(
            data=edges_data, group_header="Edges Group", decimal=decimal
        )
