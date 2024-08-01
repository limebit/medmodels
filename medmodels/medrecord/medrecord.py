from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union, overload

import polars as pl

from medmodels._medmodels import PyMedRecord
from medmodels.medrecord.builder import MedRecordBuilder
from medmodels.medrecord.indexers import EdgeIndexer, NodeIndexer
from medmodels.medrecord.querying import EdgeOperation, NodeOperation
from medmodels.medrecord.schema import Schema
from medmodels.medrecord.types import (
    Attributes,
    AttributesInput,
    EdgeIndex,
    EdgeIndexInputList,
    EdgeTuple,
    Group,
    GroupInfo,
    GroupInputList,
    NodeIndex,
    NodeIndexInputList,
    NodeTuple,
    PandasEdgeDataFrameInput,
    PandasNodeDataFrameInput,
    PolarsEdgeDataFrameInput,
    PolarsNodeDataFrameInput,
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


class MedRecord:
    """A class to manage medical records with node and edge data structures.

    Provides methods to create instances from different data formats, manage node and
    edge attributes, and perform operations like adding or removing nodes and edges.
    """

    _medrecord: PyMedRecord

    def __init__(self) -> None:
        """Initializes a new MedRecord instance with an underlying PyMedRecord object."""
        self._medrecord = PyMedRecord()

    @staticmethod
    def builder() -> MedRecordBuilder:
        """Creates a new MedRecordBuilder instance to build a MedRecord.

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
        """
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
        """
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
    def from_example_dataset(cls) -> MedRecord:
        """Creates a MedRecord instance from an example dataset.

        The example dataset was created using the Synthea™ Patient Generator:
        https://github.com/synthetichealth/synthea

        Returns:
            MedRecord: A new instance created from the example dataset.
        """
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_example_dataset()
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

        Returns:
            None
        """
        self._medrecord.to_ron(path)

    @property
    def schema(self) -> Schema:
        """Returns the schema of the MedRecord instance.

        Returns:
            Schema: The schema of the MedRecord.
        """
        return Schema._from_pyschema(self._medrecord.schema)

    @schema.setter
    def schema(self, schema: Schema) -> None:
        """Sets the schema of the MedRecord instance.

        Args:
            schema (Schema): The new schema to apply.

        Returns:
            None
        """
        self._medrecord.update_schema(schema._schema)

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
        """Provides access to node attributes within the MedRecord instance via an indexer.

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
        """Provides access to edge attributes within the MedRecord instance via an indexer.

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
        """Returns the node and edge indices associated with the specified group/s in the MedRecord.

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
    def outgoing_edges(self, node: NodeIndex) -> List[EdgeIndex]: ...

    @overload
    def outgoing_edges(
        self, node: Union[NodeIndexInputList, NodeOperation]
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...

    def outgoing_edges(
        self, node: Union[NodeIndex, NodeIndexInputList, NodeOperation]
    ) -> Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]:
        """Lists the outgoing edges of the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of its outgoing edge indices.
        If multiple nodes are specified, returns a dictionary mapping each node index to
        its list of outgoing edge indices.

        Args:
            node (Union[NodeIndex, NodeIndexInputList, NodeOperation]): One or more
                node indices or a node operation.

        Returns:
            Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]: Outgoing
                edge indices for each specified node.
        """
        if isinstance(node, NodeOperation):
            return self._medrecord.outgoing_edges(self.select_nodes(node))

        indices = self._medrecord.outgoing_edges(
            node if isinstance(node, list) else [node]
        )

        if isinstance(node, list):
            return indices

        return indices[node]

    @overload
    def incoming_edges(self, node: NodeIndex) -> List[EdgeIndex]: ...

    @overload
    def incoming_edges(
        self, node: Union[NodeIndexInputList, NodeOperation]
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...

    def incoming_edges(
        self, node: Union[NodeIndex, NodeIndexInputList, NodeOperation]
    ) -> Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]:
        """Lists the incoming edges of the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of its incoming edge indices.
        If multiple nodes are specified, returns a dictionary mapping each node index to
        its list of incoming edge indices.

        Args:
            node (Union[NodeIndex, NodeIndexInputList, NodeOperation]): One or more
                node indices or a node operation.

        Returns:
            Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]: Incoming
                edge indices for each specified node.
        """
        if isinstance(node, NodeOperation):
            return self._medrecord.incoming_edges(self.select_nodes(node))

        indices = self._medrecord.incoming_edges(
            node if isinstance(node, list) else [node]
        )

        if isinstance(node, list):
            return indices

        return indices[node]

    @overload
    def edge_endpoints(self, edge: EdgeIndex) -> tuple[NodeIndex, NodeIndex]: ...

    @overload
    def edge_endpoints(
        self, edge: Union[EdgeIndexInputList, EdgeOperation]
    ) -> Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]: ...

    def edge_endpoints(
        self, edge: Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]
    ) -> Union[
        tuple[NodeIndex, NodeIndex], Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]
    ]:
        """Retrieves the source and target nodes of the specified edge(s) in the MedRecord.

        If a single edge index is provided, returns a tuple of
        node indices (source, target). If multiple edges are specified, returns
        a dictionary mapping each edge index to its tuple of node indices.

        Args:
            edge (Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]): One or more
                edge indices.

        Returns:
            Union[tuple[NodeIndex, NodeIndex],
                Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]]:
                Tuple of node indices or a dictionary mapping each edge to its
                node indices.
        """
        if isinstance(edge, EdgeOperation):
            return self._medrecord.edge_endpoints(self.select_edges(edge))

        endpoints = self._medrecord.edge_endpoints(
            edge if isinstance(edge, list) else [edge]
        )

        if isinstance(edge, list):
            return endpoints

        return endpoints[edge]

    def edges_connecting(
        self,
        source_node: Union[NodeIndex, NodeIndexInputList, NodeOperation],
        target_node: Union[NodeIndex, NodeIndexInputList, NodeOperation],
        directed: bool = True,
    ) -> List[EdgeIndex]:
        """Retrieves the edges connecting the specified source and target nodes in the MedRecord.

        If a NodeOperation is provided for either the source or target nodes, it is
        first evaluated to obtain the corresponding node indices. The method then
        returns a list of edge indices that connect the specified source and
        target nodes.

        Args:
            source_node (Union[NodeIndex, NodeIndexInputList, NodeOperation]):
                The index or indices of the source node(s), or a NodeOperation to
                select source nodes.
            target_node (Union[NodeIndex, NodeIndexInputList, NodeOperation]):
                The index or indices of the target node(s), or a NodeOperation to
                select target nodes.
            directed (bool, optional): Whether to consider edges as directed.

        Returns:
            List[EdgeIndex]: A list of edge indices connecting the specified source and
                target nodes.

        """
        if isinstance(source_node, NodeOperation):
            source_node = self.select_nodes(source_node)

        if isinstance(target_node, NodeOperation):
            target_node = self.select_nodes(target_node)

        if directed:
            return self._medrecord.edges_connecting(
                (source_node if isinstance(source_node, list) else [source_node]),
                (target_node if isinstance(target_node, list) else [target_node]),
            )
        else:
            return self._medrecord.edges_connecting_undirected(
                (source_node if isinstance(source_node, list) else [source_node]),
                (target_node if isinstance(target_node, list) else [target_node]),
            )

    def add_node(self, node: NodeIndex, attributes: AttributesInput) -> None:
        """Adds a node with specified attributes to the MedRecord instance.

        Args:
            node (NodeIndex): The index of the node to add.
            attributes (Attributes): A dictionary of the node's attributes.

        Returns:
            None
        """
        return self._medrecord.add_node(node, attributes)

    @overload
    def remove_node(self, node: NodeIndex) -> Attributes: ...

    @overload
    def remove_node(
        self, node: Union[NodeIndexInputList, NodeOperation]
    ) -> Dict[NodeIndex, Attributes]: ...

    def remove_node(
        self, node: Union[NodeIndex, NodeIndexInputList, NodeOperation]
    ) -> Union[Attributes, Dict[NodeIndex, Attributes]]:
        """Removes a node or multiple nodes from the MedRecord and returns their attributes.

        If a single node index is provided, returns the attributes of the removed node.
        If multiple node indices are specified, returns a dictionary mapping each node
        index to its attributes.

        Args:
            node (Union[NodeIndex, NodeIndexInputList, NodeOperation]): One or more
                node indices or a node operation.

        Returns:
            Union[Attributes, Dict[NodeIndex, Attributes]]: Attributes of the
                removed node(s).
        """
        if isinstance(node, NodeOperation):
            return self._medrecord.remove_node(self.select_nodes(node))

        attributes = self._medrecord.remove_node(
            node if isinstance(node, list) else [node]
        )

        if isinstance(node, list):
            return attributes

        return attributes[node]

    def add_nodes(
        self,
        nodes: Union[
            Sequence[NodeTuple],
            PandasNodeDataFrameInput,
            List[PandasNodeDataFrameInput],
            PolarsNodeDataFrameInput,
            List[PolarsNodeDataFrameInput],
        ],
    ) -> None:
        """Adds multiple nodes to the MedRecord from different data formats.

        Accepts a list of tuples, DataFrame(s), or PolarsNodeDataFrameInput(s) to add
        nodes. If a DataFrame or list of DataFrames is used, the add_nodes_pandas method
        is called. If PolarsNodeDataFrameInput(s) are provided, each tuple must include
        a DataFrame and the index column.

        Args:
            nodes (Union[Sequence[NodeTuple], PandasNodeDataFrameInput, List[PandasNodeDataFrameInput], PolarsNodeDataFrameInput, List[PolarsNodeDataFrameInput]]):
                Data representing nodes in various formats.

        Returns:
            None
        """
        if is_pandas_node_dataframe_input(nodes) or is_pandas_node_dataframe_input_list(
            nodes
        ):
            return self.add_nodes_pandas(nodes)
        elif is_polars_node_dataframe_input(
            nodes
        ) or is_polars_node_dataframe_input_list(nodes):
            return self.add_nodes_polars(nodes)
        else:
            return self._medrecord.add_nodes(nodes)

    def add_nodes_pandas(
        self, nodes: Union[PandasNodeDataFrameInput, List[PandasNodeDataFrameInput]]
    ) -> None:
        """Adds nodes to the MedRecord instance from one or more Pandas DataFrames.

        This method accepts either a single tuple or a list of tuples, where each tuple
        consists of a Pandas DataFrame and an index column string.

        Args:
            nodes (Union[PandasNodeDataFrameInput, List[PandasNodeDataFrameInput]]):
                A tuple or list of tuples, each with a DataFrame and index column.

        Returns:
            None
        """
        return self.add_nodes_polars(
            [process_nodes_dataframe(nodes_df) for nodes_df in nodes]
            if isinstance(nodes, list)
            else [process_nodes_dataframe(nodes)]
        )

    def add_nodes_polars(
        self, nodes: Union[PolarsNodeDataFrameInput, List[PolarsNodeDataFrameInput]]
    ) -> None:
        """Adds nodes to the MedRecord instance from one or more Polars DataFrames.

        This method accepts either a single tuple or a list of tuples, where each tuple
        consists of a Polars DataFrame and an index column string.

        Args:
            nodes (Union[PolarsNodeDataFrameInput, List[PolarsNodeDataFrameInput]]):
                A tuple or list of tuples, each with a DataFrame and index column.

        Returns:
            None
        """
        return self._medrecord.add_nodes_dataframes(
            nodes if isinstance(nodes, list) else [nodes]
        )

    def add_edge(
        self,
        source_node: NodeIndex,
        target_node: NodeIndex,
        attributes: AttributesInput,
    ) -> EdgeIndex:
        """Adds an edge between two specified nodes with given attributes.

        Args:
            source_node (NodeIndex): Index of the source node.
            target_node (NodeIndex): Index of the target node.
            attributes (AttributesInput): Dictionary or mapping of edge attributes.

        Returns:
            EdgeIndex: The index of the added edge.
        """
        return self._medrecord.add_edge(source_node, target_node, attributes)

    @overload
    def remove_edge(self, edge: EdgeIndex) -> Attributes: ...

    @overload
    def remove_edge(
        self, edge: Union[EdgeIndexInputList, EdgeOperation]
    ) -> Dict[EdgeIndex, Attributes]: ...

    def remove_edge(
        self, edge: Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]
    ) -> Union[Attributes, Dict[EdgeIndex, Attributes]]:
        """Removes an edge or multiple edges from the MedRecord and returns their attributes.

        If a single edge index is provided, returns the attributes of the removed edge.
        If multiple edge indices are specified, returns a dictionary mapping each edge
        index to its attributes.

        Args:
            edge (Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]): One or more
                edge indices or an edge operation.

        Returns:
            Union[Attributes, Dict[EdgeIndex, Attributes]]: Attributes of the
                removed edge(s).
        """
        if isinstance(edge, EdgeOperation):
            return self._medrecord.remove_edge(self.select_edges(edge))

        attributes = self._medrecord.remove_edge(
            edge if isinstance(edge, list) else [edge]
        )

        if isinstance(edge, list):
            return attributes

        return attributes[edge]

    def add_edges(
        self,
        edges: Union[
            Sequence[EdgeTuple],
            PandasEdgeDataFrameInput,
            List[PandasEdgeDataFrameInput],
            PolarsEdgeDataFrameInput,
            List[PolarsEdgeDataFrameInput],
        ],
    ) -> List[EdgeIndex]:
        """Adds edges to the MedRecord instance from various data formats.

        Accepts lists of tuples, DataFrame(s), or EdgeDataFrameInput(s) to add edges.
        Each tuple must have indices for source and target nodes and a dictionary of
        attributes. If a DataFrame or list of DataFrames is used,
        the add_edges_dataframe method is invoked.

        Args:
            edges (Union[Sequence[EdgeTuple], PandasEdgeDataFrameInput, List[PolarsEdgeDataFrameInput]]):
                List[PandasEdgeDataFrameInput], PolarsEdgeDataFrameInput,
                Data representing edges in several formats.

        Returns:
            List[EdgeIndex]: A list of edge indices that were added.
        """
        if is_pandas_edge_dataframe_input(edges) or is_pandas_edge_dataframe_input_list(
            edges
        ):
            return self.add_edges_pandas(edges)
        elif is_polars_edge_dataframe_input(
            edges
        ) or is_polars_edge_dataframe_input_list(edges):
            return self.add_edges_polars(edges)
        else:
            return self._medrecord.add_edges(edges)

    def add_edges_pandas(
        self, edges: Union[PandasEdgeDataFrameInput, List[PandasEdgeDataFrameInput]]
    ) -> List[EdgeIndex]:
        """Adds edges to the MedRecord from one or more Pandas DataFrames.

        This method accepts either a single PandasEdgeDataFrameInput tuple or a list of
        such tuples, each including a DataFrame and index columns for the source and
        target nodes.

        Args:
            edges (Union[PandasEdgeDataFrameInput, List[PandasEdgeDataFrameInput]]):
                A tuple or list of tuples, each including a DataFrame and index columns
                for source and target nodes.

        Returns:
            List[EdgeIndex]: A list of the edge indices added.
        """
        return self.add_edges_polars(
            [process_edges_dataframe(edges_df) for edges_df in edges]
            if isinstance(edges, list)
            else [process_edges_dataframe(edges)]
        )

    def add_edges_polars(
        self,
        edges: Union[PolarsEdgeDataFrameInput, List[PolarsEdgeDataFrameInput]],
    ) -> List[EdgeIndex]:
        """Adds edges to the MedRecord from one or more Polars DataFrames.

        This method accepts either a single PolarsEdgeDataFrameInput tuple or a list of
        such tuples, each including a DataFrame and index columns for the source and
        target nodes.

        Args:
            edges (Union[PolarsEdgeDataFrameInput, List[PolarsEdgeDataFrameInput]]):
                A tuple or list of tuples, each including a DataFrame and index columns
                for source and target nodes.

        Returns:
            List[EdgeIndex]: A list of the edge indices added.
        """
        return self._medrecord.add_edges_dataframes(
            edges if isinstance(edges, list) else [edges]
        )

    def add_group(
        self,
        group: Group,
        nodes: Optional[Union[NodeIndex, NodeIndexInputList, NodeOperation]] = None,
        edges: Optional[Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]] = None,
    ) -> None:
        """Adds a group to the MedRecord instance with an optional list of node indices.

        If node indices are specified, they are added to the group. If no nodes are
        specified, the group is created without any nodes.

        Args:
            group (Group): The name of the group to add.
            nodes (Optional[Union[NodeIndex, NodeIndexInputList, NodeOperation]]):
                One or more node indices or a node operation to add
                to the group, optional.
            edges (Optional[Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]]):
                One or more edge indices or an edge operation to add
                to the group, optional.

        Returns:
            None
        """
        if isinstance(nodes, NodeOperation):
            nodes = self.select_nodes(nodes)

        if isinstance(edges, EdgeOperation):
            edges = self.select_edges(edges)

        if nodes is not None and edges is not None:
            return self._medrecord.add_group(
                group,
                nodes if isinstance(nodes, list) else [nodes],
                edges if isinstance(edges, list) else [edges],
            )
        elif nodes is not None:
            return self._medrecord.add_group(
                group, nodes if isinstance(nodes, list) else [nodes], None
            )
        elif edges is not None:
            return self._medrecord.add_group(
                group, None, edges if isinstance(edges, list) else [edges]
            )
        else:
            return self._medrecord.add_group(group, None, None)

    def remove_group(self, group: Union[Group, GroupInputList]) -> None:
        """Removes one or more groups from the MedRecord instance.

        Args:
            group (Union[Group, GroupInputList]): One or more group names to remove.

        Returns:
            None
        """
        return self._medrecord.remove_group(
            group if isinstance(group, list) else [group]
        )

    def add_node_to_group(
        self, group: Group, node: Union[NodeIndex, NodeIndexInputList, NodeOperation]
    ) -> None:
        """Adds one or more nodes to a specified group in the MedRecord.

        Args:
            group (Group): The name of the group to add nodes to.
            node (Union[NodeIndex, NodeIndexInputList, NodeOperation]): One or more
                node indices or a node operation to add to the group.

        Returns:
            None
        """
        if isinstance(node, NodeOperation):
            return self._medrecord.add_node_to_group(group, self.select_nodes(node))

        return self._medrecord.add_node_to_group(
            group, node if isinstance(node, list) else [node]
        )

    def add_edge_to_group(
        self, group: Group, edge: Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]
    ) -> None:
        """Adds one or more edges to a specified group in the MedRecord.

        Args:
            group (Group): The name of the group to add edges to.
            edge (Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]): One or more
                edge indices or an edge operation to add to the group.

        Returns:
            None
        """
        if isinstance(edge, EdgeOperation):
            return self._medrecord.add_edge_to_group(group, self.select_edges(edge))

        return self._medrecord.add_edge_to_group(
            group, edge if isinstance(edge, list) else [edge]
        )

    def remove_node_from_group(
        self, group: Group, node: Union[NodeIndex, NodeIndexInputList, NodeOperation]
    ) -> None:
        """Removes one or more nodes from a specified group in the MedRecord.

        Args:
            group (Group): The name of the group from which to remove nodes.
            node (Union[NodeIndex, NodeIndexInputList, NodeOperation]): One or more
                node indices or a node operation to remove from the group.

        Returns:
            None
        """
        if isinstance(node, NodeOperation):
            return self._medrecord.remove_node_from_group(
                group, self.select_nodes(node)
            )

        return self._medrecord.remove_node_from_group(
            group, node if isinstance(node, list) else [node]
        )

    def remove_edge_from_group(
        self, group: Group, edge: Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]
    ) -> None:
        """Removes one or more edges from a specified group in the MedRecord.

        Args:
            group (Group): The name of the group from which to remove edges.
            edge (Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]): One or more
                edge indices or an edge operation to remove from the group.

        Returns:
            None
        """
        if isinstance(edge, EdgeOperation):
            return self._medrecord.remove_edge_from_group(
                group, self.select_edges(edge)
            )

        return self._medrecord.remove_edge_from_group(
            group, edge if isinstance(edge, list) else [edge]
        )

    @overload
    def nodes_in_group(self, group: Group) -> List[NodeIndex]: ...

    @overload
    def nodes_in_group(self, group: GroupInputList) -> Dict[Group, List[NodeIndex]]: ...

    def nodes_in_group(
        self, group: Union[Group, GroupInputList]
    ) -> Union[List[NodeIndex], Dict[Group, List[NodeIndex]]]:
        """Retrieves the node indices associated with the specified group/s in the MedRecord.

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
        """Retrieves the edge indices associated with the specified group(s) in the MedRecord.

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
    def groups_of_node(self, node: NodeIndex) -> List[Group]: ...

    @overload
    def groups_of_node(
        self, node: Union[NodeIndexInputList, NodeOperation]
    ) -> Dict[NodeIndex, List[Group]]: ...

    def groups_of_node(
        self, node: Union[NodeIndex, NodeIndexInputList, NodeOperation]
    ) -> Union[List[Group], Dict[NodeIndex, List[Group]]]:
        """Retrieves the groups associated with the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of groups for that node.
        If multiple nodes are specified, returns a dictionary mapping each node index to
        its list of groups.

        Args:
            node (Union[NodeIndex, NodeIndexInputList, NodeOperation]): One or more
                node indices or a node operation.

        Returns:
            Union[List[Group], Dict[NodeIndex, List[Group]]]: Groups associated with
                each node.
        """
        if isinstance(node, NodeOperation):
            return self._medrecord.groups_of_node(self.select_nodes(node))

        groups = self._medrecord.groups_of_node(
            node if isinstance(node, list) else [node]
        )

        if isinstance(node, list):
            return groups

        return groups[node]

    @overload
    def groups_of_edge(self, edge: EdgeIndex) -> List[Group]: ...

    @overload
    def groups_of_edge(
        self, edge: Union[EdgeIndexInputList, EdgeOperation]
    ) -> Dict[EdgeIndex, List[Group]]: ...

    def groups_of_edge(
        self, edge: Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]
    ) -> Union[List[Group], Dict[EdgeIndex, List[Group]]]:
        """Retrieves the groups associated with the specified edge(s) in the MedRecord.

        If a single edge index is provided, returns a list of groups for that edge.
        If multiple edges are specified, returns a dictionary mapping each edge index to
        its list of groups.

        Args:
            edge (Union[EdgeIndex, EdgeIndexInputList, EdgeOperation]): One or more
                edge indices or an edge operation.

        Returns:
            Union[List[Group], Dict[EdgeIndex, List[Group]]]: Groups associated with
                each edge.
        """
        if isinstance(edge, EdgeOperation):
            return self._medrecord.groups_of_edge(self.select_edges(edge))

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
        node: NodeIndex,
        directed: bool = True,
    ) -> List[NodeIndex]: ...

    @overload
    def neighbors(
        self,
        node: Union[NodeIndexInputList, NodeOperation],
        directed: bool = True,
    ) -> Dict[NodeIndex, List[NodeIndex]]: ...

    def neighbors(
        self,
        node: Union[NodeIndex, NodeIndexInputList, NodeOperation],
        directed: bool = True,
    ) -> Union[List[NodeIndex], Dict[NodeIndex, List[NodeIndex]]]:
        """Retrieves the neighbors of the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of its neighboring
        node indices. If multiple nodes are specified, returns a dictionary mapping
        each node index to its list of neighboring nodes.

        Args:
            node (Union[NodeIndex, NodeIndexInputList, NodeOperation]): One or more
                node indices or a node operation.
            directed (bool, optional): Whether to consider edges as directed

        Returns:
            Union[List[NodeIndex], Dict[NodeIndex, List[NodeIndex]]]: Neighboring nodes.
        """
        if isinstance(node, NodeOperation):
            node = self.select_nodes(node)

        if directed:
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

    def select_nodes(self, operation: NodeOperation) -> List[NodeIndex]:
        """Selects nodes based on a specified operation and returns their indices.

        Args:
            operation (NodeOperation): The operation to apply to select nodes.

        Returns:
            List[NodeIndex]: A list of node indices that satisfy the operation.
        """
        return self._medrecord.select_nodes(operation._node_operation)

    def select_edges(self, operation: EdgeOperation) -> List[EdgeIndex]:
        """Selects edges based on a specified operation and returns their indices.

        Args:
            operation (EdgeOperation): The operation to apply to select edges.

        Returns:
            List[EdgeIndex]: A list of edge indices that satisfy the operation.
        """
        return self._medrecord.select_edges(operation._edge_operation)

    @overload
    def __getitem__(self, key: NodeOperation) -> List[NodeIndex]: ...

    @overload
    def __getitem__(self, key: EdgeOperation) -> List[EdgeIndex]: ...

    def __getitem__(
        self, key: Union[NodeOperation, EdgeOperation]
    ) -> Union[List[NodeIndex], List[EdgeIndex]]:
        """Allows selection of nodes or edges using operations directly via indexing.

        Args:
            key (Union[NodeOperation, EdgeOperation]): Operation to select nodes
                or edges.

        Returns:
            Union[List[NodeIndex], List[EdgeIndex]]: Node or edge indices selected.
        """
        if isinstance(key, NodeOperation):
            return self.select_nodes(key)

        return self.select_edges(key)
