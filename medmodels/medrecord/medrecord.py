from __future__ import annotations

from typing import Dict, List, Optional, Union, overload

import pandas as pd
import polars as pl

from medmodels._medmodels import PyMedRecord
from medmodels.medrecord.indexers import EdgeIndexer, NodeIndexer
from medmodels.medrecord.querying import EdgeOperation, NodeOperation
from medmodels.medrecord.types import (
    Attributes,
    EdgeIndex,
    Group,
    NodeIndex,
    PolarsEdgeDataFrameInput,
    PolarsNodeDataFrameInput,
    is_pandas_dataframe_list,
    is_polars_edge_dataframe_input,
    is_polars_edge_dataframe_input_list,
    is_polars_node_dataframe_input,
    is_polars_node_dataframe_input_list,
)


def process_nodes_dataframe(nodes: pd.DataFrame) -> PolarsNodeDataFrameInput:
    """
    Converts a pandas DataFrame of nodes to a Polars DataFrame with an index column.

    Ensures the DataFrame has an Index and returns a tuple containing the converted
    Polars DataFrame and the name of the index column.

    Args:
        nodes (pd.DataFrame): A DataFrame representing nodes, must have an Index.

    Returns:
        PolarsNodeDataFrameInput: A tuple of the Polars DataFrame and index column name.
    """
    assert isinstance(nodes.index, pd.Index), "Nodes dataframe must have an Index"
    assert nodes.index.name is not None, "Nodes dataframe must have an Index"
    nodes_polars = pl.from_pandas(nodes, include_index=True)
    return nodes_polars, nodes.index.name


def process_edges_dataframe(edges: pd.DataFrame) -> PolarsEdgeDataFrameInput:
    """
    Converts a pandas DataFrame of edges to a Polars DataFrame with index columns.

    Ensures the DataFrame has a MultiIndex with exactly 2 levels and returns a tuple
    containing the converted Polars DataFrame and names of the index columns.

    Args:
        edges (pd.DataFrame): A DataFrame representing edges, must have a MultiIndex.

    Returns:
        PolarsEdgeDataFrameInput: A tuple of the Polars DataFrame, source index, and
            target index column names.
    """
    assert isinstance(
        edges.index, pd.MultiIndex
    ), "Edges dataframe must have a MultiIndex"
    assert len(edges.index.names) == 2, "Edges dataframe MultiIndex must have 2 levels"
    edges_polars = pl.from_pandas(edges, include_index=True)
    return edges_polars, edges.index.names[0], edges.index.names[1]


class MedRecord:
    """
    A class to manage medical records with node and edge data structures.

    Provides methods to create instances from different data formats, manage node and
    edge attributes, and perform operations like adding or removing nodes and edges.
    """

    _medrecord: PyMedRecord

    def __init__(self) -> None:
        """
        Initializes a new MedRecord instance with an underlying PyMedRecord object.
        """
        self._medrecord = PyMedRecord()

    @classmethod
    def from_tuples(
        cls,
        nodes: List[tuple[NodeIndex, Attributes]],
        edges: Optional[List[tuple[NodeIndex, NodeIndex, Attributes]]] = None,
    ) -> MedRecord:
        """
        Creates a MedRecord instance from lists of node and edge tuples.

        Nodes and edges are specified as lists of tuples. Each node tuple contains a
        node index and attributes. Each edge tuple includes indices of the source and
        target nodes and edge attributes.

        Args:
            nodes (List[tuple[NodeIndex, Attributes]]): List of node tuples.
            edges (Optional[List[tuple[NodeIndex, NodeIndex, Attributes]]]): List of
                edge tuples, defaults to an empty list.

        Returns:
            MedRecord: A new instance created from the provided tuples.
        """
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_tuples(nodes, edges)
        return medrecord

    @classmethod
    def from_pandas(
        cls,
        nodes: Union[pd.DataFrame, List[pd.DataFrame]],
        edges: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ) -> MedRecord:
        """
        Creates a MedRecord instance from pandas DataFrames of nodes and edges.

        Accepts single DataFrames or lists of DataFrames for nodes and edges. Each node
        DataFrame must have an Index, and each edge DataFrame must have a MultiIndex
        with 2 levels.

        Args:
            nodes (Union[pd.DataFrame, List[pd.DataFrame]]): Nodes DataFrame(s).
            edges (Optional[Union[pd.DataFrame, List[pd.DataFrame]]]):
                Edges DataFrame(s).

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
        """
        Creates a MedRecord from Polars DataFrames of nodes and optionally edges.

        Accepts a tuple or a list of tuples for nodes and edges. Each node tuple
        consists of a Polars DataFrame and an index column. Edge tuples include
        a DataFrame and index columns for source and target nodes.

        Args:
            nodes (Union[PolarsNodeDataFrameInput, List[PolarsNodeDataFrameInput]]):
                Node data.
            edges (Optional[Union[PolarsEdgeDataFrameInput,
                List[PolarsEdgeDataFrameInput]]]):
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
        """
        Creates a MedRecord instance from an example dataset.

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
        """
        Creates a MedRecord instance from a RON file.

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
        """
        Writes the MedRecord instance to a RON file.

        Serializes the MedRecord instance to a RON file at the specified path.

        Args:
            path (str): Path where the RON file will be written.

        Returns:
            None
        """
        self._medrecord.to_ron(path)

    @property
    def nodes(self) -> List[NodeIndex]:
        """
        Lists the node indices in the MedRecord instance.

        Returns a list of all node indices currently managed by the MedRecord instance.

        Returns:
            List[NodeIndex]: A list of node indices.
        """
        return self._medrecord.nodes

    @property
    def node(self) -> NodeIndexer:
        """
        Provides access to node attributes within the MedRecord instance via an indexer.

        Facilitates querying, accessing, manipulating, and setting node attributes using
        various indexing methods. Supports conditions and ranges for more
        complex queries.

        Returns:
            NodeIndexer: An object for manipulating and querying node attributes.
        """
        return NodeIndexer(self)

    @property
    def edges(self) -> List[EdgeIndex]:
        """
        Lists the edge indices in the MedRecord instance.

        Returns a list of all edge indices currently managed by the MedRecord instance.

        Returns:
            List[EdgeIndex]: A list of edge indices.
        """
        return self._medrecord.edges

    @property
    def edge(self) -> EdgeIndexer:
        """
        Provides access to edge attributes within the MedRecord instance via an indexer.

        Facilitates querying, accessing, manipulating, and setting edge attributes using
        various indexing methods. Supports conditions and ranges for more
        complex queries.

        Returns:
            EdgeIndexer: An object for manipulating and querying edge attributes.
        """
        return EdgeIndexer(self)

    @property
    def groups(self) -> List[Group]:
        """
        Lists the groups in the MedRecord instance.

        Returns a list of all groups currently defined within the MedRecord instance.

        Returns:
            List[Group]: A list of groups.
        """
        return self._medrecord.groups

    @overload
    def group(self, group: Group) -> List[NodeIndex]: ...

    @overload
    def group(self, group: List[Group]) -> Dict[Group, List[NodeIndex]]: ...

    def group(
        self, group: Union[Group, List[Group]]
    ) -> Union[List[NodeIndex], Dict[Group, List[NodeIndex]]]:
        """
        Returns the node indices associated with the specified group(s)
        in the MedRecord.

        If a single group is specified, returns a list of node indices for that group.
        If multiple groups are specified, returns a dictionary with each group name
        mapping to its list of node indices.

        Args:
            group (Union[Group, List[Group]]): One or more group names.

        Returns:
            Union[List[NodeIndex], Dict[Group, List[NodeIndex]]]: Node indices for each
            specified group.
        """
        groups = self._medrecord.group(group if isinstance(group, list) else [group])
        if isinstance(group, list):
            return groups
        return groups[group]

    @overload
    def outgoing_edges(self, node: NodeIndex) -> List[EdgeIndex]: ...

    @overload
    def outgoing_edges(
        self, node: Union[List[NodeIndex], NodeOperation]
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...

    def outgoing_edges(
        self, node: Union[NodeIndex, List[NodeIndex], NodeOperation]
    ) -> Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]:
        """
        Lists the outgoing edges of the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of its outgoing edge indices.
        If multiple nodes are specified, returns a dictionary mapping each node index to
        its list of outgoing edge indices.

        Args:
            node (Union[NodeIndex, List[NodeIndex], NodeOperation]): One or more
                node indices.

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
        self, node: Union[List[NodeIndex], NodeOperation]
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...

    def incoming_edges(
        self, node: Union[NodeIndex, List[NodeIndex], NodeOperation]
    ) -> Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]:
        """
        Lists the incoming edges of the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of its incoming edge indices.
        If multiple nodes are specified, returns a dictionary mapping each node index to
        its list of incoming edge indices.

        Args:
            node (Union[NodeIndex, List[NodeIndex], NodeOperation]): One or more
                node indices.

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
        self, edge: Union[List[EdgeIndex], EdgeOperation]
    ) -> Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]: ...

    def edge_endpoints(
        self, edge: Union[EdgeIndex, List[EdgeIndex], EdgeOperation]
    ) -> Union[
        tuple[NodeIndex, NodeIndex], Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]
    ]:
        """
        Retrieves the source and target nodes of the specified edge(s) in the MedRecord.

        If a single edge index is provided, returns a tuple of
        node indices (source, target). If multiple edges are specified, returns
        a dictionary mapping each edge index to its tuple of node indices.

        Args:
            edge (Union[EdgeIndex, List[EdgeIndex], EdgeOperation]): One or more
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
        source_node: Union[NodeIndex, List[NodeIndex], NodeOperation],
        target_node: Union[NodeIndex, List[NodeIndex], NodeOperation],
    ) -> List[EdgeIndex]:
        """
        Retrieves the edges connecting the specified source and target nodes in
        the MedRecord.

        If a NodeOperation is provided for either the source or target nodes, it is
        first evaluated to obtain the corresponding node indices. The method then
        returns a list of edge indices that connect the specified source and
        target nodes.

        Args:
            source_node (Union[NodeIndex, List[NodeIndex], NodeOperation]):
                The index or indices of the source node(s), or a NodeOperation to
                select source nodes.
            target_node (Union[NodeIndex, List[NodeIndex], NodeOperation]):
                The index or indices of the target node(s), or a NodeOperation to
                select target nodes.

        Returns:
            List[EdgeIndex]: A list of edge indices connecting the specified source and
                target nodes.
        """
        if isinstance(source_node, NodeOperation):
            source_node = self.select_nodes(source_node)

        if isinstance(target_node, NodeOperation):
            target_node = self.select_nodes(target_node)

        return self._medrecord.edges_connecting(
            (source_node if isinstance(source_node, list) else [source_node]),
            (target_node if isinstance(target_node, list) else [target_node]),
        )

    def add_node(self, node: NodeIndex, attributes: Attributes) -> None:
        """
        Adds a node with specified attributes to the MedRecord instance.

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
        self, node: Union[List[NodeIndex], NodeOperation]
    ) -> Dict[NodeIndex, Attributes]: ...

    def remove_node(
        self, node: Union[NodeIndex, List[NodeIndex], NodeOperation]
    ) -> Union[Attributes, Dict[NodeIndex, Attributes]]:
        """
        Removes a node or multiple nodes from the MedRecord and returns
        their attributes.

        If a single node index is provided, returns the attributes of the removed node.
        If multiple node indices are specified, returns a dictionary mapping each node
        index to its attributes.

        Args:
            node (Union[NodeIndex, List[NodeIndex], NodeOperation]): Node index or
                indices.

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
            List[tuple[NodeIndex, Attributes]],
            pd.DataFrame,
            List[pd.DataFrame],
            PolarsNodeDataFrameInput,
            List[PolarsNodeDataFrameInput],
        ],
    ) -> None:
        """
        Adds multiple nodes to the MedRecord from different data formats.

        Accepts a list of tuples, DataFrame(s), or PolarsNodeDataFrameInput(s) to add
        nodes. If a DataFrame or list of DataFrames is used, the add_nodes_pandas method
        is called. If PolarsNodeDataFrameInput(s) are provided, each tuple must include
        a DataFrame and the index column.

        Args:
            nodes (Union[List[tuple[NodeIndex, Attributes]], pd.DataFrame,
                List[pd.DataFrame], PolarsNodeDataFrameInput,
                List[PolarsNodeDataFrameInput]]): Data representing nodes
                in various formats.

        Returns:
            None
        """
        if isinstance(nodes, pd.DataFrame) or is_pandas_dataframe_list(nodes):
            return self.add_nodes_pandas(nodes)
        elif is_polars_node_dataframe_input(
            nodes
        ) or is_polars_node_dataframe_input_list(nodes):
            return self.add_nodes_polars(nodes)
        else:
            return self._medrecord.add_nodes(nodes)  # type: ignore

    def add_nodes_pandas(self, nodes: Union[pd.DataFrame, List[pd.DataFrame]]) -> None:
        """
        Adds nodes to the MedRecord instance from one or more pandas DataFrames.

        Each DataFrame is converted to a Polars DataFrame, and then nodes are added
        to the MedRecord instance. Each DataFrame must have an Index.

        Args:
            nodes (Union[pd.DataFrame, List[pd.DataFrame]]): DataFrame(s) representing
                nodes.

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
        """
        Adds nodes to the MedRecord instance from one or more Polars DataFrames.

        This method accepts either a single tuple or a list of tuples, where each tuple
        consists of a Polars DataFrame and an index column string.

        Args:
            nodes (Union[NodeDataFrameInput, List[NodeDataFrameInput]]): A tuple or list
                of tuples, each with a DataFrame and index column.

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
        attributes: Attributes,
    ) -> EdgeIndex:
        """
        Adds an edge between two specified nodes with given attributes.

        Args:
            source_node (NodeIndex): Index of the source node.
            target_node (NodeIndex): Index of the target node.
            attributes (Attributes): Dictionary of edge attributes.

        Returns:
            EdgeIndex: The index of the added edge.
        """
        return self._medrecord.add_edge(source_node, target_node, attributes)

    @overload
    def remove_edge(self, edge: EdgeIndex) -> Attributes: ...

    @overload
    def remove_edge(
        self, edge: Union[List[EdgeIndex], EdgeOperation]
    ) -> Dict[EdgeIndex, Attributes]: ...

    def remove_edge(
        self, edge: Union[EdgeIndex, List[EdgeIndex], EdgeOperation]
    ) -> Union[Attributes, Dict[EdgeIndex, Attributes]]:
        """
        Removes an edge or multiple edges from the MedRecord and returns
        their attributes.

        If a single edge index is provided, returns the attributes of the removed edge.
        If multiple edge indices are specified, returns a dictionary mapping each edge
        index to its attributes.

        Args:
            edge (Union[EdgeIndex, List[EdgeIndex], EdgeOperation]): Edge index
                or indices.

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
            List[tuple[NodeIndex, NodeIndex, Attributes]],
            pd.DataFrame,
            List[pd.DataFrame],
            PolarsEdgeDataFrameInput,
            List[PolarsEdgeDataFrameInput],
        ],
    ) -> List[EdgeIndex]:
        """
        Adds edges to the MedRecord instance from various data formats.

        Accepts lists of tuples, DataFrame(s), or EdgeDataFrameInput(s) to add edges.
        Each tuple must have indices for source and target nodes and a dictionary of
        attributes. If a DataFrame or list of DataFrames is used,
        the add_edges_dataframe method is invoked.

        Args:
            edges (Union[List[tuple[NodeIndex, NodeIndex, Attributes]], pd.DataFrame,
                List[pd.DataFrame], PolarsEdgeDataFrameInput,
                List[PolarsEdgeDataFrameInput]]):
                Data representing edges in several formats.

        Returns:
            List[EdgeIndex]: A list of edge indices that were added.
        """
        if isinstance(edges, pd.DataFrame) or is_pandas_dataframe_list(edges):
            return self.add_edges_pandas(edges)
        elif is_polars_edge_dataframe_input(
            edges
        ) or is_polars_edge_dataframe_input_list(edges):
            return self.add_edges_polars(edges)
        else:
            return self._medrecord.add_edges(edges)  # type: ignore

    def add_edges_pandas(
        self, edges: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> List[EdgeIndex]:
        """
        Adds edges to the MedRecord from one or more pandas DataFrames.

        Each DataFrame is converted to a Polars DataFrame, and then edges are added
        using add_edges_polars. Each DataFrame must have a MultiIndex with two levels
        for source and target nodes.

        Args:
            edges (Union[pd.DataFrame, List[pd.DataFrame]]): DataFrame(s) representing
                edges.

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
        """
        Adds edges to the MedRecord from one or more Polars DataFrames.

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
        node: Optional[Union[NodeIndex, List[NodeIndex], NodeOperation]] = None,
    ) -> None:
        """
        Adds a group to the MedRecord instance with an optional list of node indices.

        If node indices are specified, they are added to the group. If no nodes are
        specified, the group is created without any nodes.

        Args:
            group (Group): The name of the group to add.
            node (Optional[Union[NodeIndex, List[NodeIndex], NodeOperation]]):
                Node index or indices to add to the group, optional.

        Returns:
            None
        """
        if isinstance(node, NodeOperation):
            return self._medrecord.add_group(group, self.select_nodes(node))

        if node is None:
            return self._medrecord.add_group(group, None)

        return self._medrecord.add_group(
            group, node if isinstance(node, list) else [node]
        )

    def remove_group(self, group: Union[Group, List[Group]]) -> None:
        """
        Removes one or more groups from the MedRecord instance.

        Args:
            group (Union[Group, List[Group]]): One or more group names to remove.

        Returns:
            None
        """
        return self._medrecord.remove_group(
            group if isinstance(group, list) else [group]
        )

    def add_node_to_group(
        self, group: Group, node: Union[NodeIndex, List[NodeIndex], NodeOperation]
    ) -> None:
        """
        Adds one or more nodes to a specified group in the MedRecord.

        Args:
            group (Group): The name of the group to add nodes to.
            node (Union[NodeIndex, List[NodeIndex], NodeOperation]): Node index
                or indices to add to the group.

        Returns:
            None
        """
        if isinstance(node, NodeOperation):
            return self._medrecord.add_node_to_group(group, self.select_nodes(node))

        return self._medrecord.add_node_to_group(
            group, node if isinstance(node, list) else [node]
        )

    def remove_node_from_group(
        self, group: Group, node: Union[NodeIndex, List[NodeIndex], NodeOperation]
    ) -> None:
        """
        Removes one or more nodes from a specified group in the MedRecord.

        Args:
            group (Group): The name of the group from which to remove nodes.
            node (Union[NodeIndex, List[NodeIndex], NodeOperation]): Node index
                or indices to remove from the group.

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

    @overload
    def groups_of_node(self, node: NodeIndex) -> List[Group]: ...

    @overload
    def groups_of_node(
        self, node: Union[List[NodeIndex], NodeOperation]
    ) -> Dict[NodeIndex, List[Group]]: ...

    def groups_of_node(
        self, node: Union[NodeIndex, List[NodeIndex], NodeOperation]
    ) -> Union[List[Group], Dict[NodeIndex, List[Group]]]:
        """
        Retrieves the groups associated with the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of groups for that node.
        If multiple nodes are specified, returns a dictionary mapping each node index to
        its list of groups.

        Args:
            node (Union[NodeIndex, List[NodeIndex], NodeOperation]): Node index
                or indices.

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

    def node_count(self) -> int:
        """
        Returns the total number of nodes currently managed by the MedRecord.

        Returns:
            int: The total number of nodes.
        """
        return self._medrecord.node_count()

    def edge_count(self) -> int:
        """
        Returns the total number of edges currently managed by the MedRecord.

        Returns:
            int: The total number of edges.
        """
        return self._medrecord.edge_count()

    def group_count(self) -> int:
        """
        Returns the total number of groups currently defined within the MedRecord.

        Returns:
            int: The total number of groups.
        """
        return self._medrecord.group_count()

    def contains_node(self, node: NodeIndex) -> bool:
        """
        Checks whether a specific node exists in the MedRecord.

        Args:
            node (NodeIndex): The index of the node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        return self._medrecord.contains_node(node)

    def contains_edge(self, edge: EdgeIndex) -> bool:
        """
        Checks whether a specific edge exists in the MedRecord.

        Args:
            edge (EdgeIndex): The index of the edge to check.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        return self._medrecord.contains_edge(edge)

    def contains_group(self, group: Group) -> bool:
        """
        Checks whether a specific group exists in the MedRecord.

        Args:
            group (Group): The name of the group to check.

        Returns:
            bool: True if the group exists, False otherwise.
        """
        return self._medrecord.contains_group(group)

    @overload
    def neighbors(self, node: NodeIndex) -> List[NodeIndex]: ...

    @overload
    def neighbors(
        self, node: Union[List[NodeIndex], NodeOperation]
    ) -> Dict[NodeIndex, List[NodeIndex]]: ...

    def neighbors(
        self, node: Union[NodeIndex, List[NodeIndex], NodeOperation]
    ) -> Union[List[NodeIndex], Dict[NodeIndex, List[NodeIndex]]]:
        """
        Retrieves the neighbors of the specified node(s) in the MedRecord.

        If a single node index is provided, returns a list of its neighboring
        node indices. If multiple nodes are specified, returns a dictionary mapping
        each node index to its list of neighboring nodes.

        Args:
            node (Union[NodeIndex, List[NodeIndex], NodeOperation]): Node index
                or indices.

        Returns:
            Union[List[NodeIndex], Dict[NodeIndex, List[NodeIndex]]]: Neighboring nodes.
        """
        if isinstance(node, NodeOperation):
            return self._medrecord.neighbors(self.select_nodes(node))

        neighbors = self._medrecord.neighbors(
            node if isinstance(node, list) else [node]
        )

        if isinstance(node, list):
            return neighbors

        return neighbors[node]

    def clear(self) -> None:
        """
        Clears all data from the MedRecord instance.

        Removes all nodes, edges, and groups, effectively resetting the instance.

        Returns:
            None
        """
        return self._medrecord.clear()

    def select_nodes(self, operation: NodeOperation) -> List[NodeIndex]:
        """
        Selects nodes based on a specified operation and returns their indices.

        Args:
            operation (NodeOperation): The operation to apply to select nodes.

        Returns:
            List[NodeIndex]: A list of node indices that satisfy the operation.
        """
        return self._medrecord.select_nodes(operation._node_operation)

    def select_edges(self, operation: EdgeOperation) -> List[EdgeIndex]:
        """
        Selects edges based on a specified operation and returns their indices.

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
        """
        Allows selection of nodes or edges using operations directly via indexing.

        Args:
            key (Union[NodeOperation, EdgeOperation]): Operation to select nodes
                or edges.

        Returns:
            Union[List[NodeIndex], List[EdgeIndex]]: Node or edge indices selected.
        """
        if isinstance(key, NodeOperation):
            return self.select_nodes(key)

        return self.select_edges(key)
