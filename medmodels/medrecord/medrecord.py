from typing import Dict, List, Optional, Union

import pandas as pd
import polars as pl

from medmodels._medmodels import PyMedRecord
from medmodels.medrecord.indexers import _EdgeIndexer, _NodeIndexer
from medmodels.medrecord.querying import EdgeOperation, NodeOperation
from medmodels.medrecord.types import (
    Attributes,
    EdgeIndex,
    Group,
    NodeIndex,
    NodeDataFrameInput,
    EdgeDataFrameInput,
    is_node_dataframe_input,
    is_edge_dataframe_input,
)


def process_nodes_dataframe(nodes: pd.DataFrame) -> NodeDataFrameInput:
    assert isinstance(nodes.index, pd.Index), "Nodes dataframe must have an Index"

    nodes_index_column = nodes.index.name

    assert nodes_index_column is not None, "Nodes dataframe must have an Index"

    nodes_polars = pl.from_pandas(nodes, include_index=True)

    return nodes_polars, nodes_index_column


def process_edges_dataframe(
    edges: pd.DataFrame,
) -> EdgeDataFrameInput:
    assert isinstance(
        edges.index, pd.MultiIndex
    ), "Edges dataframe must have a MultiIndex"

    edges_index_names = edges.index.names
    assert len(edges_index_names) == 2, "Edges dataframe MultiIndex must have 2 levels"

    edges_source_index_column = edges_index_names[0]
    edges_target_index_column = edges_index_names[1]

    edges_polars = pl.from_pandas(edges, include_index=True)

    return edges_polars, edges_source_index_column, edges_target_index_column


class MedRecord:
    _medrecord: PyMedRecord

    def __init__(self) -> None:
        self._medrecord = PyMedRecord()

    @classmethod
    def from_tuples(
        cls,
        nodes: List[tuple[NodeIndex, Attributes]],
        edges: Optional[List[tuple[NodeIndex, NodeIndex, Attributes]]] = [],
    ) -> "MedRecord":
        """
        Creates a new MedRecord instance from node and edge tuples.

        This class method takes a list of tuples representing nodes and edges, and
        creates a new MedRecord instance using these tuples. Each node tuple should
        contain a node index and a dictionary of node attributes. Each edge tuple should
        contain two node indices representing the source and target of the edge, and a
        dictionary of edge attributes.

        Args:
            nodes (List[tuple[NodeIndex, Attributes]]): A list of tuples representing
                nodes. Each tuple should contain a node index and a dictionary
                of node attributes.
            edges (Optional[List[tuple[NodeIndex, NodeIndex, Attributes]]], optional):
                An optional list of tuples representing edges. Each tuple should contain
                two node indices representing the source and target of the edge, and a
                dictionary of edge attributes. Defaults to an empty list.

        Returns:
            MedRecord: A new MedRecord instance created from the provided nodes
                and edges.
        """
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_tuples(nodes, edges)

        return medrecord

    @classmethod
    def from_pandas(
        cls,
        nodes: Union[pd.DataFrame, List[pd.DataFrame]],
        edges: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ) -> "MedRecord":
        """
        Creates a new MedRecord instance from pandas DataFrames of nodes and
        optionally edges.

        This class method takes either a single pandas DataFrame or a list of DataFrames
        representing nodes, and optionally a single DataFrame or a list of DataFrames
        representing edges, to create a new MedRecord instance. Each DataFrame in the
        nodes list must have an Index. Each DataFrame in the edges list, if provided,
        must have a MultiIndex with exactly 2 levels.

        Args:
            nodes (Union[pd.DataFrame, List[pd.DataFrame]]): A DataFrame or a list
                of DataFrames representing nodes. Each DataFrame must have an Index.
            edges (Optional[Union[pd.DataFrame, List[pd.DataFrame]]], optional):
                An optional DataFrame or list of DataFrames representing edges.
                If provided, each DataFrame must have a MultiIndex with exactly
                2 levels. Defaults to None.

        Returns:
            MedRecord: A new MedRecord instance created from the provided nodes
            and edges DataFrames.
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
        nodes: Union[NodeDataFrameInput, List[NodeDataFrameInput]],
        edges: Optional[Union[EdgeDataFrameInput, List[EdgeDataFrameInput]]] = None,
    ) -> "MedRecord":
        """
        Creates a new MedRecord from Polars DataFrames of nodes and optionally edges.

        This method accepts either a tuple or a list of tuples for nodes, each
        consisting of a Polars DataFrame and an index column. Similarly, it can also
        accept a tuple or list of tuples for edges, each containing a DataFrame and
        two strings for the index columns of source and target nodes. This setup allows
        for building a MedRecord from multiple data segments.

        Args:
            nodes (Union[NodeDataFrameInput, List[NodeDataFrameInput]]): A tuple or
                list of tuples, each with a Polars DataFrame and an index column string.
            edges (Optional[Union[EdgeDataFrameInput, List[EdgeDataFrameInput]]],
                optional): An optional tuple or list of tuples, each with a DataFrame,
                and strings for the index columns of source nodes and target nodes.
                Defaults to None.

        Returns:
            MedRecord: A new instance created from the provided nodes and edges data.
        """

        if edges is None:
            medrecord = cls.__new__(cls)
            medrecord._medrecord = PyMedRecord.from_nodes_dataframe(
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
    def from_ron(cls, path: str) -> "MedRecord":
        """
        Creates a new MedRecord instance from a RON file.

        This class method takes a path to a RON file and creates a new MedRecord
        instance using the data in the RON file.

        Args:
            path (str): The path to the RON file.

        Returns:
            MedRecord: A new MedRecord instance created from the RON file.
        """

        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_ron(path)

        return medrecord

    def to_ron(self, path: str) -> None:
        """
        Writes the MedRecord instance to a RON file.

        This method takes a path and writes the MedRecord instance to a RON file.

        Args:
            path (str): The path to write the RON file to.

        Returns:
            None
        """
        return self._medrecord.to_ron(path)

    @property
    def nodes(self) -> List[NodeIndex]:
        """
        Returns the node indices in the MedRecord instance.

        Returns:
            List[NodeIndex]: The node indices in the MedRecord instance.
        """
        return self._medrecord.nodes

    @property
    def node(self) -> _NodeIndexer:
        """
        Provides access to node attributes within a MedRecord instance via an
        indexer.

        This property returns an _NodeIndexer instance that facilitates querying,
        accessing, manipulating, and setting node attributes through various
        indexing methods. Supports simple to complex and conditional selections.

        Examples of usage:
        - Retrieving attributes:
            # Retrieves all attributes of node 1
            medrecord.node[1]
            # Retrieves the value of attribute "foo" of node 1
            medrecord.node[1, "foo"]
            # Retrieves attribute "foo" for all nodes
            medrecord.node[:, "foo"]
            # Retrieves attributes "foo" and "bar" for nodes 1 and 2
            medrecord.node[[1, 2], ["foo", "bar"]]
            # Retrieves all attributes of nodes with index >= 2
            medrecord.node[node().index() >= 2]

        - Setting, updating or adding attributes:
            # Sets the attributes of node 1
            medrecord.node[1] = {"foo": "bar"}
            # Sets or adds the attribute "foo" for node 1
            medrecord.node[1, "foo"] = "test"
            # Sets or adds the attributes "foo" and "bar" for node 1
            medrecord.node[1, ["foo", "bar"]] = "test"
            # Sets or adds the attribute "foo" for all nodes
            medrecord.node[:, "foo"] = "test"

        - Deleting attributes:
            # Deletes attribute "foo" from node 1
            del medrecord.node[1, "foo"]
            # Deletes attribute "foo" from all nodes
            del medrecord.node[:, "foo"]

        Returns:
            _NodeIndexer: An object that enables manipulation and querying of
            node attributes within a MedRecord.

        Note:
            Operations via the indexer directly update the MedRecord's internal
            representation of nodes.
        """
        return _NodeIndexer(self)

    @property
    def edges(self) -> List[EdgeIndex]:
        """
        Returns the edge indices in the MedRecord instance.

        Returns:
            List[EdgeIndex]: The edge indices in the MedRecord instance.
        """
        return self._medrecord.edges

    @property
    def edge(self) -> _EdgeIndexer:
        """
        Provides access to edge attributes within a MedRecord instance via an
        indexer.

        This property returns an _EdgeIndexer instance that facilitates querying,
        accessing, manipulating, and setting edge attributes through various
        indexing methods. Supports simple to complex and conditional selections.

        Examples of usage:
        - Retrieving attributes:
            # Retrieves all attributes of edge 1
            medrecord.edge[1]
            # Retrieves the value of attribute "foo" of edge 1
            medrecord.edge[1, "foo"]
            # Retrieves attribute "foo" for all edges
            medrecord.edge[:, "foo"]
            # Retrieves attributes "foo" and "bar" for edges 1 and 2
            medrecord.edge[[1, 2], ["foo", "bar"]]
            # Retrieves all attributes of edges with index >= 2
            medrecord.edge[edge().index() >= 2]

        - Setting, updating or adding attributes:
            # Sets the attributes of edge 1
            medrecord.edge[1] = {"foo": "bar"}
            # Sets or adds the attribute "foo" for edge 1
            medrecord.edge[1, "foo"] = "test"
            # Sets or adds the attributes "foo" and "bar" for edge 1
            medrecord.edge[1, ["foo", "bar"]] = "test"
            # Sets or adds the attribute "foo" for all edges
            medrecord.edge[:, "foo"] = "test"

        - Deleting attributes:
            # Deletes attribute "foo" from edge 1
            del medrecord.edge[1, "foo"]
            # Deletes attribute "foo" from all edges
            del medrecord.edge[:, "foo"]

        Returns:
            _EdgeIndexer: An object that enables manipulation and querying of
            edge attributes within a MedRecord.

        Note:
            Operations via the indexer directly update the MedRecord's internal
            representation of edges.
        """
        return _EdgeIndexer(self)

    @property
    def groups(self) -> List[Group]:
        """
        Returns the groups in the MedRecord instance.

        Returns:
            List[Group]: The groups in the MedRecord instance.
        """
        return self._medrecord.groups

    def group(
        self, *group: Group
    ) -> Union[List[NodeIndex], Dict[Group, List[NodeIndex]]]:
        """
        Returns the node indices in the specified group(s) in the MedRecord instance.

        This method takes one or more groups as arguments and returns a dictionary of
        the group names and the node indices in each group.

        Args:
            *group (Group): One or more group names to get the nodes from.

        Returns:
            Union[List[NodeIndex], Dict[Group, List[NodeIndex]]]: Node indices for
                a single group if one is specified, or a dictionary of groups to
                their node indices if multiple groups are provided.
        """
        groups = self._medrecord.group(*group)

        if len(groups) == 1:
            return groups[group[0]]

        return groups

    def outgoing_edges(
        self, *node_index: NodeIndex
    ) -> Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]:
        """
        Returns the outgoing edges of the specified node(s) in the MedRecord instance.

        This method takes one or more node indices as arguments and returns a dictionary
        of the node indices and the indices of the outgoing edges of each node.

        Args:
            *node_index (NodeIndex): One or more node indices.

        Returns:
            Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]: List of
                outgoing edge indices for a single node if one index is provided,
                or a dictionary mapping each node index to its list of
                outgoing edge indices if multiple nodes are specified.
        """
        indices = self._medrecord.outgoing_edges(*node_index)

        if len(indices) == 1:
            return indices[node_index[0]]

        return indices

    def incoming_edges(
        self, *node_index: NodeIndex
    ) -> Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]:
        """
        Returns the incoming edges of the specified node(s) in the MedRecord instance.

        This method takes one or more node indices as arguments and returns a dictionary
        of the node indices and the indices of the incoming edges of each node.

        Args:
            *node_index (NodeIndex): One or more node indices.

        Returns:
            Union[List[EdgeIndex], Dict[NodeIndex, List[EdgeIndex]]]: List of
                incoming edge indices for a single node if one index is provided,
                or a dictionary mapping each node index to its list of
                incoming edge indices if multiple nodes are specified.
        """
        indices = self._medrecord.incoming_edges(*node_index)

        if len(indices) == 1:
            return indices[node_index[0]]

        return indices

    def edge_endpoints(
        self, *edge_index: EdgeIndex
    ) -> Union[
        tuple[NodeIndex, NodeIndex], Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]
    ]:
        """
        Returns the source and target nodes of the specified edge(s) in the MedRecord instance.

        This method takes one or more edge indices as arguments and returns a dictionary
        of the edge indices and the source and target nodes of each edge.

        Args:
            *edge_index (EdgeIndex): One or more edge indices.

        Returns:
            Union[
                tuple[NodeIndex, NodeIndex],
                Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]
            ]: Tuple of node indices (source, target) for a single edge if one index is
                provided, or a dictionary mapping each edge index to a tuple of
                node indices if multiple edges are specified.
        """
        endpoints = self._medrecord.edge_endpoints(*edge_index)

        if len(endpoints) == 1:
            return endpoints[edge_index[0]]

        return endpoints

    def edges_connecting(
        self, source_node_index: NodeIndex, target_node_index: NodeIndex
    ) -> List[EdgeIndex]:
        """
        Returns the edge indices between the specified source and target nodes in the
        MedRecord instance.

        This method takes a source node index and a target node index as arguments and
        returns the edge indices between these nodes in the MedRecord instance.

        Args:
            source_node_index (NodeIndex): The index of the source node.
            target_node_index (NodeIndex): The index of the target node.

        Returns:
            List[EdgeIndex]: A list of edge indices.
        """
        return self._medrecord.edges_connecting(source_node_index, target_node_index)

    def add_node(self, node_index: NodeIndex, attributes: Attributes) -> None:
        """
        Adds a node to the MedRecord instance.

        This method takes a node index and a dictionary of the node's attributes,
        and adds the node to the MedRecord instance.

        Args:
            node_index (NodeIndex): The index of the node to add.
            attributes (Attributes): A dictionary of the node's attributes.

        Returns:
            None
        """
        return self._medrecord.add_node(node_index, attributes)

    def remove_node(
        self, *node_index: NodeIndex
    ) -> Union[Attributes, Dict[NodeIndex, Attributes]]:
        """
        Removes a node from the MedRecord instance.

        This method takes one or more node indices as arguments, removes the nodes
        from the MedRecord instance and returns a dictionary of the node indices and
        their attributes.

        Args:
            *node_index (NodeIndex): One or more node indices to remove.

        Returns:
            Union[Attributes, Dict[NodeIndex, Attributes]]: Attributes of the
                removed node if one index is provided, or a dictionary of node indices
                to their attributes if multiple indices are provided.
        """
        attributes = self._medrecord.remove_node(*node_index)

        if len(attributes) == 1:
            return attributes[node_index[0]]

        return attributes

    def add_nodes(
        self,
        nodes: Union[
            List[tuple[NodeIndex, Attributes]],
            pd.DataFrame,
            List[pd.DataFrame],
            NodeDataFrameInput,
            List[NodeDataFrameInput],
        ],
    ) -> None:
        """
        Adds nodes to the MedRecord instance.

        This method can accept various forms of data: a list of tuples, a DataFrame, a
        list of DataFrames, or an NodeDataFrameInput (tuple) / list of such tuples. It
        adds these to the MedRecord instance. If a DataFrame or list of DataFrames is
        used, add_nodes_pandas is called. If NodeDataFrameInput(s) are provided, each
        tuple must include a DataFrame and the index column

        Args:
            nodes (Union[List[tuple[NodeIndex, Attributes]], pd.DataFrame,
                List[pd.DataFrame], NodeDataFrameInput, List[NodeDataFrameInput]]):
                Data representing nodes. This can be a list of tuples, each with a node
                index and attributes; a DataFrame; a list of DataFrames; or a tuple (or
                list of tuples) with a DataFrame and index column.

        Returns:
            None: Nodes are added to the instance without a return value.
        """
        if isinstance(nodes, pd.DataFrame) or (
            isinstance(nodes, list) and isinstance(nodes[0], pd.DataFrame)
        ):
            return self.add_nodes_pandas(nodes)

        if is_node_dataframe_input(nodes) or (
            isinstance(nodes, list) and is_node_dataframe_input(nodes[0])
        ):
            print("in here")
            return self.add_nodes_polars(nodes)

        return self._medrecord.add_nodes(nodes)

    def add_nodes_pandas(self, nodes: Union[pd.DataFrame, List[pd.DataFrame]]) -> None:
        """
        Adds nodes to the MedRecord instance from one or more pandas DataFrames.

        This method can take a single pandas DataFrame or a list of DataFrames
        representing nodes, converts each to a polars DataFrame, and then adds the
        nodes to the MedRecord instance using the add_nodes_polars method. Each
        DataFrame must have an Index.

        Args:
            nodes (Union[pd.DataFrame, List[pd.DataFrame]]): A DataFrame or list
                of DataFrames representing nodes. Each DataFrame must have an Index.

        Returns:
            None: Nodes are added to the instance without a return value.
        """
        return self.add_nodes_polars(
            [process_nodes_dataframe(nodes_df) for nodes_df in nodes]
            if isinstance(nodes, list)
            else [process_nodes_dataframe(nodes)]
        )

    def add_nodes_polars(
        self, nodes: Union[NodeDataFrameInput, List[NodeDataFrameInput]]
    ) -> None:
        """
        Adds nodes to the MedRecord instance from one or more polars DataFrames.

        This method takes either a single tuple or a list of tuples, with each tuple
        comprising a polars DataFrame and a string representing the index column, and
        adds the nodes to the MedRecord instance.

        Args:
            nodes (Union[NodeDataFrameInput, List[NodeDataFrameInput]]): A tuple or list
                of tuples, each containing a polars DataFrame and an
                index column string.

        Returns:
            None: Nodes are added to the instance without a return value.
        """
        return self._medrecord.add_nodes_dataframes(
            nodes if isinstance(nodes, list) else [nodes]
        )

    def add_edge(
        self,
        source_node_index: NodeIndex,
        target_node_index: NodeIndex,
        attributes: Attributes,
    ) -> EdgeIndex:
        """
        Adds an edge to the MedRecord instance.

        This method takes the indices of the source and target nodes and a dictionary
        of the edge's attributes, adds the edge to the MedRecord instance and returns
        the index of the edge that was added.

        Args:
            source_node_index (NodeIndex): The index of the source node.
            target_node_index (NodeIndex): The index of the target node.
            attributes (Attributes): A dictionary of the edge's attributes.

        Returns:
            EdgeIndex: The index of the edge that was added.
        """
        return self._medrecord.add_edge(
            source_node_index, target_node_index, attributes
        )

    def remove_edge(
        self, *edge_index: EdgeIndex
    ) -> Union[Attributes, Dict[EdgeIndex, Attributes]]:
        """
        Removes an edge from the MedRecord instance.

        This method takes one or more edge indices as arguments, removes the edges
        from the MedRecord instance and returns a dictionary of the edge indices and
        their attributes.

        Args:
            *edge_index (EdgeIndex): One or more edge indices to remove.

        Returns:
            Union[Attributes, Dict[EdgeIndex, Attributes]]: Attributes of the
                removed edge if one index is provided, or a dictionary of edge indices
                to their attributes if multiple indices are provided.
        """
        attributes = self._medrecord.remove_edge(*edge_index)

        if len(attributes) == 1:
            return attributes[edge_index[0]]

        return attributes

    def add_edges(
        self,
        edges: Union[
            List[tuple[NodeIndex, NodeIndex, Attributes]],
            pd.DataFrame,
            List[pd.DataFrame],
            EdgeDataFrameInput,
            List[EdgeDataFrameInput],
        ],
    ) -> List[EdgeIndex]:
        """
        Adds edges to the MedRecord instance from various data formats.

        This method accepts lists of tuples, individual or lists of pandas DataFrames,
        or EdgeDataFrameInput(s). Each tuple must have indices for source and target
        nodes and a dictionary of attributes. DataFrames must have a MultiIndex with
        two levels for source and target nodes. If a DataFrame or list of DataFrames is
        provided, the add_edges_dataframe method is invoked.

        Args:
            edges (Union[List[tuple[NodeIndex, NodeIndex, Attributes]], pd.DataFrame,
                List[pd.DataFrame], EdgeDataFrameInput, List[EdgeDataFrameInput]]):
                Data representing edges, which can be in several formats.

        Returns:
            List[EdgeIndex]: A list of edge indices that were added.
        """
        if isinstance(edges, pd.DataFrame) or (
            isinstance(edges, list) and isinstance(edges[0], pd.DataFrame)
        ):
            return self.add_edges_pandas(edges)

        if isinstance(edges, tuple) or (
            isinstance(edges, list) and is_edge_dataframe_input(edges[0])
        ):
            return self.add_edges_polars(edges)

        return self._medrecord.add_edges(edges)

    def add_edges_pandas(
        self, edges: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> List[EdgeIndex]:
        """
        Adds edges to the MedRecord from one or more pandas DataFrames.

        This method takes either a single DataFrame or a list, converts them to
        polars DataFrames, and adds them using add_edges_polars. It returns a list
        of edge indices added. Each DataFrame must have a MultiIndex with two levels
        for source and target nodes.

        Args:
            edges (Union[pd.DataFrame, List[pd.DataFrame]]): DataFrame(s) representing
                edges, each must have a MultiIndex with two levels.

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
        edges: Union[EdgeDataFrameInput, List[EdgeDataFrameInput]],
    ) -> List[EdgeIndex]:
        """
        Adds edges to the MedRecord from one or more polars DataFrames.

        This method accepts either a single EdgeDataFrameInput tuple or a list of
        such tuples, where each tuple consists of a polars DataFrame and two strings
        representing the index columns for the source and target nodes. It adds these
        edges to the MedRecord instance and returns a list of edge indices that were
        added.

        Args:
            edges (Union[EdgeDataFrameInput, List[EdgeDataFrameInput]]): A tuple or list
                of tuples, each including a DataFrame and strings for source and
                target node index columns.

        Returns:
            List[EdgeIndex]: A list of the edge indices that were added.
        """
        return self._medrecord.add_edges_dataframes(
            edges if isinstance(edges, list) else [edges]
        )

    def add_group(
        self, group: Group, node_indices_to_add: Optional[List[NodeIndex]] = None
    ) -> None:
        """
        Adds a group to the MedRecord instance.

        This method takes a group name and an optional list of node indices, and adds
        the group to the the MedRecord instance.

        Args:
            group (Group): The name of the group to add.
            node_indices_to_add (Optional[List[NodeIndex]], optional): A list of node
                indices to add to the group. If None, no nodes are added to the group.

        Returns:
            None
        """
        return self._medrecord.add_group(group, node_indices_to_add)

    def remove_group(self, *group: Group) -> None:
        """
        Removes a group from the MedRecord instance.

        This method takes one or more group names as arguments and removes the groups
        from the MedRecord instance.

        Args:
            *group (Group): One or more group names to remove.

        Returns:
            None
        """
        return self._medrecord.remove_group(*group)

    def add_node_to_group(self, group: Group, *node_index: NodeIndex) -> None:
        """
        Adds a node to a group in the MedRecord instance.

        This method takes a group name and one or more node indices, and adds them to
        the specified group in the MedRecord.

        Args:
            group (Group): The name of the group to which to add the node.
            *node_index (NodeIndex): The index/indices of the nodes to add to the group.

        Returns:
            None
        """
        return self._medrecord.add_node_to_group(group, *node_index)

    def remove_node_from_group(self, group: Group, *node_index: NodeIndex) -> None:
        """
        Removes a node from a group in the MedRecord instance.

        This method takes a group name and one or more node indices, and removes them
        from the specified group in the MedRecord.

        Args:
            group (Group): The name of the group from which to remove the node.
            *node_index (NodeIndex): The index/indices of the nodes to remove from the
                group.

        Returns:
            None
        """
        return self._medrecord.remove_node_from_group(group, *node_index)

    def groups_of_node(
        self, *node_index: NodeIndex
    ) -> Union[List[Group], Dict[NodeIndex, List[Group]]]:
        """
        Returns the groups of the specified node(s) in the MedRecord instance.

        This method takes one or more node indices as arguments and returns a dictionary
        of the node indices and the groups to which they belong.

        Args:
            *node_index (NodeIndex): The index/indices of the node(s) for which to
                retrieve groups.

        Returns:
            Union[List[Group], Dict[NodeIndex, List[Group]]]: List of groups for a
                single node if one index is provided, or a dictionary mapping each
                node index to its list of groups if multiple nodes are specified.
        """
        groups = self._medrecord.groups_of_node(*node_index)

        if len(groups) == 1:
            return groups[node_index[0]]

        return groups

    def node_count(self) -> int:
        """
        Returns the number of nodes in the MedRecord instance.

        Returns:
            int: The number of nodes in the MedRecord instance.
        """
        return self._medrecord.node_count()

    def edge_count(self) -> int:
        """
        Returns the number of edges in the MedRecord instance.

        Returns:
            int: The number of edges in the MedRecord instance.
        """
        return self._medrecord.edge_count()

    def group_count(self) -> int:
        """
        Returns the number of groups in the MedRecord instance.

        Returns:
            int: The number of groups in the MedRecord instance.
        """
        return self._medrecord.group_count()

    def contains_node(self, node_index: NodeIndex) -> bool:
        """
        Checks if a node exists in the MedRecord instance.

        This method takes a node index as an argument and checks if the node exists in
        the MedRecord instance.

        Args:
            node_index (NodeIndex): The index of the node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        return self._medrecord.contains_node(node_index)

    def contains_edge(self, edge_index: EdgeIndex) -> bool:
        """
        Checks if an edge exists in the MedRecord instance.

        This method takes an edge index as an argument and checks if the edge exists in
        the MedRecord instance.

        Args:
            edge_index (EdgeIndex): The index of the edge to check.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        return self._medrecord.contains_edge(edge_index)

    def contains_group(self, group: Group) -> bool:
        """
        Checks if a group exists in the MedRecord instance.

        This method takes a group name as an argument and checks if the group exists in
        the MedRecord instance.

        Args:
            group (Group): The name of the group to check.

        Returns:
            bool: True if the group exists, False otherwise.
        """
        return self._medrecord.contains_group(group)

    def neighbors(
        self, *node_index: NodeIndex
    ) -> Union[List[NodeIndex], Dict[NodeIndex, List[NodeIndex]]]:
        """
        Retrieves the neighbors of a node in the MedRecord instance.

        This method takes one or more node indices and retrieves the neighbors of the
        specified nodes in the MedRecord instance.  The return type is a dictionary of
        the node's index and the node indices of the neighboring nodes.

        Args:
            *node_index (NodeIndex): The index/indices of the node(s) for which to
            retrieve neighbors.

        Returns:
            Union[List[NodeIndex], Dict[NodeIndex, List[NodeIndex]]]: List of
                neighbor node indices for a single node if one index is provided,
                or a dictionary mapping each node index to its list of
                neighbor node indices if multiple nodes are specified.
        """
        neighbors = self._medrecord.neighbors(*node_index)

        if len(neighbors) == 1:
            return neighbors[node_index[0]]

        return neighbors

    def clear(self) -> None:
        """
        Clears the MedRecord instance.

        Removes all nodes, edges, and groups from the MedRecord instance.

        Returns:
            None
        """
        return self._medrecord.clear()

    def select_nodes(self, operation: NodeOperation) -> List[NodeIndex]:
        """
        Selects nodes from the MedRecord instance.

        This method takes a NodeOperation as an argument and returns a list of node
        indices that satisfy the operation.

        Args:
            operation (NodeOperation): The NodeOperation to apply to the nodes.

        Returns:
            List[NodeIndex]: A list of node indices that satisfy the operation.
        """
        return self._medrecord.select_nodes(operation._node_operation)

    def select_edges(self, operation: EdgeOperation) -> List[EdgeIndex]:
        """
        Selects edges from the MedRecord instance.

        This method takes a EdgeOperation as an argument and returns a list of edge
        indices that satisfy the operation.

        Args:
            operation (EdgeOperation): The EdgeOperation to apply to the edges.

        Returns:
            List[EdgeIndex]: A list of edge indices that satisfy the operation.
        """
        return self._medrecord.select_edges(operation._edge_operation)

    def __getitem__(
        self, key: Union[NodeOperation, EdgeOperation]
    ) -> Union[List[NodeIndex], List[EdgeIndex]]:
        if isinstance(key, NodeOperation):
            return self.select_nodes(key)
        elif isinstance(key, EdgeOperation):
            return self.select_edges(key)
        else:
            raise TypeError("Key must be a NodeOperation or EdgeOperation")
