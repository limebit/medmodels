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
)


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
        cls, nodes: pd.DataFrame, edges: Optional[pd.DataFrame] = None
    ) -> "MedRecord":
        """
        Creates a new MedRecord instance from pandas DataFrames of nodes and edges.

        This class method takes a pandas DataFrame representing nodes and optionally a
        DataFrame representing edges, and creates a new MedRecord instance using these
        DataFrames. The nodes DataFrame must have an Index. If an edges DataFrame is
        provided, it must have a MultiIndex with exactly 2 levels.

        Args:
            nodes (pd.DataFrame): A DataFrame representing nodes. Must have an Index.
            edges (Optional[pd.DataFrame], optional): An optional DataFrame representing
                edges. If provided, it must have a MultiIndex with exactly 2 levels.
                Defaults to None.

        Returns:
            MedRecord: A new MedRecord instance created from the provided nodes and
                edges DataFrames.
        """

        assert isinstance(nodes.index, pd.Index), "Nodes dataframe must have an Index"

        nodes_index_column = nodes.index.name

        assert nodes_index_column is not None, "Nodes dataframe must have an Index"

        nodes_polars = pl.from_pandas(nodes, include_index=True)

        if edges is None:
            medrecord = cls.__new__(cls)
            medrecord._medrecord = PyMedRecord.from_nodes_dataframe(
                nodes_polars, nodes_index_column
            )

            return medrecord

        assert isinstance(
            edges.index, pd.MultiIndex
        ), "Edges dataframe must have a MultiIndex"

        edges_index_names = edges.index.names
        assert (
            len(edges_index_names) == 2
        ), "Edges dataframe MultiIndex must have 2 levels"

        edges_from_index_column = edges_index_names[0]
        edges_to_index_column = edges_index_names[1]

        edges_polars = pl.from_pandas(edges, include_index=True)

        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_dataframes(
            nodes_polars,
            nodes_index_column,
            edges_polars,
            edges_from_index_column,
            edges_to_index_column,
        )

        return medrecord

    @classmethod
    def from_polars(
        cls,
        nodes: pl.DataFrame,
        nodes_index_column: str,
        edges: Optional[pl.DataFrame] = None,
        edges_from_index_column: Optional[str] = None,
        edges_to_index_column: Optional[str] = None,
    ) -> "MedRecord":
        """
        Creates a new MedRecord instance from polars DataFrames of nodes and edges.

        This class method takes a polars DataFrame representing nodes and
        optionally a DataFrame representing edges, and creates a new MedRecord
        instance using these DataFrames. If an edges DataFrame is provided,
        edges_from_index_column and edges_to_index_column must also be provided.

        Args:
            nodes (pl.DataFrame): A DataFrame representing nodes.
            nodes_index_column (str): The index column for the nodes DataFrame.
            edges (Optional[pl.DataFrame], optional): An optional DataFrame representing
                edges. Defaults to None.
            edges_from_index_column (Optional[str], optional): The from index column for
                the edges DataFrame. Must be provided if edges is not None. Defaults
                to None.
            edges_to_index_column (Optional[str], optional): The to index column for the
                edges DataFrame. Must be provided if edges is not None.
                Defaults to None.

        Returns:
            MedRecord: A new MedRecord instance created from the provided
                nodes and edges DataFrames.
        """

        if edges is None:
            medrecord = cls.__new__(cls)
            medrecord._medrecord = PyMedRecord.from_nodes_dataframe(
                nodes, nodes_index_column
            )

            return medrecord

        assert (
            edges_from_index_column is not None
        ), "edges_from_index_column argument needs to be set when edges is not None"
        assert (
            edges_to_index_column is not None
        ), "edges_to_index_column argument needs to be set when edges is not None"

        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_dataframes(
            nodes,
            nodes_index_column,
            edges,
            edges_from_index_column,
            edges_to_index_column,
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
        self, nodes: Union[List[tuple[NodeIndex, Attributes]], pd.DataFrame]
    ) -> None:
        """
        Adds nodes to the MedRecord instance.

        This method takes a list of tuples or a pandas DataFrame representing nodes
        and adds them to the MedRecord instance. If a DataFrame is provided,
        the add_nodes_pandas method is called.

        Args:
            nodes (Union[List[tuple[NodeIndex, Attributes]], pd.DataFrame]): A list of
                tuples, each containing a node index and a dictionary of the node's
                attributes, or a DataFrame representing nodes.

        Returns:
            None
        """
        if isinstance(nodes, pd.DataFrame):
            return self.add_nodes_pandas(nodes)

        return self._medrecord.add_nodes(nodes)

    def add_nodes_pandas(self, nodes: pd.DataFrame) -> None:
        """
        Adds nodes to the MedRecord instance from a pandas DataFrame.

        This method takes a pandas DataFrame representing nodes, converts it to a
        polars DataFrame, and adds the nodes to the MedRecord  instance using the
        add_nodes_polars method. The DataFrame must have an Index.

        Args:
            nodes (pd.DataFrame): A DataFrame representing nodes. The DataFrame must
                have an Index.

        Returns:
            None
        """
        assert isinstance(nodes.index, pd.Index), "Nodes dataframe must have an Index"

        nodes_index_column = nodes.index.name

        assert nodes_index_column is not None, "Nodes dataframe must have an Index"

        nodes_polars = pl.from_pandas(nodes, include_index=True)

        return self.add_nodes_polars(nodes_polars, nodes_index_column)

    def add_nodes_polars(self, nodes: pl.DataFrame, nodes_index_column: str) -> None:
        """
        Adds nodes to the MedRecord instance from a polars DataFrame.

        This method takes a polars DataFrame representing nodes and a string
        representing the index column, and adds the nodes to the MedRecord instance.

        Args:
            nodes (pl.DataFrame): A DataFrame representing nodes.
            nodes_index_column (str): The index column for the nodes DataFrame.

        Returns:
            None
        """
        return self._medrecord.add_nodes_dataframe(nodes, nodes_index_column)

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
        self, edges: Union[List[tuple[NodeIndex, NodeIndex, Attributes]], pd.DataFrame]
    ) -> List[EdgeIndex]:
        """
        Adds edges to the MedRecord instance.

        This method takes a list of tuples or a pandas DataFrame representing edges,
        adds them to the MedRecord instance and returns a list of edge indices that
        werde added. Each tuple contains the index of the source node, the index of the
        target node, and a dictionary of the edge's attributes. If a DataFrame is
        provided, the add_edges_dataframe method is called. The DataFrame must have a
        MultiIndex with two levels, representing the source and target nodes.

        Args:
            edges (Union[List[tuple[NodeIndex, NodeIndex, Attributes]], pd.DataFrame]):
                A list of tuples, each containing the index of the source node, the
                index of the target node, and a dictionary of the edge's attributes,
                or a DataFrame representing edges. The DataFrame must have a MultiIndex
                with two levels.

        Returns:
            List[EgdeIndex]: A list of the edge indices that were added.
        """
        if isinstance(edges, pd.DataFrame):
            return self.add_edges_pandas(edges)

        return self._medrecord.add_edges(edges)

    def add_edges_pandas(self, edges: pd.DataFrame) -> List[EdgeIndex]:
        """
        Adds edges to the MedRecord instance from a pandas DataFrame.

        This method takes a pandas DataFrame representing edges, converts it to a
        polars DataFrame, adds the edges to the MedRecord instance using the
        add_edges_polars method and returns a list of edge indices that werde added.
        The DataFrame must have a MultiIndex with two levels, representing the source
        and target nodes.

        Args:
            edges (pd.DataFrame): A DataFrame representing edges. The DataFrame must
                have a MultiIndex with two levels.

        Returns:
            List[EdgeIndex]: A list of the edge indices that were added.
        """
        assert isinstance(
            edges.index, pd.MultiIndex
        ), "Edges dataframe must have a MultiIndex"

        edges_index_names = edges.index.names
        assert (
            len(edges_index_names) == 2
        ), "Edges dataframe MultiIndex must have 2 levels"

        edges_from_index_column = edges_index_names[0]
        edges_to_index_column = edges_index_names[1]

        edges_polars = pl.from_pandas(edges, include_index=True)

        return self.add_edges_polars(
            edges_polars, edges_from_index_column, edges_to_index_column
        )

    def add_edges_polars(
        self,
        edges: pl.DataFrame,
        edges_from_index_column: str,
        edges_to_index_column: str,
    ) -> List[EdgeIndex]:
        """
        Adds edges to the MedRecord instance from a polars DataFrame.

        This method takes a polars DataFrame representing edges and two strings
        representing the index columns for the source and target nodes, adds
        the edges to the MedRecord instance and returns a list of edge indices that were
        added.

        Args:
            edges (pl.DataFrame): A DataFrame representing edges.
            edges_from_index_column (str): The index column for the source nodes
                in the edges DataFrame.
            edges_to_index_column (str): The index column for the target nodes
                in the edges DataFrame.

        Returns:
            List[EdgeIndex]: A list of the edge indices that were added.
        """
        return self._medrecord.add_edges_dataframe(
            edges, edges_from_index_column, edges_to_index_column
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
