from typing import Dict, List, Optional, Union

import pandas as pd
import polars as pl

from medmodels._medmodels import PyMedRecord

MedRecordAttribute = Union[str, int]
MedRecordValue = Union[str, int, float, bool]
NodeIndex = MedRecordAttribute
EdgeIndex = int
Group = MedRecordAttribute
Attributes = Dict[MedRecordAttribute, MedRecordValue]


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

    @property
    def nodes(self) -> List[NodeIndex]:
        """
        Returns the node indices in the MedRecord instance.

        Returns:
            List[NodeIndex]: The node indices in the MedRecord instance.
        """
        return self._medrecord.nodes

    def node(self, *node_index: NodeIndex) -> Dict[NodeIndex, Attributes]:
        """
        Returns the nodes with the specified indices in the MedRecord instance.

        This method takes one or more node indices as arguments and returns the
        attributes of these nodes.

        Args:
            *node_index (NodeIndex): One or more node indices.

        Returns:
            Dict[NodeIndex, Attributes]: A dictionary of the node indices and
                their attributes.
        """
        return self._medrecord.node(*node_index)

    def edge(self, *edge_index: EdgeIndex) -> Dict[EdgeIndex, Attributes]:
        """
        Returns the edges with the specified indices in the MedRecord instance.

        This method takes one or more edge indices as arguments and returns the
        attributes of these edges.

        Args:
            *edge_index (EdgeIndex): One or more edge indices.

        Returns:
            Dict[EdgeIndex, Attributes]: A dictionary of the edge indices and
                their attributes.
        """
        return self._medrecord.edge(*edge_index)

    @property
    def edges(self) -> List[EdgeIndex]:
        """
        Returns the edge indices in the MedRecord instance.

        Returns:
            List[EdgeIndex]: The edge indices in the MedRecord instance.
        """
        return self._medrecord.edges

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

    @property
    def groups(self) -> List[Group]:
        """
        Returns the groups in the MedRecord instance.

        Returns:
            List[Group]: The groups in the MedRecord instance.
        """
        return self._medrecord.groups

    def group(self, *group: Group) -> Dict[Group, List[NodeIndex]]:
        """
        Returns the node indices in the specified group(s) in the MedRecord instance.

        This method takes one or more groups as arguments and returns a dictionary of
        the group names and the node indices in each group.

        Args:
            *group (Group): One or more group names to get the nodes from.

        Returns:
            Dict[Group, List[NodeIndex]]: A dictionary of the group names and the node
                indices in each group.
        """
        return self._medrecord.group(*group)

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

    def remove_group(self, group: Group) -> None:
        """
        Removes a group from the MedRecord instance.

        This method takes a group name and removes the group from the
        MedRecord instance.

        Args:
            group (Group): The name of the group to remove.

        Returns:
            None
        """
        return self._medrecord.remove_group(group)

    def remove_from_group(self, group: Group, node_index: NodeIndex) -> None:
        """
        Removes a node from a group in the MedRecord instance.

        This method takes a group name and a node index, and removes the node from the
        specified group in the MedRecord instance.

        Args:
            group (str): The name of the group from which to remove the node.
            node_index (NodeIndex): The index of the node to remove from the group.

        Returns:
            None
        """
        return self._medrecord.remove_from_group(group, node_index)

    def add_to_group(self, group: Group, *node_index: NodeIndex) -> None:
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
        return self._medrecord.add_to_group(group, *node_index)

    def neighbors(self, *node_index: NodeIndex) -> Dict[NodeIndex, List[NodeIndex]]:
        """
        Retrieves the neighbors of a node in the MedRecord instance.

        This method takes one or more node indices and retrieves the neighbors of the
        specified nodes in the MedRecord instance.  The return type is a dictionary of
        the node's index and the node indices of the neighboring nodes.

        Args:
            *node_index (NodeIndex): The index/indices of the node(s) for which to
            retrieve neighbors.

        Returns:
            Dict[NodeIndex, List[NodeIndex]]: A dictionary of the node's index and the
                node indices of the neighboring nodes.
        """
        return self._medrecord.neighbors(*node_index)
