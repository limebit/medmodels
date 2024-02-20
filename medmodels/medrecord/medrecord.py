from medmodels._medmodels import PyMedRecord
from typing import List, Optional, Dict, Union
import polars as pl
import pandas as pd


class MedRecord:
    _medrecord: PyMedRecord

    def __init__(self) -> None:
        self._medrecord = PyMedRecord()

    @classmethod
    def from_tuples(
        cls,
        nodes: List[tuple[str, Dict]],
        edges: Optional[List[tuple[str, str, Dict]]] = [],
    ) -> "MedRecord":
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_tuples(nodes, edges)

        return medrecord

    @classmethod
    def from_pandas(
        cls, nodes: pd.DataFrame, edges: Optional[pd.DataFrame] = None
    ) -> "MedRecord":
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
        return self._medrecord.node_count()

    def edge_count(self) -> int:
        return self._medrecord.edge_count()

    def group_count(self) -> int:
        return self._medrecord.group_count()

    @property
    def nodes(self) -> List[str]:
        return self._medrecord.nodes

    def node(self, *node_id: str) -> List[tuple[str, Dict]]:
        return self._medrecord.node(*node_id)

    @property
    def edges(self) -> List[str]:
        return self._medrecord.edges

    def edges_between(self, start_node_id: str, end_node_id: str) -> List[Dict]:
        return self._medrecord.edges_between(start_node_id, end_node_id)

    @property
    def groups(self) -> List[str]:
        return self._medrecord.groups

    def group(self, *group: str) -> List[tuple[str, Dict]]:
        return self._medrecord.group(*group)

    def add_nodes(self, nodes: Union[List[tuple[str, Dict]], pd.DataFrame]) -> None:
        if isinstance(nodes, pd.DataFrame):
            return self.add_nodes_dataframe(nodes)

        return self._medrecord.add_nodes(nodes)

    def add_nodes_pandas(self, nodes: pd.DataFrame) -> None:
        assert isinstance(nodes.index, pd.Index), "Nodes dataframe must have an Index"

        nodes_index_column = nodes.index.name

        assert nodes_index_column is not None, "Nodes dataframe must have an Index"

        nodes_polars = pl.from_pandas(nodes, include_index=True)

        return self.add_nodes_polars(nodes_polars, nodes_index_column)

    def add_nodes_polars(self, nodes: pl.DataFrame, nodes_index_column: str) -> None:
        return self._medrecord.add_nodes_dataframe(nodes, nodes_index_column)

    def add_edges(
        self, edges: Union[List[tuple[str, str, Dict]], pd.DataFrame]
    ) -> None:
        if isinstance(edges, pd.DataFrame):
            return self.add_edges_dataframe(edges)

        return self._medrecord.add_edges(edges)

    def add_edges_pandas(self, edges: pd.DataFrame) -> None:
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
    ) -> None:
        return self._medrecord.add_edges_dataframe(
            edges, edges_from_index_column, edges_to_index_column
        )

    def add_group(self, group: str, node_ids_to_add: Optional[List[str]]) -> None:
        return self._medrecord.add_group(group, node_ids_to_add)

    def remove_group(self, group: str) -> None:
        return self._medrecord.remove_group(group)

    def remove_from_group(self, group: str, node_id: str) -> None:
        return self._medrecord.remove_from_group(group, node_id)

    def add_to_group(self, group: str, node_id: str) -> None:
        return self._medrecord.add_to_group(group, node_id)

    def neighbors(self, *node_id: str) -> List[tuple[str, Dict]]:
        return self._medrecord.neighbors(node_id)
