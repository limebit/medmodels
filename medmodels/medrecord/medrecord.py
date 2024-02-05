from medmodels._medmodels import PyMedRecord
from typing import List, Optional, Dict


class MedRecord:
    _medrecord: PyMedRecord

    def __init__(self) -> None:
        self._medrecord = PyMedRecord()

    @classmethod
    def from_nodes_and_edges(
        cls, nodes: List[tuple[str, Dict]], edges: List[tuple[str, str, Dict]]
    ) -> "MedRecord":
        medrecord = cls.__new__(cls)
        medrecord._medrecord = PyMedRecord.from_nodes_and_edges(nodes, edges)

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
        return self._medrecord.node(node_id)

    @property
    def edges(self) -> List[str]:
        return self._medrecord.edges

    def edges_between(self, start_node_id: str, end_node_id: str) -> List[Dict]:
        return self._medrecord.edges_between(start_node_id, end_node_id)

    @property
    def groups(self) -> List[str]:
        return self._medrecord.groups

    def group(self, *group: str) -> List[tuple[str, Dict]]:
        return self._medrecord.group(group)

    def add_nodes(self, nodes: List[tuple[str, Dict]]) -> None:
        return self._medrecord.add_nodes(nodes)

    def add_edges(self, edges: List[tuple[str, str, Dict]]) -> None:
        return self._medrecord.add_edges(edges)

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
