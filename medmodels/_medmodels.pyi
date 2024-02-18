from typing import List, Optional, Dict

class PyMedRecord:
    nodes: List[str]
    edges: List[tuple[str, str]]
    groups: List[str]

    def __init__(self) -> None: ...
    @staticmethod
    def from_nodes_and_edges(
        nodes: List[tuple[str, Dict]], edges: List[tuple[str, str, Dict]]
    ) -> PyMedRecord: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def group_count(self) -> int: ...
    def node(self, *node_id: str) -> List[tuple[str, Dict]]: ...
    def edges_between(self, start_node_id: str, end_node_id: str) -> List[Dict]: ...
    def group(self, *group: str) -> List[tuple[str, Dict]]: ...
    def add_nodes(self, nodes: List[tuple[str, Dict]]) -> None: ...
    def add_edges(self, edges: List[tuple[str, str, Dict]]) -> None: ...
    def add_group(self, group: str, node_ids_to_add: Optional[List[str]]) -> None: ...
    def remove_group(self, group: str) -> None: ...
    def remove_from_group(self, group: str, node_id: str) -> None: ...
    def add_to_group(self, group: str, node_id: str) -> None: ...
    def neighbors(self, *node_id: str) -> List[tuple[str, Dict]]: ...