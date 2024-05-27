from typing import List

from medmodels import MedRecord
from medmodels.medrecord.querying import NodeOperation, node
from medmodels.medrecord.types import EdgeIndex, NodeIndex


def find_all_edges(
    medrecord: MedRecord, first_node_index: NodeIndex, second_node_index: NodeIndex
) -> List[EdgeIndex]:
    """Finds all edges connecting two nodes in the MedRecord object.

    Args:
        medrecord (MedRecord): An instance of the MedRecord class containing patient
            medical data.
        first_node_index (NodeIndex): The index of the first node.
        second_node_index (NodeIndex): The index of the second node.

    Returns:
        List[EdgeIndex]: A list of edge indices connecting the two nodes.
    """
    return medrecord.edges_connecting(
        first_node_index, second_node_index
    ) + medrecord.edges_connecting(second_node_index, first_node_index)


def all_neighbors(
    medrecord: MedRecord,
    index: NodeIndex,
) -> NodeOperation:
    """Finds all neighbors of a node in the MedRecord object.

    Args:
        medrecord (MedRecord): An instance of the MedRecord class containing patient
            medical data.
        index (NodeIndex): The index of the first node.

    Returns:
        NodeOperation: A node operation representing all neighbors of the node.
    """
    return node().index().is_in(medrecord.neighbors(index)) | node().has_neighbor_with(
        node().index() == index
    )
