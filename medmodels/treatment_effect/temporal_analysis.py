from typing import Literal

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import EdgeOperand
from medmodels.medrecord.types import EdgeIndex, Group, MedRecordAttribute, NodeIndex


def find_reference_edge(
    medrecord: MedRecord,
    node_index: NodeIndex,
    connected_group: Group,
    reference: Literal["first", "last"],
    time_attribute: MedRecordAttribute = "time",
) -> EdgeIndex:
    """Determines the reference edge that represents the first or last exposure of a node index to any node in the connected_group (list of nodes).

    This method is crucial for analyzing the temporal sequence of treatments and outcomes.

    This function iterates over all nodes in the gorup and finds the reference edge
    among them (first or last), ensuring that the analysis considers the reference
    exposure.

    Args:
        medrecord (MedRecord): An instance of the MedRecord class containing medical
            data.
        node_index (NodeIndex): The node for which to determine the reference
            exposure edge.
        connected_group (Group): The group of nodes that are connected to the node.
        reference (Literal["first", "last"]): The reference point for the exposure time.
            Options include "first" and "last". If "first", the function returns the
            earliest exposure edge. If "last", the function returns the latest
            exposure edge.
        time_attribute (MedRecordAttribute, optional): The attribute in the edge that
            contains the time information. Defaults to "time".

    Returns:
        EdgeIndex: The edge index of the reference exposure.

    Raises:
        ValueError: If no edge with that time attribute or datetime datatype found for
            the node in this MedRecord.

    Example:
        This function returns the edge containing the timestamp of the last exposure to any
        medication in the group "medications" for subject "P1".

        .. highlight:: python
        .. code-block:: python

            find_reference_edge(
                medrecord,
                node_index="P1",
                connected_group="medications",
                reference="last",
            )

    """

    def query_source_node(edge: EdgeOperand) -> None:
        """Query the source node of an edge to have a specific index and the target node to be in a specified group.

        Args:
            edge (EdgeOperand): The edge operand.
        """
        edge.source_node().index().equal_to(node_index)
        edge.target_node().in_group(connected_group)

    def query_target_node(edge: EdgeOperand) -> None:
        """Query the source node of an edge to be in a specified group and the target node to have a specific index.

        Args:
            edge (EdgeOperand): The edge operand.
        """
        edge.source_node().in_group(connected_group)
        edge.target_node().index().equal_to(node_index)

    def query(edge: EdgeOperand) -> None:
        """Query the edge that connects the group to the node and has the minimum or maximum time attribute.

        Args:
            edge (EdgeOperand): The edge operand.
        """
        edge.either_or(query_source_node, query_target_node)
        edge.attribute(time_attribute).is_datetime()

        if reference == "first":
            edge.attribute(time_attribute).is_min()
        elif reference == "last":
            edge.attribute(time_attribute).is_max()

    try:
        reference_edge = medrecord.select_edges(query)

    except RuntimeError:
        msg = (
            f"No edge with that time attribute or with a datetime data type was found for node "
            f"{node_index} in this MedRecord"
        )
        raise ValueError(msg)

    return reference_edge[0]
