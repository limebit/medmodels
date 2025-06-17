"""This module contains functions for temporal analysis of treatment effects."""

from typing import Literal

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import EdgeIndicesOperand, EdgeOperand
from medmodels.medrecord.types import EdgeIndex, Group, MedRecordAttribute, NodeIndex


def find_reference_edge(
    medrecord: MedRecord,
    node_index: NodeIndex,
    connected_group: Group,
    reference: Literal["first", "last"],
    time_attribute: MedRecordAttribute = "time",
) -> EdgeIndex:
    """Determines the reference edge containing the first/last last exposure.

    This function returns the edge containing the first/last exposure (depending on
    the `reference` argument) of a node to any other node in the connected_group. This
    method is crucial for analyzing the temporal sequence of treatments and outcomes.

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
        This function returns the edge containing the timestamp of the last exposure to
        any medication in the group "medications" for subject "P1".

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
        """Query for the source node of an edge.

        Query the source node of an edge to have a specific index and the target node to
        be in the specified group.

        Args:
            edge (EdgeOperand): The edge operand.
        """
        edge.source_node().index().equal_to(node_index)
        edge.target_node().in_group(connected_group)

    def query_target_node(edge: EdgeOperand) -> None:
        """Query the target  node of an edge.

        Query the source node of an edge to be in a specified group and the target node
        to have a specified index.

        Args:
            edge (EdgeOperand): The edge operand.
        """
        edge.source_node().in_group(connected_group)
        edge.target_node().index().equal_to(node_index)

    def query(edge: EdgeOperand) -> EdgeIndicesOperand:
        """Query the reference edge.

        Query the edge that:
            - Connects the specified group to the specified node.
            - Has the minimum or maximum time attribute.

        Args:
            edge (EdgeOperand): The edge operand.

        Returns:
            EdgeIndicesOperand: The edge indices of the queried edge.
        """
        edge.either_or(query_source_node, query_target_node)
        edge.attribute(time_attribute).is_datetime()

        if reference == "first":
            edge.attribute(time_attribute).is_min()
        elif reference == "last":
            edge.attribute(time_attribute).is_max()

        return edge.index()

    error_message = (
        f"No edge with that time attribute or with a datetime data type was found for node "
        f"{node_index} in this MedRecord"
    )

    reference_edge = medrecord.query_edges(query)

    if not len(reference_edge) > 0:
        raise ValueError(error_message)

    return reference_edge[0]
