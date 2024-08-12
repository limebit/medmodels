from typing import Literal

import pandas as pd

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import EdgeIndex, Group, MedRecordAttribute, NodeIndex


def find_reference_edge(
    medrecord: MedRecord,
    node_index: NodeIndex,
    connected_group: Group,
    reference: Literal["first", "last"],
    time_attribute: MedRecordAttribute = "time",
) -> EdgeIndex:
    """Determines the reference edge that represents the first or last exposure of a node index to any node in the connecte_group (list of nodes).

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
        ValueError: If no edge with a time attribute is found for the node, indicating
            an issue with the data or the connection to the specified nodes.
        ValueError: If no edge is found for the specified node with the nodes in that
            group in the MedRecord.

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
    if reference == "first":
        reference_time = pd.Timestamp.max
    elif reference == "last":
        reference_time = pd.Timestamp.min

    nodes_in_group = medrecord.nodes_in_group(connected_group)
    reference_edge = None

    for node_from_group in nodes_in_group:
        edges = medrecord.edges_connecting(node_index, node_from_group, directed=False)

        # If the node does not have an edge to that node, continue
        if not edges:
            continue

        # If the node has an edge, check if it has the time attribute
        edge_values = medrecord.edge[edges].values()

        if not all(time_attribute in edge_attribute for edge_attribute in edge_values):
            raise ValueError("Time attribute not found in the edge attributes")

        for edge in edges:
            edge_time = pd.to_datetime(str(medrecord.edge[edge][time_attribute]))

            if (reference == "first" and edge_time < reference_time) or (
                reference == "last" and edge_time > reference_time
            ):
                reference_edge = edge
                reference_time = edge_time

    if reference_edge is None:
        raise ValueError(f"No edge found for node {node_index} in this MedRecord")

    return reference_edge


def find_node_in_time_window(
    medrecord: MedRecord,
    subject_index: NodeIndex,
    event_node: NodeIndex,
    connected_group: Group,
    start_days: int,
    end_days: int,
    reference: Literal["first", "last"],
    time_attribute: MedRecordAttribute = "time",
) -> bool:
    """Determines whether an event occurred within a specified time window for a given subject node.

    This method helps in identifying events that are temporally related to a reference
    event by considering the temporal sequence of events.

    Args:
        medrecord (MedRecord): An instance of the MedRecord class containing medical
            data.
        subject_index (NodeIndex): The subject node to evaluate.
        event_node (NodeIndex): The event node to check for its occurrence within the
            time window.
        connected_group (Group): The group of nodes that are connected to the subject
            node.
        start_days (int): The start of the time window in days relative to the
            reference event.
        end_days (int): The end of the time window in days relative to the
            reference event.
        reference (Literal["first", "last"]): The reference point for the time
            window.
        time_attribute (MedRecordAttribute, optional): The attribute in the edge that
            contains the time information. Defaults to "time".

    Returns:
        bool: True if the event occurred within the specified time window;
            False otherwise.

    Raises:
        ValueError: If the time attribute is not found in the edge attributes.

    Examples:
        This function checks if the event "E1" occurred within a time window of 30 days
        before and after the last exposure to any medication in the group "medications" for
        the subject "P1":

        .. code-block:: python
            :linenos:

            find_node_in_time_window(
                medrecord,
                subject_index="P1",
                event_node="E1",
                connected_group="medications",
                start_days=-30,
                end_days=30,
                reference="last",
            )

        .. code-block:: python

            >>> True
    """
    reference_edge = find_reference_edge(
        medrecord,
        subject_index,
        connected_group,
        reference=reference,
        time_attribute=time_attribute,
    )
    reference_time = pd.to_datetime(str(medrecord.edge[reference_edge][time_attribute]))

    start_period = pd.Timedelta(days=start_days)
    end_period = pd.Timedelta(days=end_days)
    edges = medrecord.edges_connecting(subject_index, event_node, directed=False)

    for edge in edges:
        edge_attributes = medrecord.edge[edge]
        if time_attribute not in edge_attributes:
            raise ValueError("Time attribute not found in the edge attributes")

        event_time = pd.to_datetime(str(edge_attributes[time_attribute]))
        time_difference = event_time - reference_time

        # Return True if the event happened within the specified time window
        if start_period <= time_difference <= end_period:
            return True

    # Return False if no event happened within the time window
    return False
