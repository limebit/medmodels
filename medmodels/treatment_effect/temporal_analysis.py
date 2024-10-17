import datetime
from typing import Literal

import pandas as pd

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
            f"No edge with that time attribute or datetime datatype found for node "
            f"{node_index} in this MedRecord"
        )
        raise ValueError(msg)

    return reference_edge[0]


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
    reference_time = medrecord.edge[reference_edge][time_attribute]

    def query_source_node(edge: EdgeOperand) -> None:
        """Query the source node of an edge to have a specific index and the target node to be in a specified group.

        Args:
            edge (EdgeOperand): The edge operand.
        """
        edge.source_node().index().equal_to(subject_index)
        edge.target_node().index().equal_to(event_node)

    def query_target_node(edge: EdgeOperand) -> None:
        """Query the source node of an edge to be in a specified group and the target node to have a specific index.

        Args:
            edge (EdgeOperand): The edge operand.
        """
        edge.source_node().index().equal_to(event_node)
        edge.target_node().index().equal_to(subject_index)

    def query(edge: EdgeOperand) -> None:
        """Query the edge that connects the group to the node and has the minimum or maximum time attribute.

        Args:
            edge (EdgeOperand): The edge operand.
        """
        edge.either_or(query_source_node, query_target_node)
        edge.attribute(time_attribute).is_datetime()

        edge.attribute(time_attribute).greater_than_or_equal_to(start_time)
        edge.attribute(time_attribute).less_than_or_equal_to(end_time)

    if not isinstance(reference_time, datetime.datetime):
        msg = f"Reference time is not a datetime object: {reference_time}"
        raise ValueError(msg)

    start_time = reference_time + pd.Timedelta(days=start_days)
    end_time = reference_time + pd.Timedelta(days=end_days)

    return bool(medrecord.select_edges(query))
