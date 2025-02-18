"""Module for analyzing concepts and their distribution."""

import operator
from typing import Dict, List, Tuple

from medmodels.medrecord import MedRecord
from medmodels.medrecord.medrecord import EdgesDirected
from medmodels.medrecord.types import Group, NodeIndex


def count_concept_connections(
    medrecord: MedRecord, concept: Group, cohort: Group
) -> Dict[NodeIndex, int]:
    """Count for each concept node how often it is connected to the cohort group.

    Args:
        medrecord (MedRecord): Medrecord containing concept and cohort groups and with
            nodes and also edges connecting these to groups.
        concept (Group): Concept, for example 'Diagnoses', 'Procedure' or 'Medication'.
        cohort (Group): Cohort group, for example 'Patients'.

    Returns:
        Dict[NodeIndex, int]: Dictionary with the node index for the concept and the
            count for the connecting edges.
    """
    concept_counts = {}

    concept_nodes = medrecord.group(concept)["nodes"]

    for concept_node in concept_nodes:
        concept_counts[concept_node] = len(
            medrecord.edges_connecting(
                concept_node,
                medrecord.group(cohort)["nodes"],
                directed=EdgesDirected.UNDIRECTED,
            )
        )

    return concept_counts


def extract_top_k_concepts(
    concept_counts: Dict[NodeIndex, int], top_k: int
) -> List[Tuple[NodeIndex, int]]:
    """Extract the topk concepts from a concept count dictionary.

    Args:
        concept_counts (Dict[NodeIndex, int]): Dictionary of concept node indeces and
            their counts.
        top_k (int): Number of top concepts to extract.

    Raises:
        ValueError: If less than topk concepts in the concept counts

    Returns:
        List[Tuple[NodeIndex, int]]: List of top k concepts and their count.
    """
    sorted_concepts = sorted(
        concept_counts.items(), key=operator.itemgetter(1), reverse=True
    )

    if top_k > len(sorted_concepts):
        msg = f"Less than {top_k} concept connections, can not extract top {top_k}"
        raise ValueError(msg)

    return sorted_concepts[:top_k]
