"""Module for analyzing concepts and their distribution."""

import operator
from typing import Dict, List, Tuple

from tqdm import tqdm

from medmodels.medrecord import MedRecord
from medmodels.medrecord.querying import EdgeOperand
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

    def test_one(edge: EdgeOperand):
        edge.source_node().in_group(cohort)
        edge.target_node().index().equal_to(concept_node)

    def test_two(edge: EdgeOperand):
        edge.target_node().in_group(cohort)
        edge.source_node().index().equal_to(concept_node)

    for concept_node in tqdm(concept_nodes, desc="Getting concept counts"):
        concept_counts[concept_node] = len(
            medrecord.select_edges(lambda edge: edge.either_or(test_one, test_two))
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
