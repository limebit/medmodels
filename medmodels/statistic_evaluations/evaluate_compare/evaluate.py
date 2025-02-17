from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from medmodels.medrecord.medrecord import EdgesDirected, MedRecord
from medmodels.medrecord.querying import NodeQuery, EdgeOperand
from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import (
    AttributeSummary,
    Group,
    GroupInputList,
    MedRecordAttribute,
    NodeIndex,
)
from medmodels.statistic_evaluations.statistical_analysis.descriptive_statistics import (
    count_concept_connections,
    extract_top_k_concepts,
)


class CohortEvaluator:
    medrecord: MedRecord
    name: str
    cohort_group: Group
    time_attribute: MedRecordAttribute
    attributes: Optional[Dict[str, MedRecordAttribute]]
    concepts_groups: GroupInputList
    attribute_summary: Dict[Group, AttributeSummary]
    attribute_types: Dict[MedRecordAttribute, AttributeType]
    concepts_counts: Dict[Group, Dict[NodeIndex, int]]

    def __init__(
        self,
        medrecord: MedRecord,
        name: str,
        cohort_group: Union[Group, NodeQuery] = "patients",
        time_attribute: MedRecordAttribute = "time",
        attributes: Optional[Dict[str, MedRecordAttribute]] = None,
        concepts_groups: Optional[GroupInputList] = None,
    ) -> None:
        self.medrecord = medrecord.clone()
        self.name = name
        self.time_attribute = time_attribute
        self.concepts_groups = concepts_groups
        self.attributes = attributes

        # set patient cohort group
        if isinstance(cohort_group, Group):
            self.cohort_group = cohort_group
        else:
            cohort_nodes = self.medrecord.select_nodes(cohort_group)

            if "cohort" not in self.medrecord.groups:
                cohort_name = "cohort"

            else:
                i = 0

                while f"cohort{i}" in self.medrecord.groups:
                    i += 1

                cohort_name = "cohort"

            self.medrecord.add_group(group=cohort_name, nodes=cohort_nodes)

            self.cohort_group = cohort_name

        if not concepts_groups:
            self.concepts_groups = self.get_concepts_groups()

        # determine attribute types
        self.attribute_types = {}

        attribute_dictionary = self.medrecord._describe_group_nodes()[
            self.cohort_group
        ]["attribute"]

        for attribute in attribute_dictionary.keys():
            self.attribute_types[attribute] = attribute_dictionary[attribute]["type"]

        # self.concept_counts = None

    def get_concepts_groups(
        self,
    ) -> GroupInputList:
        """Get concepts groups that have connecting edges to the cohort.

        Returns:
            GroupInputList: List of concept groups.
        """
        if self.concepts_groups:
            return self.concepts_groups

        concepts_groups = []

        cohort_nodes = self.medrecord.group(self.cohort_group)["nodes"]

        for group in self.medrecord.groups:
            group_nodes = self.medrecord.group(group)["nodes"]

            if group == self.cohort_group or len(group_nodes) == 0:
                continue

            self.medrecord.group(group)["nodes"]

            count_group = len(
                self.medrecord.edges_connecting(
                    group_nodes, cohort_nodes, directed=EdgesDirected.UNDIRECTED
                )
            )
            if count_group > 0:
                concepts_groups.append(group)

        return concepts_groups

    def get_concept_counts(self) -> Dict[Group, Dict[NodeIndex, int]]:
        """Get a concept count summary for all concepts.

        Returns:
            Dict[Group, Dict[NodeIndex, int]]: Dictionary with concepts, their node
                indices and counts.
        """
        for concept in self.concepts_groups:
            self.concepts_counts[concept] = count_concept_connections(
                medrecord=self.medrecord, concept=concept, cohort=self.cohort_group
            )

        return self.concepts_counts

    def get_top_k_concepts(
        self, top_k: int, concept: Optional[Group] = None
    ) -> List[NodeIndex]:
        """Get top k entries for a specific concept group or all concepts.

        Args:
            top_k (int): Number of top concepts.
            concept (Optional[Group]): Concept group. Defaults to None.

        Raises:
            ValueError: If concept not in concepts groups.
            ValueError: If less than topk concepts in the concept counts.

        Returns:
            List[NodeIndex]: _description_
        """
        if concept:
            if concept not in self.concepts_counts:
                msg = f"Concept {concept} not in the list of concepts for this cohort."
                raise ValueError(msg)

            concepts_counts = self.concepts_counts[concept]

        else:
            # get the most common for all concepts
            concepts_counts = {
                k: v
                for inner_dict in self.concepts_counts.values()
                for k, v in inner_dict.items()
            }

        return extract_top_k_concepts(concepts_counts, top_k)

    def get_attribute_summary(
        self,
    ) -> Dict[Group, AttributeSummary]:
        ...

        # self.medrecord._describe_group_nodes()
