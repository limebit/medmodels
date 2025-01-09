from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import NodeQuery, EdgeOperand
from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import (
    AttributeSummary,
    Group,
    GroupInputList,
    MedRecordAttribute,
    NodeIndex,
)


class CohortEvaluator:
    medrecord: MedRecord
    name: str
    cohort_group: Group
    time_attribute: MedRecordAttribute
    attributes: Optional[Dict[str, MedRecordAttribute]]
    concepts_groups: Optional[GroupInputList]
    attribute_summary: Dict[Group, AttributeSummary]
    attribute_types: Dict[MedRecordAttribute, AttributeType]
    concept_counts: Dict[Group, Dict[NodeIndex, int]]

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

        # determine attribute types
        self.attribute_types = {}

        attribute_dictionary = self.medrecord._describe_group_nodes()[
            self.cohort_group
        ]["attribute"]

        for attribute in attribute_dictionary.keys():
            self.attribute_types[attribute] = attribute_dictionary[attribute]["type"]

        # self.concept_counts = None

    def get_concept_counts(
        self,
    ) -> Dict[Group, Dict[NodeIndex, int]]:
        self.concept_counts = {}
        cohort_nodes = self.medrecord.group(self.cohort_group)["nodes"]

        # use all groups that are connected to the cohort group
        if not self.concepts_groups:
            self.concepts_groups = []

            for group in self.medrecord.groups:
                group_nodes = self.medrecord.group(group)["nodes"]

                if group == self.cohort_group or len(group_nodes) == 0:
                    continue

                self.medrecord.group(group)["nodes"]

                count_group = len(
                    self.medrecord.edges_connecting(
                        group_nodes, cohort_nodes, directed=False
                    )
                )
                if count_group > 0:
                    self.concepts_groups.append(group)

        for concept in self.concepts_groups:
            self.concept_counts[concept] = {}

            concept_nodes = self.medrecord.group(concept)["nodes"]

            for concept_node in concept_nodes:
                self.concept_counts[concept][concept_node] = len(
                    self.medrecord.edges_connecting(
                        concept_node, cohort_nodes, directed=False
                    )
                )

        return self.concept_counts

    def get_top_k_concepts(self, top_k: int, concept: Group) -> List[NodeIndex]:
        if not self.concept_counts:
            self.get_concept_counts()

        if concept:
            if concept not in self.concept_counts.keys():
                msg = f"Concept {concept} not in the list of concepts for this cohort."
                raise ValueError(msg)
            else:
                concept_counts = self.concept_counts[concept]

        else:
            # get the most common for all concepts
            concept_counts = {
                k: v
                for inner_dict in self.concept_counts.values()
                for k, v in inner_dict.items()
            }

        sorted_concepts = sorted(
            concept_counts.keys(), key=lambda item: item[1], reverse=True
        )

        # how to cut off top k?

        return sorted_concepts[:top_k]

    def get_attribute_summary(
        self,
    ) -> Dict[Group, AttributeSummary]:
        ...

        # self.medrecord._describe_group_nodes()
