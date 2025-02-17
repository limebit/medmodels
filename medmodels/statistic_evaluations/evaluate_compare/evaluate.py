"""Cohort Evaluator for analyzing attribuite statistics and connected concepts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

# from medmodels.medrecord.medrecord import EdgesDirected, MedRecord
from medmodels.medrecord.querying import NodeQuery
from medmodels.statistic_evaluations.statistical_analysis.attribute_analysis import (
    AttributeStatistics,
    extract_attribute_summary,
)
from medmodels.statistic_evaluations.statistical_analysis.concepts_analysis import (
    count_concept_connections,
    extract_top_k_concepts,
)

if TYPE_CHECKING:
    from medmodels.medrecord.medrecord import MedRecord
    from medmodels.medrecord.querying import NodeOperand, NodeQuery
    from medmodels.medrecord.types import (
        Group,
        GroupInputList,
        MedRecordAttribute,
        NodeIndex,
    )


class CohortEvaluator:
    """Class for evaluating a cohort. Needed as Input for the comparer class."""

    medrecord: MedRecord
    name: str
    patient_group: Group
    cohort_group: Group
    time_attribute: MedRecordAttribute
    attributes: Optional[Dict[str, MedRecordAttribute]]
    concepts_groups: GroupInputList
    attribute_summary: Dict[MedRecordAttribute, AttributeStatistics]
    concepts_counts: Dict[Group, Dict[NodeIndex, int]]

    def __init__(
        self,
        medrecord: MedRecord,
        name: str,
        patient_group: Group = "patients",
        cohort_query: Optional[NodeQuery] = None,
        time_attribute: MedRecordAttribute = "time",
        attributes: Optional[Dict[str, MedRecordAttribute]] = None,
        concepts_groups: Optional[GroupInputList] = None,
    ) -> None:
        """Initializes the Evaluator class.

        Args:
            medrecord (MedRecord): MedRecord containing a cohort that needs evaluation.
            name (str): Name of the cohort.
            patient_group (Group): Group of patients in the MedRecord.
            cohort_query (Optional[NodeQuery], optional): NodeQuery limiting the patient
                group to a cohort group. If no cohort query is given, all patients will
                be in the cohort. Defaults to "patients".
            time_attribute (MedRecordAttribute, optional): Name of the time attribute
                used in the MedRecord. Defaults to "time".
            attributes (Optional[Dict[str, MedRecordAttribute]], optional): Mapping of
                names of attributes and their naming in the MedRecord. Defaults to None.
            concepts_groups (Optional[GroupInputList], optional): List of concepts to
                evaluate. If none are given, it will select all Groups that have
                connecting edges to the cohort group. Defaults to None.
        """
        self.medrecord = medrecord.clone()
        self.name = name
        self.time_attribute = time_attribute
        self.attributes = attributes

        # set patient cohort group
        self.patient_group = patient_group
        if not cohort_query:
            self.cohort_group = self.patient_group
        else:

            def query_cohort(node: NodeOperand) -> None:
                cohort_query(node)
                node.in_group(self.patient_group)

            cohort_nodes = self.medrecord.select_nodes(query_cohort)

            if "cohort" not in self.medrecord.groups:
                cohort_name = "cohort"

            else:
                i = 0

                while f"cohort{i}" in self.medrecord.groups:
                    i += 1

                cohort_name = "cohort"

            self.medrecord.add_group(group=cohort_name, nodes=cohort_nodes)

            self.cohort_group = cohort_name

        if concepts_groups:
            self.concepts_groups = concepts_groups
        else:
            self.concepts_groups = self._get_concepts_groups()

        self.concepts_counts = self._get_concept_counts()

        self.attribute_summary = self._get_attribute_summary()

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

    def _get_concepts_groups(
        self,
    ) -> GroupInputList:
        """Get concepts groups that have connecting edges to the cohort.

        Returns:
            GroupInputList: List of concept groups.
        """
        concepts_groups = []

        patient_nodes = self.medrecord.group(self.patient_group)["nodes"]

        for group in self.medrecord.groups:
            group_nodes = self.medrecord.group(group)["nodes"]

            if group == self.cohort_group or len(group_nodes) == 0:
                continue

            self.medrecord.group(group)["nodes"]

            count_group = len(
                self.medrecord.edges_connecting(
                    patient_nodes,
                    group_nodes,
                    # directed=EdgesDirected.UNDIRECTED
                )
            )
            if count_group > 0:
                concepts_groups.append(group)

        return concepts_groups

    def _get_concept_counts(self) -> Dict[Group, Dict[NodeIndex, int]]:
        """Get a concept count summary for all concepts.

        Returns:
            Dict[Group, Dict[NodeIndex, int]]: Dictionary with concepts, their node
                indices and counts.
        """
        concepts_counts = {}

        for concept in self.concepts_groups:
            concepts_counts[concept] = count_concept_connections(
                medrecord=self.medrecord, concept=concept, cohort=self.cohort_group
            )

        return concepts_counts

    def _get_attribute_summary(
        self,
    ) -> Dict[MedRecordAttribute, AttributeStatistics]:
        """Describe the attributes of the cohort.

        Returns:
            Dict[str, AttributeStatistics]:

        Raises:
            ValueError: If not all attributes are found for the cohort.
        """
        schema = (
            self.medrecord.schema.group(self.patient_group).nodes
            if self.patient_group in self.medrecord.schema.groups
            else None
        )
        all_attribute_summary = extract_attribute_summary(
            self.medrecord.node[self.medrecord.group(self.cohort_group)["nodes"]],
            schema=schema,
            summary_type="extended",
        )

        if not self.attributes:
            return all_attribute_summary

        # check if any attributes are not actually found in the cohort group
        missing_attributes = set(self.attributes.values()) - set(
            all_attribute_summary.keys()
        )

        if len(missing_attributes) > 0:
            msg = f"""{"Attribute " if len(missing_attributes) == 1 else "Attributes "}
                    {", ".join([str(attr) for attr in missing_attributes])} not found for the
                    cohort."""
            raise ValueError(msg)

        attribute_summary = {}

        for attribute_name, attribute in self.attributes.items():
            attribute_summary[attribute_name] = all_attribute_summary[attribute]

        return attribute_summary
