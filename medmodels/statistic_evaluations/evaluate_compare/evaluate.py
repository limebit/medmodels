"""Cohort Evaluator for analyzing attribuite statistics and connected concepts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from medmodels.medrecord.medrecord import EdgesDirected
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
        MedRecordAttribute,
        NodeIndex,
    )


class CohortEvaluator:
    """Class for evaluating a cohort. Needed as Input for the comparer class.

    medrecord (MedRecord): MedRecord with the cohort and added concept cohort groups.
    name (str): Name of the cohort.
    patient_group (Group): Group of patients in the MedRecord.
    time_attribute (MedRecordAttribute): Name of the time attribute.
    attributes ([Dict[MedRecordAttribute, MedRecordAttribute]]): Mapping of the
        attributes with {evaluator_name: old_name}.
    concepts_groups (Dict[Group, Group]): Mapping of the concepts with  {evaluator_name:
        old_name}.
    concepts_edges (Dict[Group, Group]): Concept names and the group eith the connecting
        edges to the patients.
    attribute_summary (Dict[Group, Dict[MedRecordAttribute, AttributeStatistics]]):
        Dictionary with patients/concepts and their relevant cohort attributes with
        extended descriptive statistics.
    concepts_counts (Dict[Group, Dict[NodeIndex, int]]): All concept counts sorted by
        concepts.
    """

    medrecord: MedRecord
    name: str
    patient_group: Group
    time_attribute: MedRecordAttribute
    attributes: Dict[MedRecordAttribute, MedRecordAttribute]
    concepts_groups: Dict[Group, Group]
    concepts_edges: Dict[Group, Group]
    attribute_summary: Dict[Group, Dict[MedRecordAttribute, AttributeStatistics]]
    concepts_counts: Dict[Group, Dict[NodeIndex, int]]

    def __init__(
        self,
        medrecord: MedRecord,
        name: str,
        patient_group: Group = "patient",
        cohort_query: Optional[NodeQuery] = None,
        time_attribute: MedRecordAttribute = "time",
        attributes: Optional[Dict[MedRecordAttribute, MedRecordAttribute]] = None,
        concepts_groups: Optional[Dict[Group, Group]] = None,
    ) -> None:
        """Initializes the Evaluator class.

        Args:
            medrecord (MedRecord): MedRecord containing a cohort that needs evaluation.
            name (str): Name of the cohort.
            patient_group (Group): Group of patients in the MedRecord. Defaults to
                "patient".
            cohort_query (Optional[NodeQuery], optional): NodeQuery limiting the patient
                group to a cohort group. If no cohort query is given, all patients will
                be in the cohort. Defaults to None.
            time_attribute (MedRecordAttribute, optional): Name of the time attribute
                used in the MedRecord. Defaults to "time".
            attributes (Optional[Dict[MedRecordAttribute, MedRecordAttribute]], optional):
                Mapping of names of attributes and their naming in the MedRecord.
                Defaults to None.
            concepts_groups (Optional[Dict[Group, Group]], optional): Mapping of the
                concepts to evaluate. If none are given, it will select all Groups that
                have connecting edges to the patient group. Defaults to None.
        """  # noqa: W505
        self.medrecord = medrecord.clone()
        self.name = name
        self.time_attribute = time_attribute
        self.patient_group = patient_group

        self.concepts_groups, self.concepts_edges = self._add_patient_concept_edges(
            patient_group, concepts_groups
        )

        # determine cohort subgroup from patients
        if cohort_query:
            # remove all other patients from the MedRecord
            def query_patients_not_in_cohort(node: NodeOperand) -> None:
                node.in_group(self.patient_group)
                node.exclude(cohort_query)

            self.medrecord.remove_nodes(query_patients_not_in_cohort)

        self.concepts_counts = self._get_concept_counts()

        self.attribute_summary = self._get_attribute_summary(attributes=attributes)

        # save the mapping or all cohort attributes
        if attributes:
            self.attributes = attributes

        # save all attributes if none are given
        else:
            all_attributes = []

            for dictionary in self.attribute_summary.values():
                all_attributes.extend(key for key in dictionary)

            self.attributes = {
                attribute: attribute for attribute in set(all_attributes)
            }

    def get_top_k_concepts(
        self, top_k: int, concept: Optional[Group] = None
    ) -> List[Tuple[NodeIndex, int]]:
        """Get top k entries for a specific concept group or all concepts.

        Args:
            top_k (int): Number of top concepts.
            concept (Optional[Group]): Concept group. Defaults to None.

        Raises:
            ValueError: If concept not in concepts groups.
            ValueError: If less than topk concepts in the concept counts.

        Returns:
            List[Tuple[NodeIndex, int]]: _description_
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

    def _add_patient_concept_edges(
        self,
        patient_group: Group,
        concepts_groups: Optional[Dict[Group, Group]] = None,
    ) -> Tuple[Dict[Group, Group], Dict[Group, Group]]:
        """Get concepts groups that have connecting edges to the patients.

        Check the given concepts groups or all medrecord groups for connected edges to
        the patients. Save the edges in a new group for easier retrival later. The name
        of the concept edge group is given in the mapping dictionary.

        Args:
            patient_group (Group): Group of patients.
            concepts_groups (Optional[Dict[Group, Group]], optional): Defined list of
                concepts. Defaults to None.

        Returns:
            Dict[Group, Group]: Mapping dict for the concepts and mapping dict from
                concept group name to connected edges group name.
        """
        if not concepts_groups:
            concepts_groups = {
                group: group
                for group in self.medrecord.groups
                if group != patient_group
            }

        concepts_connecting = {}

        patient_nodes = self.medrecord.group(patient_group)["nodes"]

        # save edge indices already in groups
        edges_in_groups = {
            group: self.medrecord.edges_in_group(group)
            for group in self.medrecord.groups
        }

        for group_name, group in concepts_groups.items():
            group_nodes = self.medrecord.group(group)["nodes"]

            group_patient_edges = self.medrecord.edges_connecting(
                patient_nodes, group_nodes, directed=EdgesDirected.UNDIRECTED
            )

            if not group_patient_edges:
                continue

            # check if edge group exists already
            connected_group = [
                group
                for group, edges in edges_in_groups.items()
                if set(group_patient_edges) == set(edges)
            ]

            if connected_group:
                concepts_connecting[group_name] = connected_group[0]

            # add group for easier retrievallater if it doesn't exist
            else:
                self.medrecord.add_group(
                    group=f"{group}-{patient_group}", edges=group_patient_edges
                )
                concepts_connecting[group_name] = f"{group}-{patient_group}"

        return concepts_groups, concepts_connecting

    def _get_concept_counts(self) -> Dict[Group, Dict[NodeIndex, int]]:
        """Get a concept count summary for all concepts.

        Returns:
            Dict[Group, Dict[NodeIndex, int]]: Dictionary with concepts, their node
                indices and counts.
        """
        concepts_counts = {}

        for concept_name, concept in self.concepts_groups.items():
            concepts_counts[concept_name] = count_concept_connections(
                medrecord=self.medrecord,
                concept=concept,
                cohort=self.patient_group,
            )

        return concepts_counts

    def _get_attribute_summary(
        self,
        attributes: Optional[Dict[MedRecordAttribute, MedRecordAttribute]],
    ) -> Dict[Group, Dict[MedRecordAttribute, AttributeStatistics]]:
        """Describe the attributes of the cohort.

        Args:
            attributes (Optional[Dict[MedRecordAttribute, MedRecordAttribute]]):
                Attribute mapping with a chosen name and their name in the MedRecord.

        Returns:
            Dict[Group, Dict[MedRecordAttribute, AttributeStatistics]]: Dictionary
                sorted by the patient anc concept groups. Each group has a statistics
                summary of their chosen attributes.

        Raises:
            ValueError: If not all attributes are found for the cohort.
        """
        attribute_summary = {}
        all_attribute_summary = {}
        attributes_found = []

        # patient node attribute summary
        schema = (
            self.medrecord.schema.group(self.patient_group).nodes
            if self.patient_group in self.medrecord.schema.groups
            else None
        )

        all_attribute_summary[self.patient_group] = extract_attribute_summary(
            self.medrecord.node[self.medrecord.group(self.patient_group)["nodes"]],
            schema=schema,
            summary_type="extended",
        )

        # selecting and remapping of the attributes
        if attributes:
            attribute_summary[self.patient_group] = {}

            for attribute_name, attribute in attributes.items():
                if attribute in all_attribute_summary[self.patient_group]:
                    attribute_summary[self.patient_group][attribute_name] = (
                        all_attribute_summary[self.patient_group][attribute]
                    )

                    attributes_found.append(attribute_name)

        # concepts edge attribute summary
        for concept_name, edge_group in self.concepts_edges.items():
            schema_edges = (
                self.medrecord.schema.group(edge_group).edges
                if edge_group in self.medrecord.schema.groups
                else None
            )

            all_attribute_summary[concept_name] = extract_attribute_summary(
                self.medrecord.edge[self.medrecord.group(edge_group)["edges"]],
                schema=schema_edges,
                summary_type="extended",
            )

            if attributes:
                attribute_summary[concept_name] = {}

                for attribute_name, attribute in attributes.items():
                    if attribute in all_attribute_summary[concept_name]:
                        attribute_summary[concept_name][attribute_name] = (
                            all_attribute_summary[concept_name][attribute]
                        )

                        attributes_found.append(attribute_name)

        if not attributes:
            return all_attribute_summary

        # check if any attributes are not actually found in the cohort group
        missing_attributes = set(attributes) - set(attributes_found)

        if len(missing_attributes) > 0:
            msg = f"""{"Attribute " if len(missing_attributes) == 1 else "Attributes "}
                    {", ".join([str(attr) for attr in missing_attributes])} not found
                    for the cohort."""
            raise ValueError(msg)

        return attribute_summary
