import json
import random
from math import floor
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import pandas as pd
from torch import nn

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import edge, node
from medmodels.medrecord.types import (
    Group,
    MedRecordAttribute,
    MedRecordValue,
    NodeIndex,
)
from medmodels.treatment_effect.temporal_analysis import find_reference_edge


class PreprocessingHyperparameters(TypedDict, total=False):
    minimum_occurrences_concept: int
    time_interval_days: int
    minimum_codes_per_window: int
    number_sampled_patients: int


class PreprocessingHyperparametersTotal(TypedDict, total=True):
    minimum_occurrences_concept: int
    time_interval_days: int
    minimum_codes_per_window: int
    number_sampled_patients: int


class MTGANPreprocessor(nn.Module):
    def __init__(
        self,
        patients_group: Group = "patients",
        concepts_group: Group = "concepts",
        time_attribute: str = "time",
    ):
        super(MTGANPreprocessor, self).__init__()
        self.patients_group = patients_group
        self.concepts_group = concepts_group

        # Attribute names
        self.time_attribute = time_attribute
        self.first_admission_attribute = "no_attribute"
        self.time_window_attribute = "no_attribute"
        self.concept_index_attribute = "no_attribute"
        self.concept_edge_attribute = "no_attribute"
        self.number_admissions_attribute = "no_attribute"
        self.absolute_time_window_attribute = "no_attribute"

        self.index_to_concept_dict = {}
        self.hyperparameters: PreprocessingHyperparametersTotal = {
            "minimum_occurrences_concept": 1,
            "time_interval_days": 1,
            "minimum_codes_per_window": 1,
            "number_sampled_patients": 0,
        }

    def _load_hyperparameters(self, hyperparameters_path: Path) -> None:
        """Load hyperparameters from a JSON file.

        Raises
            FileNotFoundError: If the hyperparameters file is not found.
            ValueError: If the hyperparameters file does not contain a 'preprocessing' key.
        """
        if not hyperparameters_path.exists():
            msg = f"The hyperparameters file '{hyperparameters_path}' was not found."
            raise FileNotFoundError(msg)
        else:
            with open(hyperparameters_path, "r") as f:
                hyperparameters = json.load(f)
                if "generation" not in hyperparameters:
                    msg = "Hyperparameters file must contain a 'preprocessing' key."
                    raise ValueError(msg)
                self.hyperparameters = hyperparameters["preprocessing"]

    def _remove_uncommon_concepts(
        self,
        medrecord: MedRecord,
        min_number_occurrences: int,
        return_dict: bool = False,
    ) -> Tuple[Dict[int, NodeIndex], MedRecord]:
        """Remove uncommon concepts from the MedRecord and give them an index attribute.

        Args:
            medrecord (MedRecord): The MedRecord.
            min_number_occurrences (int): The minimum number of occurrences for a concept.
            return_dict (bool): Whether to return the index to concept dictionary.

        Return
            MedRecord: The MedRecord without the uncommon concepts
        """
        # TODO: Need to perfect this method with new queries.
        for concept_index in sorted(medrecord.nodes_in_group(self.concepts_group)):
            if (
                len(
                    medrecord.edges_connecting(
                        concept_index,
                        sorted(medrecord.nodes_in_group(self.patients_group)),
                        directed=False,
                    )
                )
                < min_number_occurrences
            ):
                medrecord.remove_node(concept_index)

        if not return_dict:
            return {}, medrecord

        index_to_concept_dict = {}
        for concept_index, concept in enumerate(
            sorted(medrecord.nodes_in_group(self.concepts_group))
        ):
            index_to_concept_dict[concept_index] = concept
            medrecord.node[concept, self.concept_index_attribute] = concept_index

        return index_to_concept_dict, medrecord

    def _find_first_admission(self, medrecord: MedRecord) -> MedRecord:
        """Finds the first admission for each patient.

        Args:
            medrecord (MedRecord): The MedRecord.

        Returns:
            pd.DataFrame: A dataframe with the first admission for each patient.
        """
        self.first_admission_attribute = self._get_attribute(
            medrecord, "first_admission"
        )
        # TODO: Need to perfect this method with new queries.
        for patient_index in sorted(medrecord.nodes_in_group(self.patients_group)):
            medrecord.node[patient_index, self.first_admission_attribute] = (
                medrecord.edge[
                    find_reference_edge(
                        medrecord,
                        patient_index,
                        self.concepts_group,
                        reference="first",
                        time_attribute=self.time_attribute,
                    )
                ][self.time_attribute]
            )

        return medrecord

    def _sample_patients(
        self, medrecord: MedRecord, num_sampled_patients: int
    ) -> MedRecord:
        """Samples patients from the MedRecord.

        Args:
            medrecord (MedRecord): The MedRecord.
            num_sampled_patients (int): The number of sampled patients.

        Returns:
            MedRecord: The MedRecord with the sampled patients.

        Raises:
            ValueError: If the number of sampled patients is greater than the number of patients.
        """
        for patient_index in sorted(medrecord.nodes_in_group(self.patients_group)):
            if (
                len(
                    medrecord.edges_connecting(
                        patient_index,
                        sorted(medrecord.nodes_in_group(self.concepts_group)),
                        directed=False,
                    )
                )
                == 0
            ):
                medrecord.remove_node(patient_index)

        num_patients = len(medrecord.nodes_in_group(self.patients_group))
        num_sample_patients = self.hyperparameters.get("number_sampled_patients", 0)

        if num_sample_patients > num_patients:
            raise ValueError(
                f"Number of sampled patients ({num_sample_patients}) is greater than the number of patients ({num_patients})."
            )

        if num_sample_patients == 0 or num_sample_patients >= num_patients:
            return medrecord

        patient_indices = sorted(medrecord.nodes_in_group(self.patients_group))
        removed_num = len(patient_indices) - num_sampled_patients
        sampled_patient_indices = random.sample(patient_indices, removed_num)

        medrecord.remove_node(sampled_patient_indices)

        return medrecord

    def _get_attribute(
        self, medrecord: MedRecord, attribute_name: MedRecordAttribute
    ) -> MedRecordAttribute:
        """Gets the attribute name with the suffix "_attribute" if the attribute is already present in the MedRecord.

        Args:
            medrecord (MedRecord): The MedRecord.
            attribute_name (MedRecordAttribute): The attribute name.

        Returns:
            MedRecordAttribute: The attribute name with the suffix "_attribute" if the
                attribute is already present in the MedRecord.
        """
        return (
            f"{attribute_name}_attribute"
            if medrecord.select_edges(edge().has_attribute(attribute_name))
            or medrecord.select_nodes(node().has_attribute(attribute_name))
            else attribute_name
        )

    def _remove_low_number_time_windows(
        self,
        medrecord: MedRecord,
        patient_index: NodeIndex,
        absolute_time_attribute: MedRecordAttribute,
        min_codes_per_window: int,
    ) -> List[MedRecordValue]:
        """Remove the time windows that appear less than min_codes_per_window times.

        If min_codes_per_window is 0, return all time windows without removing any.

        Args:
            medrecord (MedRecord): The MedRecord.
            patient_index (NodeIndex): The patient index.
            absolute_time_attribute (MedRecordAttribute): The absolute time attribute.
            min_codes_per_window (int): The minimum number of codes per window.

        Returns:
            List[MedRecordValue]: The time windows.
        """
        patient_edges = medrecord.edges_connecting(
            patient_index,
            sorted(medrecord.nodes_in_group(self.concepts_group)),
            directed=False,
        )
        time_windows = [
            medrecord.edge[e, absolute_time_attribute] for e in patient_edges
        ]
        if not min_codes_per_window:
            return time_windows

        # Find which time windows appear less than min_codes_per_window times
        time_window_counts = pd.Series(time_windows).value_counts()
        time_windows_low_number = [
            time_window
            for time_window in time_windows
            if time_window_counts.get(
                int(time_window)
                if isinstance(time_window, (int, float))
                else str(time_window),
                0,
            )
            < min_codes_per_window
        ]

        # Remove edges and time windows that appear less than min_codes_per_window times
        time_windows = [
            time_window
            for time_window in time_windows
            if time_window not in time_windows_low_number
        ]
        medrecord.remove_edge(
            medrecord.select_edges(
                edge().index().is_in(patient_edges)
                & edge()
                .attribute(absolute_time_attribute)
                .is_in(time_windows_low_number)
            )
        )

        return time_windows

    def _find_relative_times(
        self, medrecord: MedRecord, time_interval_days: int, min_codes_per_window: int
    ) -> MedRecord:
        """Finds the relative times for each patient.

        It also remove the time windows with less than min_codes_per_window codes.

        Args:
            medrecord (MedRecord): The MedRecord.
            time_interval_days (int): The time interval in days.
            min_codes_per_window (int): The minimum number of codes per window.

        Returns:
            MedRecord: The MedRecord with the relative time windows added to the edges.
        """
        self.absolute_time_window_attribute = self._get_attribute(
            medrecord, "absolute_time_window"
        )

        # TODO: need to perfect this with new queries.
        for patient_index in sorted(medrecord.nodes_in_group(self.patients_group)):
            first_admission = pd.Timestamp(
                str(medrecord.node[patient_index]["first_admission"])
            )
            for single_edge in medrecord.edges_connecting(
                patient_index,
                sorted(medrecord.nodes_in_group(self.concepts_group)),
                directed=False,
            ):
                time = pd.Timestamp(
                    str(medrecord.edge[single_edge][self.time_attribute])
                )
                medrecord.edge[single_edge, self.absolute_time_window_attribute] = (
                    floor((time - first_admission).days // time_interval_days)
                )
                medrecord.edge[single_edge, self.concept_edge_attribute] = (
                    medrecord.select_nodes(
                        node()
                        .index()
                        .is_in(list(medrecord.edge_endpoints(single_edge)))
                        & node().in_group(self.concepts_group)
                    )[0]
                )

            time_windows = self._remove_low_number_time_windows(
                medrecord,
                patient_index,
                self.absolute_time_window_attribute,
                min_codes_per_window,
            )
            unique_sorted = sorted(
                set(time_windows), key=lambda x: (isinstance(x, bool), x)
            )
            value_to_rank = {value: rank for rank, value in enumerate(unique_sorted)}

            # Remove patients with only one time window
            if len(value_to_rank) < 2:
                medrecord.remove_node(patient_index)
                continue

            medrecord.node[patient_index, self.number_admissions_attribute] = len(
                value_to_rank
            )

            for single_edge in medrecord.edges_connecting(
                patient_index,
                sorted(medrecord.nodes_in_group(self.concepts_group)),
                directed=False,
            ):
                medrecord.edge[single_edge, self.time_window_attribute] = value_to_rank[
                    medrecord.edge[single_edge, self.absolute_time_window_attribute]
                ]

        return medrecord

    def preprocess(self, medrecord: MedRecord) -> MedRecord:
        """Preprocess the MedRecord.

        - Remove uncommon concepts.
        - Sample patients.
        - Find the first admission for each patient.
        - Remove admissions with less than min_codes_per_window codes.
        - Find the relative times for each patient per concept.

        Args:
            medrecord (MedRecord): The MedRecord.

        Returns:
            MedRecord: The preprocessed MedRecord.

        Raises:
            ValueError: If no patients are left after preprocessing.
            ValueError: If no concepts are left after preprocessing
        """
        self.time_window_attribute = self._get_attribute(medrecord, "time_window")
        self.concept_index_attribute = self._get_attribute(medrecord, "concept_index")
        self.number_admissions_attribute = self._get_attribute(
            medrecord, "number_admissions"
        )
        self.concept_edge_attribute = self._get_attribute(medrecord, "concept_edge")

        minimum_number_concept = self.hyperparameters.get(
            "minimum_occurrences_concept", 1
        )
        _, medrecord = self._remove_uncommon_concepts(
            medrecord,
            minimum_number_concept,
        )

        medrecord = self._sample_patients(
            medrecord, self.hyperparameters.get("number_sampled_patients", 0)
        )
        medrecord = self._find_first_admission(medrecord)
        medrecord = self._find_relative_times(
            medrecord,
            self.hyperparameters.get("time_interval_days", 1),
            self.hyperparameters.get("minimum_codes_per_window", 1),
        )

        # Pruning the concepts after removing edges and nodes
        self.index_to_concept_dict, medrecord = self._remove_uncommon_concepts(
            medrecord,
            min_number_occurrences=1,
            return_dict=True,
        )

        if not medrecord.nodes_in_group(self.patients_group):
            raise ValueError("No patients left after preprocessing.")

        if not medrecord.nodes_in_group(self.concepts_group):
            raise ValueError("No concepts left after preprocessing.")

        return medrecord
