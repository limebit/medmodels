"""Module for preprocessing the data for the MTGAN model.

This module preprocesses the data to be used in the MTGAN model, following these steps:

Steps:
    1. Remove uncommon concepts.
    2. Sample patients.
    3. Find the first admission for each patient.
    4. Remove admissions with less than min_codes_per_window codes.
    5. Find the relative time windows for each patient per concept.

It adds the following attributes to the MedRecord:
    - first_admission_attribute: The first admission of the patient (first time they
        have a connecting edge to a concept pertaining to the concepts_group).
    - concept_index_attribute: The index of the concept. This is used to map the
        concept to a numerical index that will be used in the model (connecting them
        to the column on the boolean Data Matrix).
    - absolute_time_window_attribute: The absolute time window for each edge. This is,
        the number of days from the first admission divided by the time_interval_days.
    - number_windows_attribute: The number of unique time windows for each patient.
    - time_window_attribute: The relative time window for each edge. This means
        collapsing the absolute time window to a ranking of unique time windows. For
        instance, if a patient has absolute time windows [0, 2, 2, 3], the method will
        assign relative time windows [0, 1, 1, 2] to the corresponding edges.
"""

import datetime
import random
from math import floor
from typing import List, Optional, Tuple, TypedDict

import pandas as pd
from torch import nn
from typing_extensions import TypeAlias

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import edge, node
from medmodels.medrecord.types import (
    Group,
    MedRecordAttribute,
    MedRecordValue,
    NodeIndex,
)
from medmodels.treatment_effect.temporal_analysis import find_reference_edge


class PreprocessingHyperparameters(TypedDict, total=True):
    minimum_occurrences_concept: int
    time_interval_days: int
    minimum_concepts_per_window: int
    number_of_sampled_patients: int


class PreprocessingHyperparametersOptional(TypedDict, total=False):
    minimum_occurrences_concept: int
    time_interval_days: int
    minimum_concepts_per_window: int
    number_of_sampled_patients: int


class PreprocessingAttributes(TypedDict):
    first_admission_attribute: MedRecordAttribute
    time_window_attribute: MedRecordAttribute
    concept_index_attribute: MedRecordAttribute
    number_of_windows_attribute: MedRecordAttribute
    absolute_time_window_attribute: MedRecordAttribute


ConceptsList: TypeAlias = List[NodeIndex]


class MTGANPreprocessor(nn.Module):
    """Preprocessor for the MTGAN model."""

    patients_group: Group
    concepts_group: Group
    time_attribute: MedRecordAttribute
    hyperparameters: PreprocessingHyperparameters
    seed: Optional[int]

    def __init__(
        self,
        hyperparameters: PreprocessingHyperparameters,
        patients_group: Group = "patients",
        concepts_group: Group = "concepts",
        time_attribute: MedRecordAttribute = "time",
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the MTGANPreprocessor.

        Args:
            hyperparameters (PreprocessingHyperparameters): The hyperparameters for the
                preprocessing.
            patients_group (Group): The group of patients in the MedRecord.
            concepts_group (Group): The group of concepts in the MedRecord.
            time_attribute (MedRecordAttribute): The attribute name for the time in the
                edges of the MedRecord.
            seed (Optional[int]): The seed for the random number generator. Default is
                None.
        """
        super().__init__()
        self.patients_group = patients_group
        self.concepts_group = concepts_group
        self.time_attribute = time_attribute
        self.hyperparameters = hyperparameters
        self.seed = seed

    def _get_attribute_name(
        self, medrecord: MedRecord, attribute_name: MedRecordAttribute
    ) -> MedRecordAttribute:
        """Generates a unique attribute name.

        If the attribute name is 'time', it will return 'time_1' if 'time' already
        exists in the MedRecord. If 'time_1' exists, it will return 'time_2', and so
        on.

        Args:
            medrecord (MedRecord): The MedRecord.
            attribute_name (MedRecordAttribute): The attribute name.

        Returns:
            MedRecordAttribute: Unique attribute name for the MedRecord.
        """
        created_name = attribute_name
        counter = 1

        # TODO: improve with query engine
        while medrecord.select_edges(
            edge().has_attribute(created_name)
        ) or medrecord.select_nodes(node().has_attribute(created_name)):
            created_name = f"{attribute_name}_{counter}"
            counter += 1

        return created_name

    def _remove_unconnected_patients(self, medrecord: MedRecord) -> None:
        """Remove patients with no connections to nodes in the concepts group.

        Args:
            medrecord (MedRecord): The MedRecord.
        """
        # TODO: Need to perfect this method with new query engine.
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

    def _remove_uncommon_concepts(
        self,
        medrecord: MedRecord,
        minimum_number_ocurrences: int,
    ) -> None:
        """Remove uncommon 'concepts' from the MedRecord.

        It removes the concepts that have less than the minimum number of occurrences.

        For example, consider a MedRecord with 5 'concepts'
        ['c1', 'c2', 'c3', 'c4', 'c5'].
        Each of these concepts occurs 4, 4, 1, 1, and 1 times respectively. If our
        'min_number_occurrences' is 2, the method will remove 'c3', 'c4', and 'c5'.

        Args:
            medrecord (MedRecord): The MedRecord.
            minimum_number_ocurrences (int): The minimum number of occurrences for a
                concept.
        """
        # TODO: Need to perfect this method with the new query engine
        for concept_index in sorted(medrecord.nodes_in_group(self.concepts_group)):
            if (
                len(
                    medrecord.edges_connecting(
                        concept_index,
                        sorted(medrecord.nodes_in_group(self.patients_group)),
                        directed=False,
                    )
                )
                < minimum_number_ocurrences
            ):
                medrecord.remove_node(concept_index)

        self._remove_unconnected_patients(medrecord)

    def _sample_patients(
        self, medrecord: MedRecord, number_of_sampled_patients: int
    ) -> None:
        """Samples patients from the MedRecord.

        If the 'number_of_sampled_patients' is 0, it will sample all patients. If the
        'number_of_sampled_patients' is greater than the number of patients, it will raise
        a ValueError.

        Args:
            medrecord (MedRecord): The MedRecord.
            number_of_sampled_patients (int): The number of sampled patients.

        Raises:
            ValueError: If the number of sampled patients is greater than the number of
                patients.
        """
        number_patients = len(medrecord.nodes_in_group(self.patients_group))

        if number_of_sampled_patients > number_patients:
            raise ValueError(
                f"Number of sampled patients ({number_of_sampled_patients}) is greater than the number of patients in the MedRecord ({number_patients})"
            )

        if (
            number_of_sampled_patients == 0
            or number_of_sampled_patients == number_patients
        ):
            return

        patient_indices = sorted(medrecord.nodes_in_group(self.patients_group))
        removed_number_of_patients = len(patient_indices) - number_of_sampled_patients

        random.seed(self.seed)
        removed_patient_indices = random.sample(
            patient_indices, removed_number_of_patients
        )

        medrecord.remove_node(removed_patient_indices)

    def _find_first_admission(self, medrecord: MedRecord) -> MedRecordAttribute:
        """Finds the first admission for each patient and assigns it as a patient node attribute.

        For instance, consider a MedRecord with patients 'p1', 'p2', 'p3' where each
        had admissions on several dates.
        'p1': ['2021-01-01', '2021-02-14', '2021-04-08']
        'p2': ['2021-02-01', '2021-05-10', '2021-08-22']
        'p3': ['2021-01-15', '2021-06-20', '2021-07-30']

        Running the _find_first_admission will add a new attribute 'first_admission'
        (or 'first_admission_1' if 'first admission' is already an available attribute)
        to each patient noting the date of their first admission:
        'p1': '2021-01-01'
        'p2': '2021-02-01'
        'p3': '2021-01-15'

        It also returns the name of the attribute where the first admission is stored.

        Args:
            medrecord (MedRecord): The MedRecord.

        Returns:
            MedRecordAttribute: The name of the attribute where the first admission is
                stored in the MedRecord.
        """
        first_admission_attribute = self._get_attribute_name(
            medrecord, "first_admission"
        )
        # TODO: Need to perfect this method with new query engine.
        for patient_index in sorted(medrecord.nodes_in_group(self.patients_group)):
            medrecord.node[patient_index, first_admission_attribute] = medrecord.edge[
                find_reference_edge(
                    medrecord,
                    patient_index,
                    self.concepts_group,
                    reference="first",
                    time_attribute=self.time_attribute,
                )
            ][self.time_attribute]

        return first_admission_attribute

    def _remove_low_number_time_windows(
        self,
        medrecord: MedRecord,
        patient_index: NodeIndex,
        absolute_time_attribute: MedRecordAttribute,
        min_codes_per_window: int,
    ) -> List[MedRecordValue]:
        """Remove the time windows with a lower number of codes than the minimum (min_codes_per_window).

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
        if min_codes_per_window == 0:
            return time_windows

        # Find which time windows appear less than min_codes_per_window times
        time_window_counts = pd.Series(time_windows).value_counts()

        # TODO: maybe we should use the new query engine here?
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

    def _assign_relative_time_windows(
        self,
        medrecord: MedRecord,
        patient_index: NodeIndex,
        time_windows: List[MedRecordValue],
        absolute_time_window_attribute: MedRecordAttribute,
        time_window_attribute: MedRecordAttribute,
        number_of_windows_attribute: MedRecordAttribute,
    ) -> None:
        """Assign relative time windows to patient edges and remove patients with only one time window.

        This method creates a ranking of unique time windows, assigns these rankings to
        the edges of the patient, and removes patients with less than two unique time
        windows.

        For example, if a patient has absolute time windows [0, 2, 2, 3], the method
        will assign relative time windows [0, 1, 1, 2] to the corresponding edges.

        Args:
            medrecord (MedRecord): The MedRecord to be modified.
            patient_index (NodeIndex): The index of the patient being processed.
            time_windows (List[MedRecordValue]): The list of absolute time windows for
                the patient.
            absolute_time_window_attribute (MedRecordAttribute): The attribute name for
                absolute time windows.
            time_window_attribute (MedRecordAttribute): The attribute name for relative
                time windows.
            number_windows_attribute (MedRecordAttribute): The attribute name for
                the number of unique time windows.

        Note:
            This method will remove the patient node if there are less than two unique
                time windows.
        """
        unique_sorted = sorted(
            set(time_windows), key=lambda x: (isinstance(x, bool), x)
        )
        value_to_rank = {value: rank for rank, value in enumerate(unique_sorted)}

        # Remove patients with only one time window
        if len(value_to_rank) < 2:
            medrecord.remove_node(patient_index)
            return

        medrecord.node[patient_index, number_of_windows_attribute] = len(value_to_rank)

        for single_edge in medrecord.edges_connecting(
            patient_index,
            sorted(medrecord.nodes_in_group(self.concepts_group)),
            directed=False,
        ):
            medrecord.edge[single_edge, time_window_attribute] = value_to_rank[
                medrecord.edge[single_edge, absolute_time_window_attribute]
            ]

    def _find_relative_times(
        self,
        medrecord: MedRecord,
        first_admission_attribute: MedRecordAttribute,
        concept_index_attribute: MedRecordAttribute,
        time_interval_days: int,
        min_codes_per_window: int,
    ) -> PreprocessingAttributes:
        """Finds the relative time windows for each patient and removes the time windows with less than min_codes_per_window codes.

        For instance, consider a single patient, 'p1', with admission dates as follows:
        'p1': ['2021-01-01', '2021-01-15', '2021-01-25']

        If the `time_interval_days` is 7 days, the method will compute time window as
        the floor division of the difference in days from the first admission and the
        `time_interval_days`. So, the resulting ABSOLUTE time windows will be:
        'p1': [0, 2, 3]

        The relative time windows will be assigned as the ranking of the unique time
        windows. For instance, if a patient has absolute time windows [0, 2, 3], the
        method will assign relative time windows [0, 1, 2] to the corresponding edges.

        Args:
            medrecord (MedRecord): The MedRecord.
            first_admission_attribute (MedRecordAttribute): The first admission
                attribute.
            time_interval_days (int): The time interval in days.
            min_codes_per_window (int): The minimum number of codes per window.

        Returns:
            PreprocessingAttributes: The preprocessing attributes' names in the
                MedRecord.

        Raises:
            ValueError: If the first admission attribute or the time attribute are not
                datetime objects.

        Note:
            This method will remove the patient node if it has concepts in less than
                two unique time windows.
        """
        absolute_time_window_attribute = self._get_attribute_name(
            medrecord, "absolute_time_window"
        )
        time_window_attribute = self._get_attribute_name(medrecord, "time_window")
        number_of_windows_attribute = self._get_attribute_name(
            medrecord, "number_of_windows"
        )

        # TODO: need to perfect this with new queries.
        for patient_index in sorted(medrecord.nodes_in_group(self.patients_group)):
            first_admission = medrecord.node[patient_index, first_admission_attribute]
            if not isinstance(first_admission, datetime.datetime):
                raise ValueError(
                    f"First admission attribute needs to be a datetime object, but got {type(first_admission)}"
                )

            for single_edge in medrecord.edges_connecting(
                patient_index,
                sorted(medrecord.nodes_in_group(self.concepts_group)),
                directed=False,
            ):
                time = medrecord.edge[single_edge, self.time_attribute]
                if not isinstance(time, datetime.datetime):
                    raise ValueError(
                        f"Time attribute needs to be a datetime object, but got {type(time)}"
                    )

                # Assign the absolute time window and the concept index to the edge for easier querying later
                medrecord.edge[single_edge, absolute_time_window_attribute] = floor(
                    (time - first_admission).days // time_interval_days
                )
                medrecord.edge[single_edge, concept_index_attribute] = (
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
                absolute_time_window_attribute,
                min_codes_per_window,
            )
            self._assign_relative_time_windows(
                medrecord=medrecord,
                patient_index=patient_index,
                time_windows=time_windows,
                absolute_time_window_attribute=absolute_time_window_attribute,
                time_window_attribute=time_window_attribute,
                number_of_windows_attribute=number_of_windows_attribute,
            )

        preprocessing_attributes = PreprocessingAttributes(
            first_admission_attribute=first_admission_attribute,
            time_window_attribute=time_window_attribute,
            concept_index_attribute=concept_index_attribute,
            number_of_windows_attribute=number_of_windows_attribute,
            absolute_time_window_attribute=absolute_time_window_attribute,
        )

        return preprocessing_attributes

    def _assign_concept_indices(
        self, medrecord: MedRecord, concept_index_attribute: MedRecordAttribute
    ) -> ConceptsList:
        """Assigns an index to each concept in the MedRecord and returns a list of concepts in order of their index.

        For instance, consider a MedRecord with concepts 'c5', 'c2', 'c3', 'c4', 'c1'.
        The method will assign an index to each concept, sorted alphabetically:
        'c5': 4, 'c2': 1, 'c3': 2, 'c4': 3, 'c1': 0. And the list of concepts will be
        ['c1', 'c2', 'c3', 'c4', 'c5'].


        Args:
            medrecord (MedRecord): The MedRecord.
            concept_index_atribute (MedRecordAttribute): The attribute name for the
                concept index.

        Returns:
            ConceptsList: The list of concepts in order of their index.
        """

        concepts_list = []
        for concept_index, concept in enumerate(
            sorted(medrecord.nodes_in_group(self.concepts_group))
        ):
            concepts_list.append(concept)
            medrecord.node[concept, concept_index_attribute] = concept_index

        return concepts_list

    def preprocess(
        self, medrecord: MedRecord
    ) -> Tuple[MedRecord, ConceptsList, PreprocessingAttributes]:
        """Preprocess the MedRecord.

        - Remove uncommon concepts.
        - Sample patients.
        - Find the first admission for each patient.
        - Remove admissions with less than min_codes_per_window codes.
        - Find the relative times for each patient per concept.

        Args:
            medrecord (MedRecord): The MedRecord.

        Returns:
            Tuple[MedRecord, ConceptsList, PreprocessingAttributes]:
                The preprocessed MedRecord, the concepts list, and the preprocessing
                attributes' names in the preprocessed MedRecord.

        Raises:
            ValueError: If no patients or no concepts are in the MedRecord groups with
                the given group names before preprocessing. Also another ValueError
                raised if none are left after preprocessing in either group. If no
                edges have the given time attribute, also a ValueError is raised.

        Note:
            It requires the time attribute to be present in the edges of the MedRecord
                and its values to be datetime objects.
            This method will remove patient nodes with less than two unique time
                windows.
        """
        if not medrecord.select_edges(edge().has_attribute(self.time_attribute)):
            raise ValueError("No edges in the MedRecord with that time attribute")

        # TODO: copy of a MedRecord instead of modifying the original one once it is implemented
        self._remove_uncommon_concepts(
            medrecord,
            minimum_number_ocurrences=self.hyperparameters[
                "minimum_occurrences_concept"
            ],
        )
        concept_index_attribute = self._get_attribute_name(medrecord, "concept_index")
        first_admission_attribute = self._find_first_admission(medrecord)
        preprocessing_attributes = self._find_relative_times(
            medrecord,
            first_admission_attribute,
            concept_index_attribute,
            self.hyperparameters["time_interval_days"],
            self.hyperparameters["minimum_concepts_per_window"],
        )
        # Pruning the concepts after removing some edges and patient nodes
        self._remove_uncommon_concepts(
            medrecord,
            minimum_number_ocurrences=1,
        )
        self._sample_patients(
            medrecord, self.hyperparameters["number_of_sampled_patients"]
        )
        concepts_list = self._assign_concept_indices(medrecord, concept_index_attribute)

        if not medrecord.nodes_in_group(self.patients_group):
            raise ValueError("No patients left after preprocessing")

        if not medrecord.nodes_in_group(self.concepts_group):
            raise ValueError("No concepts left after preprocessing")

        return medrecord, concepts_list, preprocessing_attributes
