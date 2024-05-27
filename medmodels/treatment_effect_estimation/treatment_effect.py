"""This module provides a class for analyzing treatment effects in medical records.

The TreatmentEffect class facilitates the analysis of treatment effects over time or
across different patient groups. It allows users to identify patients who underwent
treatment and experienced outcomes, and find a control group with similar criteria but
without undergoing the treatment. The class supports customizable criteria filtering,
time constraints between treatment and outcome, and optional matching of control groups
to treatment groups using a specified matching class.
"""

import logging
from typing import Optional, Set, Tuple

import pandas as pd

from medmodels import MedRecord
from medmodels.medrecord import node
from medmodels.medrecord.querying import NodeOperation
from medmodels.medrecord.types import Group, MedRecordAttribute, NodeIndex
from medmodels.treatment_effect_estimation.analysis_modules.adjust import Adjust
from medmodels.treatment_effect_estimation.analysis_modules.configure import Configure
from medmodels.treatment_effect_estimation.analysis_modules.estimate import Estimate
from medmodels.treatment_effect_estimation.analysis_modules.report import Report
from medmodels.treatment_effect_estimation.utils import all_neighbors, find_all_edges


class TreatmentEffect:
    """
    This class facilitates the analysis of treatment effects over time or across
    different patient groups.
    """

    def __init__(
        self,
        medrecord: MedRecord,
        treatment: Group,
        outcome: Group,
        patients_group: Group = "patients",
        time_attribute: MedRecordAttribute = "time",
    ) -> None:
        """
        Initializes a Treatment Effect analysis setup with specified treatments and
        outcomes within a medical record dataset.
        Validates the presence of specified dimensions and attributes within the
        provided MedRecord object, ensuring the specified treatments and outcomes are
        valid and available for analysis.

        Args:
            medrecord (MedRecord): An instance of MedRecord containing medical records.
            treatments (NodeIndexInputList): A list of treatment codes to analyze.
            outcomes (NodeIndexInputList): A list of outcome codes to analyze.
            patients_group (Group, optional): The group within MedRecord representing
                patients within the dataset. Defaults to "patients".
            time_attribute (MedRecordAttribute, optional): The attribute representing
                time within the dataset. Defaults to "time".

        Raises:
            AssertionError: If treatments or outcomes are not provided as lists, are
                empty, or if the specified patients group is not found within the
                MedRecord groups.
        """
        assert treatment in medrecord.groups, (
            "Treatment group not found in the data. "
            f"Available groups: {medrecord.groups}"
        )
        assert outcome in medrecord.groups, (
            "Outcome group not found in the data. "
            f"Available groups: {medrecord.groups}"
        )
        assert patients_group in medrecord.groups, (
            f"Patient group {patients_group} not found in the data. "
            f"Available groups: {medrecord.groups}"
        )

        self.medrecord = medrecord
        self.patients_group = patients_group
        self.time_attribute = time_attribute

        # Add the treatments and outcomes to the class
        self.treatments = medrecord.group(treatment)
        self.outcomes = medrecord.group(outcome)

        # Initialize the groups and attributes and find the groups
        self._initialize_groups()
        self._initialize_attributes()
        self._find_groups()

    def _initialize_groups(self) -> None:
        """Initialize the groups for the treatment and control groups."""

        # Set the groups_sorted attribute to False
        self.groups_sorted = False
        self._washout_nodes = set()
        self._outcome_before_treatment_nodes = set()

        # Initialize the groups
        self.control_false = set()
        self.control_true = set()
        self.treatment_false = set()
        self.treatment_true = set()

    def _initialize_attributes(self) -> None:
        # Set washout period
        self._washout_period_days = dict()
        self._washout_period_reference = "first"

        # Set follow-up period
        self._follow_up_period_days = 365
        self._follow_up_reference = "last"

        # Set grace period
        self._grace_period_days = 0
        # TODO: apply reference, for now together with follow-up period
        self._grace_period_reference = "last"

        # Set outcome before treatment
        self._outcome_before_treatment_days: Optional[int] = None

        # Set filter for controls with node operation
        self._filter_controls: Optional[NodeOperation] = None

    def _find_groups(self) -> None:
        """
        Identifies patients who underwent treatment and experienced outcomes, and finds
        a control group with similar criteria but without undergoing the treatment. This
        method supports customizable criteria filtering, time constraints between
        treatment and outcome, and optional matching of control groups to treatment
        groups using a specified matching class.

        Initializes and sorts patient groups based on treatment and outcome, applying
        specified criteria and time constraints. Optionally matches treatment and
        control groups using a provided matching class and arguments with Adjust class.
        """
        # Restart the groups in case the function is called again
        self._initialize_groups()

        # Find the patients that underwent the treatment
        treatment_all = self._find_treated_patients()
        treatment_all = self._apply_washout_period(treatment_all)
        treatment_all, self.treatment_true = self._find_outcomes(treatment_all)
        self.treatment_false = treatment_all - self.treatment_true

        # Find the controls (patients that did not undergo the treatment)
        self.control_true, self.control_false = self._find_controls(
            treatment_all=treatment_all, filter_controls=self._filter_controls
        )
        self.groups_sorted = True

    def _find_treated_patients(self) -> Set[NodeIndex]:
        """Find the patients that underwent the treatment.

        Returns:
            Set[NodeIndex]: A set of patient nodes that underwent the treatment.
        """
        treatment_all = set()

        # Create the group with all the patients that underwent the treatment
        for treatment in self.treatments:
            treatment_all.update(
                set(
                    self.medrecord.select_nodes(
                        node().index().is_in(self.medrecord.group(self.patients_group))
                        & all_neighbors(self.medrecord, treatment)
                    )
                )
            )
        assert (
            len(treatment_all) > 0
        ), "No patients found for the treatment groups in this MedRecord"

        return treatment_all

    def _find_outcomes(
        self, treatment_all: Set[NodeIndex]
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex]]:
        """Find the patients that had the outcome after the treatment and, if specified,
        did not have the outcome before the treatment.

        Args:
            treatment_all (Set[NodeIndex]): A set of patient nodes that underwent the
                treatment.

        Returns:
            Set[NodeIndex]: A set of patient nodes that underwent the treatment and had
                the outcome.
        """
        treatment_true = set()
        for outcome in self.outcomes:
            nodes_to_check = set(
                self.medrecord.select_nodes(
                    all_neighbors(self.medrecord, outcome)
                    & node().index().is_in(list(treatment_all))
                )
            )

            # Finding the patients that had the outcome before the treatment
            if self._outcome_before_treatment_days:
                self._outcome_before_treatment_nodes.update(
                    {
                        node_index
                        for node_index in nodes_to_check
                        if self._find_node_in_time_window(
                            node_index,
                            outcome,
                            start_days=-self._outcome_before_treatment_days,
                            end_days=0,
                            reference="first",
                        )
                    }
                )
                nodes_to_check -= self._outcome_before_treatment_nodes

            # Finding the patients that had the outcome after the treatment
            treatment_true.update(
                {
                    node_index
                    for node_index in nodes_to_check
                    if self._find_node_in_time_window(
                        node_index,
                        outcome,
                        start_days=self._grace_period_days,
                        end_days=self._follow_up_period_days,
                        reference=self._follow_up_reference,
                    )
                }
            )

        treatment_all -= self._outcome_before_treatment_nodes
        if self._outcome_before_treatment_nodes:
            dropped_num = len(self._outcome_before_treatment_nodes)
            logging.warning(
                f"{dropped_num} subjects were dropped due to outcome before treatment"
            )

        return treatment_all, treatment_true

    def _apply_washout_period(self, treatment_all: Set[NodeIndex]) -> Set[NodeIndex]:
        """Apply the washout period to the treatment group.

        Args:
            treatment_all (Set[NodeIndex]): A set of patient nodes that underwent the
                treatment.

        Returns:
            Set[NodeIndex]: A set of patient nodes that underwent the treatment after the
                washout period.
        """
        if not self._washout_period_days:
            return treatment_all

        # Apply the washout period to the treatment group
        # TODO: washout in both directions? We need a List then
        for washout_group_id, washout_days in self._washout_period_days.items():
            for washout_node in self.medrecord.group(washout_group_id):
                self._washout_nodes.update(
                    {
                        treated_node
                        for treated_node in treatment_all
                        if self._find_node_in_time_window(
                            treated_node,
                            washout_node,
                            start_days=-washout_days,
                            end_days=0,
                            reference=self._washout_period_reference,
                        )
                    }
                )

                treatment_all -= self._washout_nodes

        if self._washout_nodes:
            dropped_num = len(self._washout_nodes)
            logging.warning(
                f"{dropped_num} subjects were dropped due to washout period"
            )
        return treatment_all

    def _find_node_in_time_window(
        self,
        node_index: NodeIndex,
        event_node: NodeIndex,
        start_days: int,
        end_days: int,
        reference: str,
    ) -> bool:
        """
        Determines whether an event occurred within a specified time window for a given
        patient node. This method helps in identifying events that are temporally
        related to a reference event by considering the temporal sequence of events.

        Args:
            node_index (NodeIndex): The patient node to evaluate.
            event_node (NodeIndex): The event node to check for its occurrence.
            start_days (int): The start of the time window in days relative to the
                reference event.
            end_days (int): The end of the time window in days relative to the
                reference event.
            reference (str): The reference event to calculate the time window from.

        Returns:
            bool: True if the event occurred within the specified time window;
            False otherwise.

        This method supports the analysis of event timing by ensuring events are
            temporally linked.
        """
        start_period = pd.Timedelta(days=start_days)
        end_period = pd.Timedelta(days=end_days)
        reference_time = self._find_reference_time(node_index, reference=reference)

        # TODO: implement adirectional edges method.
        edges = find_all_edges(self.medrecord, node_index, event_node)

        # Check if the event happened within the specified time window
        for edge in edges:
            edge_attributes = self.medrecord.edge[edge]
            assert (
                self.time_attribute in edge_attributes
            ), "Time attribute not found in the edge attributes"

            event_time = pd.to_datetime(str(edge_attributes[self.time_attribute]))
            time_diff = event_time - reference_time

            # Check that the event happened within the specified time window
            if start_period <= time_diff <= end_period:
                return True

        # Return False if no event happened within the time window
        return False

    def _find_reference_time(
        self, node_index: NodeIndex, reference: str
    ) -> pd.Timestamp:
        """
        Determines the timestamp of the first exposure to any treatment in the
        predefined treatment list for a specified patient node. This method is crucial
        for analyzing the temporal sequence of treatments and outcomes.

        Args:
            node_index (NodeIndex): The patient node for which to determine the first
                treatment exposure time.

        Returns:
            pd.Timestamp: The timestamp of the first treatment exposure.

        Raises:
            AssertionError: If no treatment edge with a time attribute is found for the
                node, indicating an issue with the data or the specified treatments.

        This function iterates over all treatments and finds the earliest timestamp
        among them, ensuring that the analysis considers the initial treatment exposure.
        """
        if reference == "first":
            time_treat = pd.Timestamp.max
            operation = min
        elif reference == "last":
            time_treat = pd.Timestamp.min
            operation = max
        else:
            raise ValueError(
                "The follow_up_from parameter must be either 'first' or 'last'."
            )

        for treatment in self.treatments:
            # TODO: implement adirectional edges method.
            edges = find_all_edges(self.medrecord, node_index, treatment)

            # If the node does not have the treatment, continue
            if not edges:
                continue

            # If the node has the treatment, check if it has the time attribute
            edge_values = self.medrecord.edge[edges].values()
            assert all(
                self.time_attribute in edge_attribute for edge_attribute in edge_values
            ), "Time attribute not found in the edge attributes"

            # Find the minimum time of the treatments
            edge_times = [
                pd.to_datetime(str(edge_attribute[self.time_attribute]))
                for edge_attribute in edge_values
            ]
            if edge_times:
                reference_time = operation(edge_times)
                time_treat = operation(reference_time, time_treat)

        assert (
            time_treat != pd.Timestamp.max and time_treat != pd.Timestamp.min
        ), f"No treatment found for node {node_index} in this MedRecord"

        return time_treat

    def _find_controls(
        self,
        treatment_all: Set[NodeIndex],
        filter_controls: Optional[NodeOperation] = None,
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex]]:
        """
        Identifies control groups among patients who did not undergo the specified
        treatments. Control groups are divided into those who had the outcome
        (control_true) and those who did not (control_false), based on the presence of
        specified outcome codes.

        Args:
            treatment_all (Set[NodeIndex]): A set of patient nodes that underwent the
                treatment.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex]]: Two sets representing the IDs of
                control patients. The first set includes patients who experienced the
                specified outcomes (control_true), and the second set includes those who
                did not (control_false).

        This method facilitates the separation of patients into relevant control groups
        for further analysis of treatment effects, ensuring only those not undergoing
        the treatment are considered for control.
        """
        # Define the unwanted nodes
        unwanted_nodes = self._outcome_before_treatment_nodes | self._washout_nodes
        control_all = set(self.medrecord.group(self.patients_group))

        # Apply the filter to the control group if specified
        if filter_controls:
            control_all = (
                set(self.medrecord.select_nodes(filter_controls)) & control_all
            )

        control_all = control_all - treatment_all - unwanted_nodes
        assert (
            len(control_all) > 0
        ), "No patients found for the control groups in this MedRecord"

        control_true = set()
        control_false = set()

        # Finding the patients that had the outcome in the control group
        for outcome in self.outcomes:
            control_true.update(
                self.medrecord.select_nodes(
                    node().index().is_in(list(control_all))
                    & all_neighbors(self.medrecord, outcome)
                )
            )
        control_false = control_all - control_true - unwanted_nodes

        return control_true, control_false

    @property
    def _subject_counts(self) -> Tuple[int, int, int, int]:
        """
        Provides the count of subjects in each of the defined groups: treatment true,
        treatment false, control true, and control false. This property ensures that
        group sorting is completed before attempting to count the subjects in each
        group.

        Returns:
            Tuple[int, int, int, int]: A tuple containing the number of subjects in the
                treatment true, treatment false, control true, and control false groups,
                respectively.

        Raises:
            AssertionError: If groups have not been sorted using the `find_groups`
                method, or if any of the groups are found to be empty, indicating a
                potential issue in group formation or data filtering.

        This method is crucial for understanding the distribution of subjects across
        different groups for subsequent analysis of treatment effects.
        """

        num_treat_true = len(self.treatment_true)
        num_treat_false = len(self.treatment_false)
        num_control_true = len(self.control_true)
        num_control_false = len(self.control_false)

        assert num_treat_false != 0, "No subjects found in the treatment false group"
        assert num_control_true != 0, "No subjects found in the control true group"
        assert num_control_false != 0, "No subjects found in the control false group"

        return num_treat_true, num_treat_false, num_control_true, num_control_false

    @property
    def estimate(self) -> Estimate:
        """Creates an Estimate object for the TreatmentEffect instance.

        Returns:
            Estimate: An Estimate object for the current TreatmentEffect instance.
        """
        return Estimate(self)

    @property
    def configure(self) -> Configure:
        """Creates a Configure object for the TreatmentEffect instance.

        Returns:
            Configure: A Configure object for the current TreatmentEffect instance.
        """
        return Configure(self)

    @property
    def adjust(self) -> Adjust:
        """Creates an Adjust object for the TreatmentEffect instance.

        Returns:
            Adjust: An Adjust object for the current TreatmentEffect instance.
        """
        return Adjust(self)

    @property
    def report(self) -> Report:
        """Creates a Report object for the TreatmentEffect instance.

        Returns:
            Report: A Report object for the current TreatmentEffect instance.
        """
        return Report(self)
