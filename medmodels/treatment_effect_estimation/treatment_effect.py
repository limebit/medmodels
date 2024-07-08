"""
This module provides a class for analyzing treatment effects in medical records.

The TreatmentEffect class facilitates the analysis of treatment effects over time or
across different patient groups. It allows users to identify patients who underwent
treatment and experienced outcomes, and find a control group with similar criteria but
without undergoing the treatment. The class supports customizable criteria filtering,
time constraints between treatment and outcome, and optional matching of control groups
to treatment groups using a specified matching class.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, Set, Tuple

import pandas as pd

from medmodels import MedRecord
from medmodels.matching.algorithms.propensity_score import Model
from medmodels.matching.matching import MatchingMethod
from medmodels.matching.metrics import Metric
from medmodels.medrecord import node
from medmodels.medrecord.querying import NodeOperation
from medmodels.medrecord.types import (
    Group,
    MedRecordAttribute,
    MedRecordAttributeInputList,
    NodeIndex,
)
from medmodels.treatment_effect_estimation.builder import TreatmentEffectBuilder
from medmodels.treatment_effect_estimation.estimate import Estimate
from medmodels.treatment_effect_estimation.report import Report


class TreatmentEffect:
    """
    This class facilitates the analysis of treatment effects over time and across
    different patient groups.
    """

    _treatments_group: Group
    _outcomes_group: Group

    _patients_group: Group
    _time_attribute: MedRecordAttribute

    _washout_period_days: Dict[str, int]
    _washout_period_reference: Literal["first", "last"]

    _grace_period_days: int
    _grace_period_reference: Literal["first", "last"]

    _follow_up_period_days: int
    _follow_up_period_reference: Literal["first", "last"]

    _outcome_before_treatment_days: Optional[int]

    _filter_controls_operation: Optional[NodeOperation]

    _matching_method: Optional[MatchingMethod]
    _matching_essential_covariates: MedRecordAttributeInputList
    _matching_one_hot_covariates: MedRecordAttributeInputList
    _matching_model: Model
    _matching_distance_metric: Metric
    _matching_number_of_neighbors: int
    _matching_hyperparam: Optional[Dict[str, Any]]

    def __init__(
        self,
        treatment: Group,
        outcome: Group,
    ) -> None:
        """
        Initializes a Treatment Effect analysis setup with the group of the Medrecord
        that contains the treatment node IDs and the group of the Medrecord that
        contains the outcome node IDs.

        Args:
            treatment (Group): The group of treatments to analyze.
            outcome (Group): The group of outcomes to analyze.
        """
        TreatmentEffect._set_configuration(self, treatment=treatment, outcome=outcome)

    @classmethod
    def builder(cls) -> TreatmentEffectBuilder:
        """Creates a TreatmentEffectBuilder instance for the TreatmentEffect class."""

        return TreatmentEffectBuilder()

    @staticmethod
    def _set_configuration(
        treatment_effect: TreatmentEffect,
        *,
        treatment: Group,
        outcome: Group,
        patients_group: Group = "patients",
        time_attribute: MedRecordAttribute = "time",
        washout_period_days: Dict[str, int] = dict(),
        washout_period_reference: Literal["first", "last"] = "first",
        grace_period_days: int = 0,
        grace_period_reference: Literal["first", "last"] = "last",
        follow_up_period_days: int = 365,
        follow_up_period_reference: Literal["first", "last"] = "last",
        outcome_before_treatment_days: Optional[int] = None,
        filter_controls_operation: Optional[NodeOperation] = None,
        matching_method: Optional[MatchingMethod] = None,
        matching_essential_covariates: MedRecordAttributeInputList = ["gender", "age"],
        matching_one_hot_covariates: MedRecordAttributeInputList = ["gender"],
        matching_model: Model = "logit",
        matching_distance_metric: Metric = "absolute",
        matching_number_of_neighbors: int = 1,
        matching_hyperparam: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes a Treatment Effect analysis setup with specified treatments and
        outcomes within a medical record dataset.
        Validates the presence of specified dimensions and attributes within the
        provided MedRecord object, ensuring the specified treatments and outcomes are
        valid and available for analysis.

        Args:
            treatment (Group): The group of treatments to analyze.
            outcome (Group): The group of outcomes to analyze.
            patients_group (Group, optional): The group of patients to analyze.
                Defaults to "patients".
            time_attribute (MedRecordAttribute, optional): The time attribute to use for
                time-based analysis. Defaults to "time".
            washout_period_days (Dict[str, int], optional): The washout period in days
                for each treatment group. Defaults to dict().
            washout_period_reference (Literal["first", "last"], optional): The reference
                point for the washout period. Defaults to "first".
            grace_period_days (int, optional): The grace period in days after the
                treatment. Defaults to 0.
            grace_period_reference (Literal["first", "last"], optional): The reference
                point for the grace period. Defaults to "last".
            follow_up_period_days (int, optional): The follow-up period in days after
                the treatment. Defaults to 365.
            follow_up_period_reference (Literal["first", "last"], optional): The
                reference point for the follow-up period. Defaults to "last".
            outcome_before_treatment_days (Optional[int], optional): The number of days
                before the treatment to consider for outcomes. Defaults to None.
            filter_controls_operation (Optional[NodeOperation], optional): An optional
                operation to filter the control group based on specified criteria.
                Defaults to None.
            matching_method (Optional[MatchingMethod]): The method to match treatment
                and control groups. Defaults to None.
            matching_essential_covariates (MedRecordAttributeInputList, optional):
                The essential covariates to use for matching. Defaults to
                ["gender", "age"].
            matching_one_hot_covariates (MedRecordAttributeInputList, optional):
                The one-hot covariates to use for matching. Defaults to
                ["gender"].
            matching_model (Model, optional): The model to use for matching.
                Defaults to "logit".
            matching_distance_metric (Metric, optional): The distance metric
                to use for matching. Defaults to "mahalanobis".
            matching_number_of_neighbors (int, optional): The number of
                neighbors to match for each treated subject. Defaults to 1.
            matching_hyperparam (Optional[Dict[str, Any]], optional): The
                hyperparameters for the matching model. Defaults to None.
        """
        treatment_effect._patients_group = patients_group
        treatment_effect._time_attribute = time_attribute

        treatment_effect._treatments_group = treatment
        treatment_effect._outcomes_group = outcome

        treatment_effect._washout_period_days = washout_period_days
        treatment_effect._washout_period_reference = washout_period_reference
        treatment_effect._grace_period_days = grace_period_days
        treatment_effect._grace_period_reference = grace_period_reference
        treatment_effect._follow_up_period_days = follow_up_period_days
        treatment_effect._follow_up_period_reference = follow_up_period_reference
        treatment_effect._outcome_before_treatment_days = outcome_before_treatment_days
        treatment_effect._filter_controls_operation = filter_controls_operation

        treatment_effect._matching_method = matching_method
        treatment_effect._matching_essential_covariates = matching_essential_covariates
        treatment_effect._matching_one_hot_covariates = matching_one_hot_covariates
        treatment_effect._matching_model = matching_model
        treatment_effect._matching_distance_metric = matching_distance_metric
        treatment_effect._matching_number_of_neighbors = matching_number_of_neighbors
        treatment_effect._matching_hyperparam = matching_hyperparam

    def _find_groups(
        self, medrecord: MedRecord
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]:
        """
        Identifies patients who underwent treatment and experienced outcomes, and finds
        a control group with similar criteria but without undergoing the treatment. This
        method supports customizable criteria filtering, time constraints between
        treatment and outcome, and optional matching of control groups to treatment
        groups using a specified matching class.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]: A
                tuple containing the IDs of patients in the treatment true, treatment
                false, control true, and control false groups, respectively.
        """
        # Find patients that underwent the treatment
        treated_group = self._find_treated_patients(medrecord)
        treated_group, washout_nodes = self._apply_washout_period(
            medrecord, treated_group
        )
        treated_group, treatment_true, outcome_before_treatment_nodes = (
            self._find_outcomes(medrecord, treated_group)
        )
        treatment_false = treated_group - treatment_true

        # Find the controls (patients that did not undergo the treatment)
        control_group = set(medrecord.group(self._patients_group))
        control_true, control_false = self._find_controls(
            medrecord=medrecord,
            control_group=control_group,
            treated_group=treated_group,
            rejected_nodes=washout_nodes | outcome_before_treatment_nodes,
            filter_controls_operation=self._filter_controls_operation,
        )

        return treatment_true, treatment_false, control_true, control_false

    def _find_treated_patients(self, medrecord: MedRecord) -> Set[NodeIndex]:
        """Find the patients that underwent the treatment.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.

        Returns:
            Set[NodeIndex]: A set of patient nodes that underwent the treatment.

        Raises:
            ValueError: If no patients are found for the treatment groups in the
                MedRecord.
        """
        treated_group = set()

        treatments = medrecord.group(self._treatments_group)

        # Create the group with all the patients that underwent the treatment
        for treatment in treatments:
            treated_group.update(
                set(
                    medrecord.select_nodes(
                        node().in_group(self._patients_group)
                        & node()
                        .index()
                        .is_in(medrecord.neighbors(treatment, directed=False))
                    )
                )
            )
        if not treated_group:
            raise ValueError(
                "No patients found for the treatment groups in this MedRecord."
            )

        return treated_group

    def _find_outcomes(
        self, medrecord: MedRecord, treated_group: Set[NodeIndex]
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]:
        """Find the patients that had the outcome after the treatment. If set in the
        configuration, remove the ones that already had the outcome before the
        treatment.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.
            treated_group (Set[NodeIndex]): A set of patient nodes that underwent the
                treatment.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]: A tuple containing:
                - The updated set of patient nodes that underwent the treatment.
                - The nodes that had the outcome after the treatment.
                - The nodes that had the outcome before the treatment (to be rejected).
                    Only if the outcome_before_treatment_days is set.

        Raises:
            ValueError: If no outcomes are found in the MedRecord for the specified
                outcome group.
        """
        treatment_true = set()
        outcome_before_treatment_nodes = set()

        # Find nodes with the outcomes
        outcomes = medrecord.group(self._outcomes_group)
        if not outcomes:
            raise ValueError(
                f"No outcomes found in the MedRecord for group {self._outcomes_group}"
            )

        for outcome in outcomes:
            nodes_to_check = set(
                medrecord.select_nodes(
                    node().index().is_in(medrecord.neighbors(outcome, directed=False))
                    & node().index().is_in(list(treated_group))
                )
            )

            # Find patients that had the outcome before the treatment
            if self._outcome_before_treatment_days:
                outcome_before_treatment_nodes.update(
                    {
                        node_index
                        for node_index in nodes_to_check
                        if self._find_node_in_time_window(
                            medrecord,
                            node_index,
                            outcome,
                            start_days=-self._outcome_before_treatment_days,
                            end_days=0,
                            reference="first",
                        )
                    }
                )
                nodes_to_check -= outcome_before_treatment_nodes

            # Find patients that had the outcome after the treatment
            treatment_true.update(
                {
                    node_index
                    for node_index in nodes_to_check
                    if self._find_node_in_time_window(
                        medrecord,
                        node_index,
                        outcome,
                        start_days=self._grace_period_days,
                        end_days=self._follow_up_period_days,
                        reference=self._follow_up_period_reference,
                    )
                }
            )

        treated_group -= outcome_before_treatment_nodes
        if outcome_before_treatment_nodes:
            dropped_num = len(outcome_before_treatment_nodes)
            logging.warning(
                f"{dropped_num} subject{' was' if dropped_num == 1 else 's were'} "
                f"dropped due to outcome before treatment."
            )

        return treated_group, treatment_true, outcome_before_treatment_nodes

    def _apply_washout_period(
        self, medrecord: MedRecord, treated_group: Set[NodeIndex]
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex]]:
        """Apply the washout period to the treatment group.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.
            treated_group (Set[NodeIndex]): A set of patient nodes that underwent the
                treatment.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex]]: A tuple containing the updated set of
                patient nodes that underwent the treatment and the nodes that were
                dropped due to the washout period.
        """
        washout_nodes = set()
        if not self._washout_period_days:
            return treated_group, washout_nodes

        # Apply the washout period to the treatment group
        # TODO: washout in both directions? We need a List then
        for washout_group_id, washout_days in self._washout_period_days.items():
            for washout_node in medrecord.group(washout_group_id):
                washout_nodes.update(
                    {
                        treated_node
                        for treated_node in treated_group
                        if self._find_node_in_time_window(
                            medrecord,
                            treated_node,
                            washout_node,
                            start_days=-washout_days,
                            end_days=0,
                            reference=self._washout_period_reference,
                        )
                    }
                )

                treated_group -= washout_nodes

        if washout_nodes:
            dropped_num = len(washout_nodes)
            logging.warning(
                f"{dropped_num} subject{' was' if dropped_num == 1 else 's were'} "
                f"dropped due to outcome before treatment."
            )
        return treated_group, washout_nodes

    def _find_node_in_time_window(
        self,
        medrecord: MedRecord,
        node_index: NodeIndex,
        event_node: NodeIndex,
        start_days: int,
        end_days: int,
        reference: Literal["first", "last"],
    ) -> bool:
        """
        Determines whether an event occurred within a specified time window for a given
        patient node. This method helps in identifying events that are temporally
        related to a reference event by considering the temporal sequence of events.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.
            node_index (NodeIndex): The patient node to evaluate.
            event_node (NodeIndex): The event node to check for its occurrence.
            start_days (int): The start of the time window in days relative to the
                reference event.
            end_days (int): The end of the time window in days relative to the
                reference event.
            reference (Literal["first", "last"]): The reference point for the time
                window.

        Returns:
            bool: True if the event occurred within the specified time window;
                False otherwise.

        Raises:
            ValueError: If the time attribute is not found in the edge attributes.
        """
        # Find the reference time for the node
        start_period = pd.Timedelta(days=start_days)
        end_period = pd.Timedelta(days=end_days)
        reference_time = self._find_reference_time(
            medrecord, node_index, reference=reference
        )

        # Check if the event happened within the specified time window
        edges = medrecord.edges_connecting(node_index, event_node, directed=False)
        for edge in edges:
            edge_attributes = medrecord.edge[edge]
            if self._time_attribute not in edge_attributes:
                raise ValueError("Time attribute not found in the edge attributes")

            event_time = pd.to_datetime(str(edge_attributes[self._time_attribute]))
            time_diff = event_time - reference_time

            # Check that the event happened within the specified time window
            if start_period <= time_diff <= end_period:
                return True

        # Return False if no event happened within the time window
        return False

    def _find_reference_time(
        self,
        medrecord: MedRecord,
        node_index: NodeIndex,
        reference: Literal["first", "last"],
    ) -> pd.Timestamp:
        """
        Determines the timestamp of the reference exposure to any treatment in the
        predefined treatment list for a specified patient node. This method is crucial
        for analyzing the temporal sequence of treatments and outcomes.

        This function iterates over all treatments and finds the reference timestamp
        among them (first or last), ensuring that the analysis considers the reference
        treatment exposure.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.
            node_index (NodeIndex): The patient node for which to determine the first
                treatment exposure time.
            reference (Literal["first", "last"]): The reference point for the treatment
                exposure time. Options include "first" and "last".

        Returns:
            pd.Timestamp: The timestamp of the reference treatment exposure.

        Raises:
            ValueError: If no treatments are found in the MedRecord for the specified
                treatment group.
            ValueError: If no treatment edge with a time attribute is found for the
                node, indicating an issue with the data or the specified treatments.
            ValueError: If no treatment is found for the specified node in the
                MedRecord.
        """
        if reference == "first":
            time_treat = pd.Timestamp.max
            operation = min
        elif reference == "last":
            time_treat = pd.Timestamp.min
            operation = max

        treatments = medrecord.group(self._treatments_group)
        if not treatments:
            raise ValueError(
                f"No treatments found in MedRecord for group {self._treatments_group}"
            )

        for treatment in treatments:
            edges = medrecord.edges_connecting(node_index, treatment, directed=False)

            # If the node does not have the treatment, continue
            if not edges:
                continue

            # If the node has the treatment, check if it has the time attribute
            edge_values = medrecord.edge[edges].values()

            if not all(
                self._time_attribute in edge_attribute for edge_attribute in edge_values
            ):
                raise ValueError("Time attribute not found in the edge attributes")

            # Find the minimum time of the treatments
            edge_times = [
                pd.to_datetime(str(edge_attribute[self._time_attribute]))
                for edge_attribute in edge_values
            ]
            if edge_times:
                reference_time = operation(edge_times)
                time_treat = operation(reference_time, time_treat)

        if time_treat == pd.Timestamp.max or time_treat == pd.Timestamp.min:
            raise ValueError(
                f"No treatment found for node {node_index} in this MedRecord"
            )

        return time_treat

    def _find_controls(
        self,
        medrecord: MedRecord,
        control_group: Set[NodeIndex],
        treated_group: Set[NodeIndex],
        rejected_nodes: Set[NodeIndex] = set(),
        filter_controls_operation: Optional[NodeOperation] = None,
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex]]:
        """
        Identifies control groups among patients who did not undergo the specified
        treatments.

        It takes the control group and removes the rejected nodes, the treated nodes,
        and applies the filter_controls_operation if specified.

        Control groups are divided into those who had the outcome
        (control_true) and those who did not (control_false), based on the presence of
        the specified outcome codes.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.
            control_group (Set[NodeIndex]): A set of patient nodes that did not undergo
                the treatment.
            treated_group (Set[NodeIndex]): A set of patient nodes that underwent the
                treatment.
            rejected_nodes (Set[NodeIndex]): A set of patient nodes that were rejected
                due to the washout period or outcome before treatment.
            filter_controls_operation (Optional[NodeOperation], optional): An optional
                operation to filter the control group based on specified criteria.
                Defaults to None.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex]]: Two sets representing the IDs of
                control patients. The first set includes patients who experienced the
                specified outcomes (control_true), and the second set includes those who
                did not (control_false).

        Raises:
            ValueError: If no patients are found for the control groups in the
                MedRecord.
            ValueError: If no outcomes are found in the MedRecord for the specified
                outcome group.
        """
        # Apply the filter to the control group if specified
        if filter_controls_operation:
            control_group = (
                set(medrecord.select_nodes(filter_controls_operation)) & control_group
            )

        control_group = control_group - treated_group - rejected_nodes
        if len(control_group) == 0:
            raise ValueError("No patients found for control groups in this MedRecord.")

        control_true = set()
        control_false = set()
        outcomes = medrecord.group(self._outcomes_group)
        if not outcomes:
            raise ValueError(
                f"No outcomes found in the MedRecord for group {self._outcomes_group}"
            )

        # Finding the patients that had the outcome in the control group
        for outcome in outcomes:
            control_true.update(
                medrecord.select_nodes(
                    node().index().is_in(list(control_group))
                    & node().index().is_in(medrecord.neighbors(outcome, directed=False))
                )
            )
        control_false = control_group - control_true

        return control_true, control_false

    @property
    def estimate(self) -> Estimate:
        """
        Creates an Estimate object for the TreatmentEffect instance.

        Returns:
            Estimate: An Estimate object for the current TreatmentEffect instance.
        """
        return Estimate(self)

    @property
    def report(self) -> Report:
        """
        Creates a Report object for the TreatmentEffect instance.

        Returns:
            Report: A Report object for the current TreatmentEffect instance.
        """
        return Report(self)
