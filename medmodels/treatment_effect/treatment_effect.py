"""This module provides a class for analyzing treatment effects in medical records.

The TreatmentEffect class facilitates the analysis of treatment effects over time or
across different patient groups. It allows users to identify patients who underwent
treatment and experienced outcomes, and find a control group with similar criteria but
without undergoing the treatment. The class supports customizable criteria filtering,
time constraints between treatment and outcome, and optional matching of control groups
to treatment groups using a specified matching class.

The default TreatmentEffect class performs an static analysis without considering time.
To perform a time-based analysis, users can specify a time attribute in the configuration
and set the washout period, grace period, and follow-up period.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, Literal, Optional, Set, Tuple

from medmodels import MedRecord
from medmodels.medrecord.querying import EdgeDirection, NodeOperand, NodeQuery
from medmodels.medrecord.types import (
    Group,
    MedRecordAttribute,
    MedRecordAttributeInputList,
    NodeIndex,
)
from medmodels.treatment_effect.builder import TreatmentEffectBuilder
from medmodels.treatment_effect.estimate import Estimate
from medmodels.treatment_effect.matching.algorithms.propensity_score import Model
from medmodels.treatment_effect.matching.matching import MatchingMethod
from medmodels.treatment_effect.report import Report


class TreatmentEffect:
    """This class facilitates the analysis of treatment effects over time and across different patient groups."""

    _treatments_group: Group
    _outcomes_group: Group

    _patients_group: Group
    _time_attribute: Optional[MedRecordAttribute]

    _washout_period_days: Dict[str, int]
    _washout_period_reference: Literal["first", "last"]

    _grace_period_days: int
    _grace_period_reference: Literal["first", "last"]

    _follow_up_period_days: int
    _follow_up_period_reference: Literal["first", "last"]

    _outcome_before_treatment_days: Optional[int]

    _filter_controls_query: Optional[NodeQuery]

    _matching_method: Optional[MatchingMethod]
    _matching_essential_covariates: MedRecordAttributeInputList
    _matching_one_hot_covariates: MedRecordAttributeInputList
    _matching_model: Model
    _matching_number_of_neighbors: int
    _matching_hyperparam: Optional[Dict[str, Any]]

    def __init__(
        self,
        treatment: Group,
        outcome: Group,
    ) -> None:
        """Initializes a Treatment Effect analysis setup with the group of the Medrecord that contains the treatment node IDs and the group of the Medrecord that contains the outcome node IDs.

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
        time_attribute: Optional[MedRecordAttribute] = None,
        washout_period_days: Dict[str, int] = dict(),
        washout_period_reference: Literal["first", "last"] = "first",
        grace_period_days: int = 0,
        grace_period_reference: Literal["first", "last"] = "last",
        follow_up_period_days: int = 365,
        follow_up_period_reference: Literal["first", "last"] = "last",
        outcome_before_treatment_days: Optional[int] = None,
        filter_controls_query: Optional[NodeQuery] = None,
        matching_method: Optional[MatchingMethod] = None,
        matching_essential_covariates: MedRecordAttributeInputList = ["gender", "age"],
        matching_one_hot_covariates: MedRecordAttributeInputList = ["gender"],
        matching_model: Model = "logit",
        matching_number_of_neighbors: int = 1,
        matching_hyperparam: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes a Treatment Effect analysis setup with specified treatments and outcomes within a medical record dataset.

        Validates the presence of specified dimensions and attributes within the
        provided MedRecord object, ensuring the specified treatments and outcomes are
        valid and available for analysis.

        Args:
            treatment (Group): The group of treatments to analyze.
            outcome (Group): The group of outcomes to analyze.
            patients_group (Group, optional): The group of patients to analyze.
                Defaults to "patients".
            time_attribute (Optional[MedRecordAttribute], optional):  The time
                attribute. If None, the treatment effect analysis is performed in an
                static way (without considering time). Defaults to None.
            washout_period_days (Dict[str, int], optional): The washout period in days
                for each treatment group. In the case of no time attribute, it is not
                applied. Defaults to dict().
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
            filter_controls_query (Optional[NodeQuery], optional): An optional
                query to filter the control group based on specified criteria.
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
            matching_number_of_neighbors (int, optional): The number of
                neighbors to match for each treated subject. Defaults to 1.
            matching_hyperparam (Optional[Dict[str, Any]], optional): The
                hyperparameters for the matching model. Defaults to None.

        Raises:
            ValueError: If the follow-up period is less than the grace period.
        """
        treatment_effect._patients_group = patients_group
        treatment_effect._time_attribute = time_attribute

        treatment_effect._treatments_group = treatment
        treatment_effect._outcomes_group = outcome

        if follow_up_period_days < grace_period_days:
            raise ValueError(
                "The follow-up period must be greater than or equal to the grace period."
            )

        treatment_effect._washout_period_days = washout_period_days
        treatment_effect._washout_period_reference = washout_period_reference
        treatment_effect._grace_period_days = grace_period_days
        treatment_effect._grace_period_reference = grace_period_reference
        treatment_effect._follow_up_period_days = follow_up_period_days
        treatment_effect._follow_up_period_reference = follow_up_period_reference
        treatment_effect._outcome_before_treatment_days = outcome_before_treatment_days
        treatment_effect._filter_controls_query = filter_controls_query

        treatment_effect._matching_method = matching_method
        treatment_effect._matching_essential_covariates = matching_essential_covariates
        treatment_effect._matching_one_hot_covariates = matching_one_hot_covariates
        treatment_effect._matching_model = matching_model
        treatment_effect._matching_number_of_neighbors = matching_number_of_neighbors
        treatment_effect._matching_hyperparam = matching_hyperparam

    def _find_groups(
        self, medrecord: MedRecord
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]:
        """Identifies patients who underwent treatment and experienced outcomes, and finds a control group with similar criteria but without undergoing the treatment.

        This method supports customizable criteria filtering, time constraints between
        treatment and outcome, and optional matching of control groups to treatment
        groups using a specified matching class.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]: A
                tuple containing the IDs of patients in the treated group who had the
                outcome (treated_outcome_true), the IDs of patients in the treated group
                who did not have the outcome (treatment_outcome_false), the IDs of
                patients in the control group who had the outcome (control_outcome_true),
                and the IDs of patients in the control group who did not have the outcome
                (control_outcome_false).
        """
        # Find patients that underwent the treatment
        treated_set = self._find_treated_patients(medrecord)

        if self._time_attribute:
            treated_set, washout_nodes = self._apply_washout_period(
                medrecord, treated_set
            )
        else:
            logging.warning(
                "Washout period is not applied because the time attribute is not set."
            )
            washout_nodes = set()

        treated_set, treated_outcome_true, outcome_before_treatment_nodes = (
            self._find_outcomes(medrecord, treated_set)
        )
        treated_outcome_false = treated_set - treated_outcome_true

        # Find the controls (patients that did not undergo the treatment)
        control_set = set(medrecord.nodes_in_group(self._patients_group))
        control_outcome_true, control_outcome_false = self._find_controls(
            medrecord=medrecord,
            control_set=control_set,
            treated_set=treated_set,
            rejected_nodes=washout_nodes | outcome_before_treatment_nodes,
            filter_controls_query=self._filter_controls_query,
        )

        return (
            treated_outcome_true,
            treated_outcome_false,
            control_outcome_true,
            control_outcome_false,
        )

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

        def query(node: NodeOperand):
            node.in_group(self._patients_group)
            node.neighbors(edge_direction=EdgeDirection.BOTH).in_group(
                self._treatments_group
            )

        treated_set = set(medrecord.select_nodes(query))
        if not treated_set:
            raise ValueError(
                "No patients found for the treatment groups in this MedRecord."
            )

        return treated_set

    def _find_outcomes(
        self, medrecord: MedRecord, treated_set: Set[NodeIndex]
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]:
        """Find the patients that had the outcome after the treatment.

        If set in the configuration, remove the ones that already had the outcome
        before the treatment.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.
            treated_set (Set[NodeIndex]): A set of patient nodes that underwent the
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
        outcome_before_treatment_nodes = set()
        outcome_before_treatment_days = self._outcome_before_treatment_days

        # Find nodes with the outcomes
        outcomes = medrecord.nodes_in_group(self._outcomes_group)
        if not outcomes:
            raise ValueError(
                f"No outcomes found in the MedRecord for group {self._outcomes_group}"
            )

        if outcome_before_treatment_days and self._time_attribute:
            outcome_before_treatment_nodes = set(
                medrecord.select_nodes(
                    lambda node: self._query_node_within_time_window(
                        node,
                        treated_set,
                        self._outcomes_group,
                        -outcome_before_treatment_days,
                        0,
                        "first",
                    )
                )
            )
            treated_set -= outcome_before_treatment_nodes

            dropped_num = len(outcome_before_treatment_nodes)
            logging.warning(
                f"{dropped_num} subject{' was' if dropped_num == 1 else 's were'} "
                f"dropped due to outcome before treatment."
            )

        if self._time_attribute:
            treated_outcome_true = set(
                medrecord.select_nodes(
                    lambda node: self._query_node_within_time_window(
                        node,
                        treated_set,
                        self._outcomes_group,
                        self._grace_period_days,
                        self._follow_up_period_days,
                        self._follow_up_period_reference,
                    )
                )
            )
        else:
            treated_outcome_true = set(
                medrecord.select_nodes(
                    lambda node: self._query_set_outcome_true(node, treated_set)
                )
            )
            logging.warning(
                "Time attribute is not set, thus the grace period, follow-up period,"
                + "and outcome before treatment are not applied. The treatment effect"
                + "analysis is performed in a static way."
            )

        return treated_set, treated_outcome_true, outcome_before_treatment_nodes

    def _apply_washout_period(
        self, medrecord: MedRecord, treated_set: Set[NodeIndex]
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex]]:
        """Apply the washout period to the treatment group.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.
            treated_set (Set[NodeIndex]): A set of patient nodes that underwent the
                treatment.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex]]: A tuple containing the updated set of
                patient nodes that underwent the treatment and the nodes that were
                dropped due to the washout period.
        """
        washout_nodes = set()
        if not self._washout_period_days:
            return treated_set, washout_nodes

        # Apply the washout period to the treatment group
        # TODO: washout in both directions? We need a List then
        for washout_group_id, washout_days in self._washout_period_days.items():
            washout_nodes.update(
                medrecord.select_nodes(
                    lambda node: self._query_node_within_time_window(
                        node,
                        treated_set,
                        washout_group_id,
                        -washout_days,
                        0,
                        self._washout_period_reference,
                    )
                )
            )
            treated_set -= washout_nodes

        if washout_nodes:
            dropped_num = len(washout_nodes)
            logging.warning(
                f"{dropped_num} subject{' was' if dropped_num == 1 else 's were'} "
                f"dropped due to outcome before treatment."
            )

        return treated_set, washout_nodes

    def _find_controls(
        self,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        rejected_nodes: Set[NodeIndex] = set(),
        filter_controls_query: Optional[NodeQuery] = None,
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex]]:
        """Identifies control groups among patients who did not undergo the specified treatments.

        It takes the control group and removes the rejected nodes, the treated nodes,
        and applies the filter_controls_query if specified.

        Control groups are divided into those who had the outcome
        (control_outcome_true) and those who did not (control_outcome_false),
        based on the presence of the specified outcome codes.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.
            control_set (Set[NodeIndex]): A set of patient nodes that did not undergo
                the treatment.
            treated_set (Set[NodeIndex]): A set of patient nodes that underwent the
                treatment.
            rejected_nodes (Set[NodeIndex]): A set of patient nodes that were rejected
                due to the washout period or outcome before treatment.
            filter_controls_query (Optional[NodeQuery], optional): An optional
                query to filter the control group based on specified criteria.
                Defaults to None.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex]]: Two sets representing the IDs of
                control patients. The first set includes patients who experienced the
                specified outcomes (control_outcome_true), and the second set includes
                patients who did not experience the outcomes (control_outcome_false).

        Raises:
            ValueError: If no patients are found for the control groups in the
                MedRecord.
            ValueError: If no outcomes are found in the MedRecord for the specified
                outcome group.
        """
        # Apply the filter to the control group if specified
        if filter_controls_query:
            control_set = (
                set(medrecord.select_nodes(filter_controls_query)) & control_set
            )

        control_set = control_set - treated_set - rejected_nodes
        if len(control_set) == 0:
            raise ValueError("No patients found for control groups in this MedRecord.")

        control_outcome_true = set()
        outcomes = medrecord.nodes_in_group(self._outcomes_group)
        if not outcomes:
            raise ValueError(
                f"No outcomes found in the MedRecord for group {self._outcomes_group}"
            )

        # Finding the patients that had the outcome in the control group
        control_outcome_true = set(
            medrecord.select_nodes(
                lambda node: self._query_set_outcome_true(node, control_set)
            )
        )
        control_outcome_false = control_set - control_outcome_true

        return control_outcome_true, control_outcome_false

    def _query_set_outcome_true(self, node: NodeOperand, set: Set[NodeIndex]):
        """Query for nodes that are in the given set and have the outcome.

        Args:
            node (NodeOperand): The node to query.
            set (Set[NodeIndex]): The set of nodes to query.
        """
        node.index().is_in(list(set))
        node.neighbors(edge_direction=EdgeDirection.BOTH).in_group(self._outcomes_group)

    def _query_node_within_time_window(
        self,
        node: NodeOperand,
        treated_set: Set[NodeIndex],
        outcome_group: Group,
        start_days: int,
        end_days: int,
        reference: Literal["first", "last"],
    ) -> None:
        """Queries for nodes with edges containing time information within a specified time window.

        It queries for nodes that:
            - Are in the treated group.
            - Have edges with time information.
            - Have edges that connect to the treatment group.
            - Have edges that connect to the outcome group.
            - The time of the outcome is within the specified time window: it being
                greater or equal than the first or last time of treatment (depending on
                the `reference`) and less or equal than the time of treatment plus the
                `end_days` specified.

        Args:
            node (NodeOperand): The node to query.
            treated_set (Set[NodeIndex]): A set of patient nodes that underwent the
                treatment.
            start_days (int): The start of the time window in days relative to the
                reference event.
            end_days (int): The end of the time window in days relative to the reference
                event.
            reference (Literal["first", "last"]): The reference point for the time window.

        Raises:
            ValueError: If the time attribute is not set.
        """
        node.index().is_in(list(treated_set))
        if self._time_attribute is None:
            raise ValueError("Time attribute is not set.")

        edges_to_treatment = node.edges()
        edges_to_treatment.attribute(self._time_attribute).is_datetime()
        edges_to_treatment.either_or(
            lambda edge: edge.source_node().in_group(self._treatments_group),
            lambda edge: edge.target_node().in_group(self._treatments_group),
        )

        edges_to_outcome = node.edges()
        edges_to_outcome.attribute(self._time_attribute).is_datetime()
        edges_to_outcome.either_or(
            lambda edge: edge.source_node().in_group(outcome_group),
            lambda edge: edge.target_node().in_group(outcome_group),
        )

        if reference == "first":
            time_of_treatment = edges_to_treatment.attribute(self._time_attribute).min()
        else:
            time_of_treatment = edges_to_treatment.attribute(self._time_attribute).max()

        time_of_outcome = edges_to_outcome.attribute(self._time_attribute)

        min_time_window = time_of_treatment.clone()
        if start_days < 0:
            min_time_window.subtract(timedelta(-start_days))
        else:
            min_time_window.add(timedelta(start_days))

        max_time_window = time_of_treatment.clone()
        if end_days < 0:
            max_time_window.subtract(timedelta(-end_days))
        else:
            max_time_window.add(timedelta(end_days))

        time_of_outcome.greater_than_or_equal_to(min_time_window)
        time_of_outcome.less_than_or_equal_to(max_time_window)

    @property
    def estimate(self) -> Estimate:
        """Creates an Estimate object for the TreatmentEffect instance.

        Returns:
            Estimate: An Estimate object for the current TreatmentEffect instance.
        """
        return Estimate(self)

    @property
    def report(self) -> Report:
        """Creates a Report object for the TreatmentEffect instance.

        Returns:
            Report: A Report object for the current TreatmentEffect instance.
        """
        return Report(self)
