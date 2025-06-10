"""This module provides a class for analyzing treatment effects in medical records.

The TreatmentEffect class facilitates the analysis of treatment effects over time or
across different patient groups. It allows users to identify patients who underwent
treatment and experienced outcomes, and find a control group with similar criteria but
without undergoing the treatment. The class supports customizable criteria filtering,
time constraints between treatment and outcome, and optional matching of control groups
to treatment groups using a specified matching class.

The default TreatmentEffect class performs an static analysis without considering time.
To perform a time-based analysis, users can specify a time attribute in the
configuration and set the washout period, grace period, and follow-up period.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Set, Tuple

from medmodels import MedRecord
from medmodels.medrecord.types import (
    Group,
    MedRecordAttribute,
    MedRecordAttributeInputList,
    NodeIndex,
)
from medmodels.treatment_effect.builder import TreatmentEffectBuilder
from medmodels.treatment_effect.estimate import Estimate
from medmodels.treatment_effect.report import Report

if TYPE_CHECKING:
    from medmodels import MedRecord
    from medmodels.medrecord.querying import (
        NodeIndicesOperand,
        NodeIndicesQuery,
        NodeOperand,
    )
    from medmodels.medrecord.types import (
        Group,
        MedRecordAttribute,
        MedRecordAttributeInputList,
        NodeIndex,
    )
    from medmodels.treatment_effect.matching.algorithms.propensity_score import Model
    from medmodels.treatment_effect.matching.matching import MatchingMethod


logger = logging.getLogger(__name__)


class TreatmentEffect:
    """The TreatmentEffect class for analyzing treatment effects in medical records."""

    _treatments_group: Group
    _outcomes_group: Group

    _patients_group: Group
    _time_attribute: Optional[MedRecordAttribute]

    _washout_period_days: Dict[Group, int]
    _washout_period_reference: Literal["first", "last"]

    _grace_period_days: int
    _grace_period_reference: Literal["first", "last"]

    _follow_up_period_days: int
    _follow_up_period_reference: Literal["first", "last"]

    _outcome_before_treatment_days: Optional[int]

    _filter_controls_query: Optional[NodeIndicesQuery]

    _matching_method: Optional[MatchingMethod]
    _matching_essential_covariates: Optional[MedRecordAttributeInputList]
    _matching_one_hot_covariates: Optional[MedRecordAttributeInputList]
    _matching_model: Model
    _matching_number_of_neighbors: int
    _matching_hyperparameters: Optional[Dict[str, Any]]

    def __init__(
        self,
        treatment: Group,
        outcome: Group,
    ) -> None:
        """Instantiates a Treatment Effect class.

        It requires the group of the Medrecord that contains the treatment node IDs and
        the group of the Medrecord that contains the outcome node IDs.

        Args:
            treatment (Group): The group of treatments to analyze.
            outcome (Group): The group of outcomes to analyze.
        """
        TreatmentEffect._set_configuration(self, treatment=treatment, outcome=outcome)

    @classmethod
    def builder(cls) -> TreatmentEffectBuilder:
        """Creates a TreatmentEffectBuilder instance for the TreatmentEffect class.

        Returns:
            TreatmentEffectBuilder: A TreatmentEffectBuilder instance for the
                TreatmentEffect class.
        """
        return TreatmentEffectBuilder()

    @staticmethod
    def _set_configuration(
        treatment_effect: TreatmentEffect,
        *,
        treatment: Group,
        outcome: Group,
        patients_group: Group = "patient",
        time_attribute: Optional[MedRecordAttribute] = None,
        washout_period_days: Optional[Dict[Group, int]] = None,
        washout_period_reference: Literal["first", "last"] = "first",
        grace_period_days: int = 0,
        grace_period_reference: Literal["first", "last"] = "last",
        follow_up_period_days: int = 1000 * 365,
        follow_up_period_reference: Literal["first", "last"] = "last",
        outcome_before_treatment_days: Optional[int] = None,
        filter_controls_query: Optional[NodeIndicesQuery] = None,
        matching_method: Optional[MatchingMethod] = None,
        matching_essential_covariates: Optional[MedRecordAttributeInputList] = None,
        matching_one_hot_covariates: Optional[MedRecordAttributeInputList] = None,
        matching_model: Model = "logit",
        matching_number_of_neighbors: int = 1,
        matching_hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Sets the configuration for the TreatmentEffect instance.

        Validates the presence of specified dimensions and attributes within the
        provided MedRecord object, ensuring the specified treatments and outcomes are
        valid and available for analysis.

        Args:
            treatment_effect (TreatmentEffect): The TreatmentEffect instance to
                configure.
            treatment (Group): The group of treatments to analyze.
            outcome (Group): The group of outcomes to analyze.
            patients_group (Group, optional): The group of patients to analyze.
                Defaults to "patient".
            time_attribute (Optional[MedRecordAttribute], optional):  The time
                attribute. If None, the treatment effect analysis is performed in an
                static way (without considering time). Defaults to None.
            washout_period_days (Dict[str, int], optional): The washout period in days
                for each treatment group. In the case of no time attribute, it is not
                applied. Defaults to dict().
            washout_period_reference (Literal["first", "last"], optional): The
                reference point for the washout period. Defaults to "first".
            grace_period_days (int, optional): The grace period in days after the
                treatment. Defaults to 0.
            grace_period_reference (Literal["first", "last"], optional): The reference
                point for the grace period. Defaults to "last".
            follow_up_period_days (int, optional): The follow-up period in days after
                the treatment. Defaults to 365000.
            follow_up_period_reference (Literal["first", "last"], optional): The
                reference point for the follow-up period. Defaults to "last".
            outcome_before_treatment_days (Optional[int], optional): The number of days
                before the treatment to consider for outcomes. Defaults to None.
            filter_controls_query (Optional[NodeIndicesQuery], optional): An optional
                query to filter the control group based on specified criteria.
                Defaults to None.
            matching_method (Optional[MatchingMethod]): The method to match treatment
                and control groups. Defaults to None.
            matching_essential_covariates (Optional[MedRecordAttributeInputList], optional):
                The essential covariates to use for matching. Defaults to
                ["gender", "age"].
            matching_one_hot_covariates (Optional[MedRecordAttributeInputList], optional):
                The one-hot covariates to use for matching. Defaults to
                ["gender"].
            matching_model (Model, optional): The model to use for matching.
                Defaults to "logit".
            matching_number_of_neighbors (int, optional): The number of
                neighbors to match for each treated subject. Defaults to 1.
            matching_hyperparameters (Optional[Dict[str, Any]], optional): The
                hyperparameters for the matching model. Defaults to None.

        Raises:
            ValueError: If the follow-up period is less than the grace period.
        """  # noqa: W505
        if washout_period_days is None:
            washout_period_days = {}

        treatment_effect._patients_group = patients_group
        treatment_effect._time_attribute = time_attribute

        treatment_effect._treatments_group = treatment
        treatment_effect._outcomes_group = outcome

        if follow_up_period_days < grace_period_days:
            msg = (
                "The follow-up period must be greater than or equal to the grace period"
            )
            raise ValueError(msg)

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
        treatment_effect._matching_hyperparameters = matching_hyperparameters

        if washout_period_days and not time_attribute:
            logger.warning(
                "Washout period is not applied because the time attribute is not set."
            )

        if (
            grace_period_days
            or (follow_up_period_days != 1000 * 365)
            or outcome_before_treatment_days
        ) and not time_attribute:
            msg = (
                "Time attribute is not set, thus the grace period, follow-up "
                + "period, and outcome before treatment cannot be applied. The "
                + "treatment effect analysis is performed in a static way."
            )
            logger.warning(msg)

    def _find_groups(
        self, medrecord: MedRecord
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]:
        """Finds the treated and control groups in the MedRecord.

        This method finds the patients in the treated group and the control groups and
        whether they had the outcome or not. It supports customizable criteria
        filtering, time constraints between treatment and outcome, and optional
        matching of control groups to treatment groups using a specified matching class.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]: A
                tuple containing the IDs of patients in the treated group who had the
                outcome (treated_outcome_true), the IDs of patients in the treated group
                who did not have the outcome (treatment_outcome_false), the IDs of
                patients in the control group who had the outcome
                (control_outcome_true), and the IDs of patients in the control group
                who did not have the outcome (control_outcome_false).
        """
        # Find patients that underwent the treatment
        treated_set = self._find_treated_patients(medrecord)

        if self._time_attribute:
            treated_set, washout_nodes = self._apply_washout_period(
                medrecord, treated_set
            )
        else:
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

        def query(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group(self._patients_group)
            node.neighbors().in_group(self._treatments_group)

            return node.index()

        # Create the group with all the patients that underwent the treatment
        treated_set = set(medrecord.query_nodes(query))
        if not treated_set:
            msg = "No patients found for the treatment group in this MedRecord"
            raise ValueError(msg)

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
            msg = f"No outcomes found in the MedRecord for group {self._outcomes_group}"
            raise ValueError(msg)

        if outcome_before_treatment_days and self._time_attribute:
            outcome_before_treatment_nodes = set(
                medrecord.query_nodes(
                    lambda node: self._query_node_within_time_window(
                        node,
                        treated_set,
                        self._outcomes_group,
                        start_days=-outcome_before_treatment_days,
                        end_days=0,
                        reference="first",
                    )
                )
            )
            treated_set -= outcome_before_treatment_nodes

            dropped_num = len(outcome_before_treatment_nodes)
            msg = (
                f"{dropped_num} subject{' was' if dropped_num == 1 else 's were'} "
                f"dropped due to having an outcome before the treatment."
            )
            logger.warning(msg)

        if self._time_attribute:
            treated_outcome_true = set(
                medrecord.query_nodes(
                    lambda node: self._query_node_within_time_window(
                        node,
                        treated_set,
                        self._outcomes_group,
                        start_days=self._grace_period_days,
                        end_days=self._follow_up_period_days,
                        reference=self._follow_up_period_reference,
                    )
                )
            )
        else:
            treated_outcome_true = set(
                medrecord.query_nodes(
                    lambda node: self._query_set_outcome_true(node, treated_set)
                )
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
        # TODO: washout in both directions? We need a List then  # noqa: TD003, TD002
        for washout_group_id, washout_days in self._washout_period_days.items():
            washout_nodes.update(
                medrecord.query_nodes(
                    lambda node,
                    group_id=washout_group_id,
                    days=washout_days,
                    treated=treated_set: self._query_node_within_time_window(
                        node,
                        treated,
                        group_id,
                        start_days=-days,
                        end_days=0,
                        reference=self._washout_period_reference,
                    )
                )
            )
            treated_set -= washout_nodes

        if washout_nodes:
            dropped_num = len(washout_nodes)
            msg = (
                f"{dropped_num} subject{' was' if dropped_num == 1 else 's were'} "
                f"dropped due to having a treatment in the washout period."
            )
            logger.warning(msg)

        return treated_set, washout_nodes

    def _find_controls(
        self,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        rejected_nodes: Optional[Set[NodeIndex]] = None,
        filter_controls_query: Optional[NodeIndicesQuery] = None,
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex]]:
        """Identifies control patients based on specified criteria.

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
            rejected_nodes (Optional[Set[NodeIndex]], optional): A set of patient nodes
                that were rejected due to the washout period or outcome before
                treatment.
            filter_controls_query (Optional[NodeIndicesQuery], optional): An optional
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
        if rejected_nodes is None:
            rejected_nodes = set()
        if filter_controls_query:
            control_set = (
                set(medrecord.query_nodes(filter_controls_query)) & control_set
            )

        control_set = control_set - treated_set - rejected_nodes
        if len(control_set) == 0:
            msg = "No patients found for control groups in this MedRecord"
            raise ValueError(msg)

        control_outcome_true = set()
        outcomes = medrecord.nodes_in_group(self._outcomes_group)
        if not outcomes:
            msg = f"No outcomes found in the MedRecord for group {self._outcomes_group}"
            raise ValueError(msg)

        # Finding the patients that had the outcome in the control group
        control_outcome_true = set(
            medrecord.query_nodes(
                lambda node: self._query_set_outcome_true(node, control_set)
            )
        )
        control_outcome_false = control_set - control_outcome_true

        return control_outcome_true, control_outcome_false

    def _query_set_outcome_true(
        self, node: NodeOperand, set: Set[NodeIndex]
    ) -> NodeIndicesOperand:
        """Query for nodes that are in the given set and have the outcome.

        Args:
            node (NodeOperand): The node to query.
            set (Set[NodeIndex]): The set of nodes to query.

        Returns:
            NodeIndicesOperand: The node indices of the queried node.
        """
        node.index().is_in(list(set))
        node.neighbors().in_group(self._outcomes_group)

        return node.index()

    def _query_node_within_time_window(
        self,
        node: NodeOperand,
        treated_set: Set[NodeIndex],
        outcome_group: Group,
        start_days: int,
        end_days: int,
        reference: Literal["first", "last"],
    ) -> NodeIndicesOperand:
        """Queries for nodes with edges containing time info within a time window.

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
            outcome_group (Group): The group of outcomes to analyze.
            start_days (int): The start of the time window in days relative to the
                reference event.
            end_days (int): The end of the time window in days relative to the reference
                event.
            reference (Literal["first", "last"]): The reference point for the time
                window.

        Returns:
            NodeIndicesOperand: The node indices of the queried node.
        """
        node.index().is_in(list(treated_set))
        if self._time_attribute is None:
            msg = "Should never be reached"
            raise NotImplementedError(msg)

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

        # end_days should always be positive
        max_time_window.add(timedelta(end_days))

        time_of_outcome.greater_than_or_equal_to(min_time_window)
        time_of_outcome.less_than_or_equal_to(max_time_window)

        return node.index()

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
