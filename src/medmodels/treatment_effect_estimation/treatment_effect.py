import numpy as np
import pandas as pd
from typing import List, Tuple, Type
from medmodels.dataclass.dataclass import MedRecord
from medmodels.matching.matching import Matching

TREATMENT_ALL = "treatment_all"
CRITERIA_ALL = "criteria_all"


class TreatmentEffect:
    def __init__(
        self,
        medrecord: MedRecord,
        treatments: List[str],
        outcomes: List[str],
        patients_dimension: str = "patients",
        time_attribute: str = "time",
    ) -> None:
        """Treatment Effect Class.

        :param medrecord: MedRecord object
        :type medrecord: MedRecord
        :param treatment: list of treatment codes
        :type treatment: List[str]
        :param outcome: list of outcome codes
        :type outcome: List[str]
        :param patients_dimension: dimension of the patients, defaults to "patients"
        :type patients_dimension: str, optional
        :param time_attribute: time attribute, defaults to "time"
        :type time_attribute: str, optional
        """
        assert isinstance(treatments, list), "Treatment must be a list"
        assert treatments, "Treatment list is empty"
        assert isinstance(outcomes, list), "Outcome must be a list"
        assert outcomes, "Outcome list is empty"
        assert patients_dimension in medrecord.dimensions, (
            f"Dimension {patients_dimension} not found in the data. "
            f"Available dimensions: {medrecord.dimensions}"
        )

        self.medrecord = medrecord
        self.patients_dimension = patients_dimension
        self.time_attribute = time_attribute

        # Format the concepts to the format of the MedRecord if they have are prefixes
        self.treatments, self.not_found_treatments = self.format_concepts(treatments)
        self.outcomes, self.not_found_outcomes = self.format_concepts(outcomes)

        # To initialize any other method, find_groups() must be called first so that the
        # groups are sorted and the number of patients in each group is known
        self.initialize_groups()

    def initialize_groups(self) -> None:
        """Initialize the groups for the treatment and control groups."""
        self.groups_sorted = False
        self.control_false = set()
        self.control_true = set()
        self.treatment_false = set()
        self.treatment_true = set()

    def format_concepts(self, concepts: List[str]) -> Tuple[List[str], List[str]]:
        """Format the concepts to the format of the MedRecord if they have a prefix.
        Add to the not_found_concepts list if the concept is not found in the MedRecord
        with or without the prefix.

        Example: if the MedRecord has the dimension "diagnoses" and the concept is
        "diabetes", the formatted concept goes from "diabetes" to "diagnoses_diabetes".

        :param concepts: list of concepts
        :type concepts: List[str]
        :return: tuple of list of formatted concepts and list of not found concepts
        :rtype: Tuple[List[str], List[str]]
        """
        formatted_concepts = []
        not_found_concepts = []

        for concept in concepts:
            # If the concept is already in the MedRecord, add it to the list
            if concept in self.medrecord.nodes:
                formatted_concepts.append(concept)
                continue

            # Find the first concept with a prefix. If it can't be found return None
            dimension_concept = next(
                (
                    dimension_concept
                    for dimension in self.medrecord.dimensions
                    if (dimension_concept := f"{dimension}_{concept}")
                    in self.medrecord.dimension(dimension)
                ),
                None,
            )

            if dimension_concept:
                formatted_concepts.append(dimension_concept)
            else:
                not_found_concepts.append(concept)

        return formatted_concepts, not_found_concepts

    def find_groups(
        self,
        criteria_filter: List[str] = [],
        max_time_in_years: float = 1.0,
        matching_class: Type[Matching] = None,
        **matching_args,
    ) -> None:
        """Find the patients that underwent the treatment and did or did not have the
        outcome. Find a control group that did not undergo the treatment and did or
        did not have the outcome.

        :param criteria_filter: criteria to filter the patients
        :type criteria_filter: List[str]
        :param max_time_in_years: maximum time in years between treatment and outcome
            for the outcome to be considered as positive. Defaults to 1.0.
        :type max_time_in_years: float, optional
        :param matching_class: class to match the control groups with the treatment
            groups. Defaults to None. Must have an attribute matched_control where the
            set of matched control patients is stored.
        :type matching_class: Type[Matching], optional
        :param matching_args: additional keyword arguments to pass to the matching_class
        """
        assert isinstance(criteria_filter, list), "Criteria must come in a list"

        # Restart the groups in case the function is called again
        self.initialize_groups()

        # Create the group with all the patients that underwent the treatment
        for treatment in self.treatments:
            self.medrecord.add_group(
                TREATMENT_ALL,
                self.medrecord.neighbors(
                    treatment, dimension_filter=[self.patients_dimension]
                ),
                criteria=criteria_filter,
            )
        assert (
            TREATMENT_ALL in self.medrecord.groups
        ), "No patients found for the treatment groups in this MedRecord"
        treatment_all = set(self.medrecord.group(TREATMENT_ALL))
        self.treatment_all = treatment_all

        # Remove the group so that the function can be called again
        self.medrecord.remove_group(TREATMENT_ALL)

        treatment_true = set()
        treatment_false = set()

        # Finding the patients that had the outcome after the treatment
        for outcome in self.outcomes:
            treatment_true.update(
                {
                    node
                    for node in treatment_all
                    if self._is_outcome_after_treatment(
                        node, outcome, max_time_in_years
                    )
                }
            )
        treatment_false = treatment_all - treatment_true

        # Find the controls (patients that did not undergo the treatment)
        control_true, control_false = self.find_controls(
            criteria_filter=criteria_filter,
        )
        control_all = control_true | control_false

        # Matching control groups to treatment groups using the given matching class
        if matching_class:
            matching_instance = matching_class(
                treated_group=treatment_all,
                control_group=control_all,
                medrecord=self.medrecord,
                **matching_args,
            )
            matched_control = matching_instance.matched_control
            control_true = matched_control & control_true
            control_false = matched_control & control_false

        self.control_true = control_true
        self.control_false = control_false
        self.treatment_true = treatment_true
        self.treatment_false = treatment_false

        # The groups have been sorted and the other methods can be called
        self.groups_sorted = True

    def _is_outcome_after_treatment(
        self, node: str, outcome: str, max_time_in_years: float
    ) -> bool:
        """Check if the outcome occurred after the treatment and within the time limit.

        :param node: node to check
        :type node: str
        :param outcome: outcome to check
        :type outcome: str
        :param max_time_in_years: maximum time (in years) between treatment and outcome
            for the outcome to be considered as positive
        :type max_time_in_years: float
        :return: True if the outcome happened after treatment and within the time limit,
            False otherwise.
        :rtype: bool
        """
        max_time = pd.Timedelta(
            days=365 * max_time_in_years
        )  # Convert to time into days
        time_treat = self.find_first_time(node)

        # Check that the node has at least one edge to the outcome
        if not self.medrecord.edge(node, outcome):
            return False

        # Check if the outcome happened after treatment and within the time limit and
        # for each edge to the outcome
        for _, edge_attributes in self.medrecord.edge(node, outcome).items():
            # Check that the edge has the time attribute
            assert (
                self.time_attribute in edge_attributes
            ), "Time attribute not found in the edge attributes"
            time_out = pd.to_datetime(edge_attributes[self.time_attribute])
            time_diff = time_out - time_treat

            # Check that the outcome happened after treatment and within the time limit
            if time_out > time_treat and time_diff < max_time:
                return True

        # Return False if no outcome happened after treatment and within the time limit
        return False

    def find_first_time(self, node: str) -> pd.Timestamp:
        """Find the time of the first exposure to the treatments on the list for that
        node.

        :param node: node to find the first exposure time for.
        :type node: str
        :return: time of the first exposure to the treatments.
        :rtype: pd.Timestamp
        """
        time_treat = pd.Timestamp.max

        for treatment in self.treatments:
            edges = self.medrecord.edge(node, treatment)
            # If the node does not have the treatment, continue
            if edges is None:
                continue

            # If the node has the treatment, check if it has the time attribute
            edge_values = edges.values()
            assert all(
                self.time_attribute in edge for edge in edge_values
            ), "Time attribute not found in the edge attributes"

            # Find the minimum time of the treatments
            edge_times = [
                pd.to_datetime(edge[self.time_attribute]) for edge in edge_values
            ]
            if edge_times:
                min_time = np.min(edge_times)
                time_treat = min(min_time, time_treat)

        assert (
            time_treat != pd.Timestamp.max
        ), f"No treatment found for node {node} in this MedRecord"

        return time_treat

    def find_controls(
        self,
        criteria_filter: List[str] = [],
    ) -> Tuple[set, set]:
        """Find the control groups. The control groups are the patients that did not
        undergo the treatment and did or did not have the outcome.

        :param criteria_filter: criteria to filter the patients. Defaults to [].
        :type criteria_filter: List[str], optional
        :return: IDs of the control groups. True being for the control patients that
            had the outcome and false for the ones that did not have it.
        :rtype: Tuple[set, set]
        """
        if not criteria_filter:
            # If no criteria are given, use all the patients
            criteria_set = set(self.medrecord.dimension(self.patients_dimension))
        else:
            self.medrecord.add_group(
                CRITERIA_ALL,
                criteria=criteria_filter,
            )
            criteria_set = set(self.medrecord.group(CRITERIA_ALL))
            # Remove the group so that it can be created again when function is called
            self.medrecord.remove_group(CRITERIA_ALL)

        control_all = criteria_set - self.treatment_all
        assert (
            control_all != set()
        ), "No patients found for the control groups in this MedRecord"

        # Finding the patients that had the outcome in the control group
        control_true = set()
        control_false = set()
        for outcome in self.outcomes:
            control_true.update(
                {node for node in control_all if self.medrecord.edge(node, outcome)}
            )

        control_false = control_all - control_true

        return control_true, control_false

    @property
    def subject_counts(self) -> Tuple[int, int, int, int]:
        """Count the number of subjects in each group

        :return: number of events in each group
        :rtype: Tuple[int, int, int, int]
        """
        assert (
            self.groups_sorted is True
        ), "Groups must be sorted, use find_groups() method first"

        num_treat_true = len(self.treatment_true)
        num_treat_false = len(self.treatment_false)
        num_control_true = len(self.control_true)
        num_control_false = len(self.control_false)

        assert num_treat_false != 0, "No subjects found in the treatment false group"
        assert num_control_true != 0, "No subjects found in the control true group"
        assert num_control_false != 0, "No subjects found in the control false group"

        return num_treat_true, num_treat_false, num_control_true, num_control_false

    def relative_risk(self) -> float:
        """Calculate the relative risk among the groups.

        Relative risk (RR) is a measure used in epidemiological studies to estimate the
        risk of a certain event (such as disease or condition) happening in one group
        compared to another.

        The interpretation of RR is as follows:

        RR = 1 suggests no difference in risk between the two groups.
        RR > 1 suggests a higher risk of the event in the first group.
        RR < 1 suggests a lower risk of the event in the first group.

        :return: relative risk
        :rtype: float
        """
        (
            num_treat_true,
            num_treat_false,
            num_control_true,
            num_control_false,
        ) = self.subject_counts

        return (num_treat_true / (num_treat_true + num_treat_false)) / (
            num_control_true / (num_control_true + num_control_false)
        )

    def odds_ratio(self) -> float:
        """Calculate the odds ratio among the groups.

        The odds ratio (OR) is another measure used in epidemiological and statistical
        studies that quantifies the strength of the association between two events. It
        is similar to relative risk, but instead of comparing the risks of the events,
        it compares the odds.

        Here's how to interpret an odds ratio:

        OR = 1: There's no difference in odds between the two groups.
        OR > 1: The event is more likely to occur in the first group than in the second.
        OR < 1: The event is less likely to occur in the first group than in the second.

        :return: odds ratio
        :rtype: float
        """
        (
            num_treat_true,
            num_treat_false,
            num_control_true,
            num_control_false,
        ) = self.subject_counts

        return (num_treat_true / num_control_true) / (
            num_treat_false / num_control_false
        )

    def confounding_bias(self) -> float:
        """Calculates the confounding bias for controlling confounding.

        Confounder: Variable that influences dependent and independent variables.

        CB = 1: No confounding bias
        CB != 1: Confounding bias, could be potential confounder

        :return: confounding bias
        :rtype: float
        """
        (
            num_treat_true,
            num_treat_false,
            num_control_true,
            num_control_false,
        ) = self.subject_counts
        relative_risk = self.relative_risk()

        if relative_risk == 1:
            return 1

        else:
            multiplier = relative_risk - 1
            numerator = (
                num_treat_true / (num_treat_true + num_treat_false)
            ) * multiplier + 1
            denominator = (
                num_control_true / (num_control_true + num_control_false)
            ) * multiplier + 1
            return numerator / denominator
