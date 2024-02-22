from typing import List, Optional, Set, Tuple, Type

import numpy as np
import pandas as pd

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
        """
        Initializes a Treatment Effect analysis setup with specified treatments and
        outcomes within a medical record dataset. This class facilitates the analysis of
        treatment effects over time or across different patient groups.

        Validates the presence of specified dimensions and attributes within the
        provided MedRecord object, ensuring the specified treatments and outcomes are
        valid and available for analysis.

        Args:
            medrecord (MedRecord): An instance of MedRecord containing patient records.
            treatments (List[str]): A list of treatment codes to analyze.
            outcomes (List[str]): A list of outcome codes to analyze.
            patients_dimension (str, optional): The dimension representing patients
                within the dataset. Defaults to "patients".
            time_attribute (str, optional): The attribute representing time within the
                dataset. Defaults to "time".

        Raises:
            AssertionError: If treatments or outcomes are not provided as lists, are
                empty, or if the specified patients dimension is not found within the
                MedRecord dimensions.
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
        """
        Formats the provided concept codes to match the MedRecord's naming convention if
        they have a prefix, and identifies any concepts not found in the MedRecord,
        either with or without a prefix. For example, if the MedRecord contains a
        "diagnoses" dimension and the concept is "diabetes", the formatted concept would
        be "diagnoses_diabetes".

        Args:
            concepts (List[str]): A list of concept codes to be formatted and validated.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists: the first with
                the formatted and found concept codes, and the second with the concept
                codes not found in the MedRecord.

        This method aids in ensuring concept codes are correctly formatted and exist
        within the MedRecord dataset, facilitating accurate analysis.
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
        matching_class: Optional[Type[Matching]] = None,
        **matching_args,
    ) -> None:
        """
        Identifies patients who underwent treatment and experienced outcomes, and finds
        a control group with similar criteria but without undergoing the treatment. This
        method supports customizable criteria filtering, time constraints between
        treatment and outcome, and optional matching of control groups to treatment
        groups using a specified matching class.

        Args:
            criteria_filter (List[str], optional): Criteria to filter patients. Defaults
                to an empty list.
            max_time_in_years (float, optional): Maximum time in years between treatment
                and outcome for considering the outcome as positive. Defaults to 1.0.
            matching_class (Optional[Type[Matching]], optional): Class used for matching
                control groups with treatment groups. Must have an  attribute
                `matched_control` where the set of matched control patients is stored.
                Defaults to None.
            **matching_args: Additional keyword arguments passed to the `matching_class`
                constructor.

        Initializes and sorts patient groups based on treatment and outcome, applying
        specified criteria and time constraints. Optionally matches treatment and
        control groups using a provided matching class and arguments.
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
        """
        Determines whether an outcome occurred after treatment within a specified time
        frame for a given patient node. This method helps in identifying positive
        outcomes that are directly attributable to the treatment by considering the
        temporal sequence of events.

        Args:
            node (str): The patient node to evaluate.
            outcome (str): The outcome to check for its occurrence after treatment.
            max_time_in_years (float): The maximum allowed time in years between
                treatment and outcome occurrence for the outcome to be considered as a
                result of the treatment.

        Returns:
            bool: True if the outcome occurred after the treatment and within the
                specified time limit; False otherwise.

        This method supports the analysis of treatment effectiveness by ensuring
        outcomes are temporally linked to the treatments.
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
        """
        Determines the timestamp of the first exposure to any treatment in the
        predefined treatment list for a specified patient node. This method is crucial
        for analyzing the temporal sequence of treatments and outcomes.

        Args:
            node (str): The patient node for which to determine the first treatment
                exposure time.

        Returns:
            pd.Timestamp: The timestamp of the first treatment exposure.

        Raises:
            AssertionError: If no treatment edge with a time attribute is found for the
                node, indicating an issue with the data or the specified treatments.

        This function iterates over all treatments and finds the earliest timestamp
        among them, ensuring that the analysis considers the initial treatment exposure.
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
    ) -> Tuple[Set[str], Set[str]]:
        """
        Identifies control groups among patients who did not undergo the specified
        treatments. Control groups are divided into those who had the outcome
        (control_true) and those who did not (control_false), based on the presence of
        specified outcome codes.

        Args:
            criteria_filter (List[str], optional): Criteria to filter patients for the
                control groups. Defaults to an empty list, meaning no additional
                filtering criteria.

        Returns:
            Tuple[Set[str], Set[str]]: Two sets representing the IDs of control
                patients. The first set includes patients who experienced the specified
                outcomes (control_true), and the second set includes those who did not
                (control_false).

        This method facilitates the separation of patients into relevant control groups
        for further analysis of treatment effects, ensuring only those not undergoing
        the treatment are considered for control.
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
        """
        Calculates the relative risk (RR) of an event occurring in the treatment group
        compared to the control group. RR is a key measure in epidemiological studies
        for estimating the likelihood of an event in one group relative to another.

        The interpretation of RR is as follows:
        - RR = 1 indicates no difference in risk between the two groups.
        - RR > 1 indicates a higher risk in the treatment group.
        - RR < 1 indicates a lower risk in the treatment group.

        Returns:
            float: The calculated relative risk between the treatment and control
                groups.

        Preconditions:
            - Groups must be sorted using the `find_groups` method.
            - Subject counts for each group must be non-zero to avoid division by zero
                errors.

        Raises:
            AssertionError: If the preconditions are not met, indicating a potential
                issue with group formation or subject count retrieval.
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
        """
        Calculates the odds ratio (OR) to quantify the association between exposure to a
        treatment and the occurrence of an outcome. OR compares the odds of an event
        occurring in the treatment group to the odds in the control group, providing
        insight into the strength of the association between the treatment and the
        outcome.

        Interpretation of the odds ratio:
        - OR = 1 indicates no difference in odds between the two groups.
        - OR > 1 suggests the event is more likely in the treatment group.
        - OR < 1 suggests the event is less likely in the treatment group.

        Returns:
            float: The calculated odds ratio between the treatment and control groups.

        Preconditions:
            - Groups must be sorted using the `find_groups` method.
            - Subject counts in each group must be non-zero to ensure valid
                calculations.

        Raises:
            AssertionError: If preconditions are not met, indicating potential issues
                with group formation or subject count retrieval.
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
        """
        Calculates the confounding bias (CB) to assess the impact of potential
        confounders on the observed association between treatment and outcome. A
        confounder is a variable that influences both the dependent (outcome) and
        independent (treatment) variables, potentially biasing the study results.

        Interpretation of CB:
        - CB = 1 indicates no confounding bias.
        - CB != 1 suggests the presence of confounding bias, indicating potential
            confounders.

        Returns:
            float: The calculated confounding bias.

        The method relies on the relative risk (RR) as an intermediary measure and
        adjusts the observed association for potential confounding effects. This
        adjustment helps in identifying whether the observed association might be
        influenced by factors other than the treatment.
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
