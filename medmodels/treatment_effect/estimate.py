"""Estimators class for calculating treatment effect metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Set, Tuple, TypedDict

from medmodels.treatment_effect.continuous_estimators import (
    average_treatment_effect,
    cohens_d,
    hedges_g,
)
from medmodels.treatment_effect.matching.neighbors import NeighborsMatching
from medmodels.treatment_effect.matching.propensity import PropensityMatching

if TYPE_CHECKING:
    from medmodels.medrecord.medrecord import MedRecord
    from medmodels.medrecord.types import MedRecordAttribute, NodeIndex
    from medmodels.treatment_effect.matching.matching import Matching
    from medmodels.treatment_effect.treatment_effect import TreatmentEffect


class SubjectIndices(TypedDict):
    """Dictionary with the patient ids of all contingency table groups."""

    treated_outcome_true: Set[NodeIndex]
    treated_outcome_false: Set[NodeIndex]
    control_outcome_true: Set[NodeIndex]
    control_outcome_false: Set[NodeIndex]


class ContingencyTable:
    """Contingency table for the treatment and control groups with/without outcome."""

    number_treated_outcome_true: int
    number_treated_outcome_false: int
    number_control_outcome_true: int
    number_control_outcome_false: int

    def __init__(
        self,
        number_treated_outcome_true: int,
        number_treated_outcome_false: int,
        number_control_outcome_true: int,
        number_control_outcome_false: int,
    ) -> None:
        """Initializes the ContingencyTable object.

        It stores the number of patients in the treatment and control groups with and
        without the outcome.

        Args:
            number_treated_outcome_true (int): Number of patients in the treatment
                group with the outcome.
            number_treated_outcome_false (int): Number of patients in the treatment
                group without the outcome.
            number_control_outcome_true (int): Number of patients in the control group
                with the outcome.
            number_control_outcome_false (int): Number of patients in the control group
                without the outcome.
        """
        self.number_treated_outcome_true = number_treated_outcome_true
        self.number_treated_outcome_false = number_treated_outcome_false
        self.number_control_outcome_true = number_control_outcome_true
        self.number_control_outcome_false = number_control_outcome_false

    def __str__(self) -> str:
        """Returns a string representation of the ContingencyTable object.

        The contingency table provides an overview of the number of subjects in the
        treatment and control groups with the outcome (true) and without the outcome
        (false).

        Example:
            .. code-block:: text

                -----------------------------------
                                Outcome
                Group           True     False
                -----------------------------------
                Treated         2        1
                Control         3        3
                -----------------------------------
        """
        line = "-" * 35
        upper_header_line = "{:<18} {:<10}".format("", "Outcome")
        lower_header_line = "{:<15} {:<8} {:<8}".format("Group", "True", "False")
        treated = "{:<15} {:<8} {:<8}".format(
            "Treated",
            self.number_treated_outcome_true,
            self.number_treated_outcome_false,
        )
        control = "{:<15} {:<8} {:<8}".format(
            "Control",
            self.number_control_outcome_true,
            self.number_control_outcome_false,
        )
        return f"{line}\n{upper_header_line}\n{lower_header_line}\n{line}\n{treated}\n{control}\n{line}"

    def __repr__(self) -> str:
        """Returns a string representation of the ContingencyTable object."""
        return self.__str__()

    def __getitem__(
        self,
        key: Literal[
            "treated_outcome_true",
            "treated_outcome_false",
            "control_outcome_true",
            "control_outcome_false",
        ],
    ) -> int:
        """Returns the number of subjects in the given group.

        Args:
            key (Literal["treated_outcome_true", "treated_outcome_false", "control_outcome_true", "control_outcome_false"]):
                The key to access the number of subjects in the treatment and control
                groups with and without the outcome.

        Returns:
            int: Number of subject in the selected group.
        """  # noqa: W505
        completed_key = "number_" + key
        return getattr(self, completed_key)


class Estimate:
    """Estimators class for calculating treatment effect metrics."""

    _treatment_effect: TreatmentEffect

    def __init__(self, treatment_effect: TreatmentEffect) -> None:
        """Initializes the Estimate object."""
        self._treatment_effect = treatment_effect

    def _check_medrecord(self, medrecord: MedRecord) -> None:
        """Checks if the required groups are present in the MedRecord.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Raises:
            ValueError: Raises Error if the required groups are not present in the
                MedRecord (patients, treatments, outcomes).
        """
        if self._treatment_effect._patients_group not in medrecord.groups:
            msg = (
                f"Patient group {self._treatment_effect._patients_group} not found in "
                f"the MedRecord. Available groups: {medrecord.groups}"
            )
            raise ValueError(msg)
        if self._treatment_effect._treatments_group not in medrecord.groups:
            msg = (
                "Treatment group not found in the MedRecord. "
                f"Available groups: {medrecord.groups}"
            )
            raise ValueError(msg)
        if self._treatment_effect._outcomes_group not in medrecord.groups:
            msg = (
                "Outcome group not found in the MedRecord."
                f"Available groups: {medrecord.groups}"
            )
            raise ValueError(msg)

    def _sort_subjects_in_groups(
        self, medrecord: MedRecord
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]]:
        """Sorts subjects into the different groups of the contingency table.

        This means, sorting the subjects into treatment-outcome, treatment-no outcome,
        control-outcome and control-no outcome.

        The treatment group and control matching is determined based on the treatment
        effect configuration.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex], Set[NodeIndex], Set[NodeIndex]: The
                patient ids of true and false subjects in the treatment and control
                groups, respectively.
        """
        self._check_medrecord(medrecord=medrecord)
        (
            treated_outcome_true,
            treated_outcome_false,
            control_outcome_true,
            control_outcome_false,
        ) = self._treatment_effect._find_groups(medrecord)
        treated_set = treated_outcome_true | treated_outcome_false

        if self._treatment_effect._matching_method:
            matching: Matching = (
                NeighborsMatching(
                    number_of_neighbors=self._treatment_effect._matching_number_of_neighbors,
                )
                if self._treatment_effect._matching_method == "nearest_neighbors"
                else PropensityMatching(
                    number_of_neighbors=self._treatment_effect._matching_number_of_neighbors,
                    model=self._treatment_effect._matching_model,
                    hyperparameters=self._treatment_effect._matching_hyperparameters,
                )
            )

            control_set = control_outcome_true | control_outcome_false

            matched_controls = matching.match_controls(
                medrecord=medrecord,
                treated_set=treated_set,
                control_set=control_set,
                patients_group=self._treatment_effect._patients_group,
                essential_covariates=self._treatment_effect._matching_essential_covariates,
                one_hot_covariates=self._treatment_effect._matching_one_hot_covariates,
            )
            control_outcome_true, control_outcome_false = (
                self._treatment_effect._find_controls(
                    medrecord=medrecord,
                    control_set=matched_controls,
                    treated_set=treated_set,
                )
            )

        return (
            treated_outcome_true,
            treated_outcome_false,
            control_outcome_true,
            control_outcome_false,
        )

    def _compute_subject_counts(
        self, medrecord: MedRecord
    ) -> Tuple[int, int, int, int]:
        """Computes the subject counts for the treatment and control groups.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            Tuple[int, int, int, int]: The number of true and false subjects in the
                treatment and control groups, respectively.

        Raises:
            ValueError: Raises error if the required groups are not present in the
                MedRecord (patients, treatments, outcomes).
            ValueError: If there are no subjects in the group of treated with no
                outcome, in the one of controls with outcome or in the one of controls
                with no outcome, an error is raised. This would result in division by
                zero errors.
        """
        (
            treated_outcome_true,
            treated_outcome_false,
            control_outcome_true,
            control_outcome_false,
        ) = self._sort_subjects_in_groups(medrecord=medrecord)

        if len(treated_outcome_false) == 0:
            msg = "No subjects found in the group of treated with no outcome"
            raise ValueError(msg)

        if len(control_outcome_true) == 0:
            msg = "No subjects found in the group of controls with outcome"
            raise ValueError(msg)

        if len(control_outcome_false) == 0:
            msg = "No subjects found in the group of controls with no outcome"
            raise ValueError(msg)

        return (
            len(treated_outcome_true),
            len(treated_outcome_false),
            len(control_outcome_true),
            len(control_outcome_false),
        )

    def subject_indices(self, medrecord: MedRecord) -> SubjectIndices:
        """Overview of which subjects are in which group from the contingency table.

        Returns a dictionary with the patient ids of all contingency table groups, i.e.,
        the treated group with and without the outcome, and the control group with and
        without the outcome.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            SubjectIndices: Dictionary with the patient ids of true and false subjects
                in the treatment and control groups, respectively.
        """
        (
            treated_outcome_true,
            treated_outcome_false,
            control_outcome_true,
            control_outcome_false,
        ) = self._sort_subjects_in_groups(medrecord=medrecord)

        return SubjectIndices(
            treated_outcome_true=treated_outcome_true,
            treated_outcome_false=treated_outcome_false,
            control_outcome_true=control_outcome_true,
            control_outcome_false=control_outcome_false,
        )

    def subject_counts(self, medrecord: MedRecord) -> ContingencyTable:
        """Overview of how many subjects are in which group from the contingency table.

        Returns a contingency table object with the number of subjects in the treatment
        and control groups with and without the outcome.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            ContingencyTable: The contingency table object containing the number of
                subjects in the treatment and control groups with and without the
                outcome.
        """
        (
            number_treated_outcome_true,
            number_treated_outcome_false,
            number_control_outcome_true,
            number_control_outcome_false,
        ) = self._compute_subject_counts(medrecord=medrecord)

        return ContingencyTable(
            number_treated_outcome_true=number_treated_outcome_true,
            number_treated_outcome_false=number_treated_outcome_false,
            number_control_outcome_true=number_control_outcome_true,
            number_control_outcome_false=number_control_outcome_false,
        )

    def relative_risk(self, medrecord: MedRecord) -> float:
        """Calculates the relative risk (RR) of an event.

        RR is a key measure in epidemiological studies for estimating the likelihood of
        an event in one group relative to another, in this case, the treatment group
        compared to the control group.

        The interpretation of RR is as follows:
        - RR = 1 indicates no difference in risk between the two groups.
        - RR > 1 indicates a higher risk in the treatment group.
        - RR < 1 indicates a lower risk in the treatment group.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            float: The calculated relative risk between the treatment and control
                groups.
        """
        (
            number_treated_outcome_true,
            number_treated_outcome_false,
            number_control_outcome_true,
            number_control_outcome_false,
        ) = self._compute_subject_counts(medrecord=medrecord)

        return (
            number_treated_outcome_true
            / (number_treated_outcome_true + number_treated_outcome_false)
        ) / (
            number_control_outcome_true
            / (number_control_outcome_true + number_control_outcome_false)
        )

    def odds_ratio(self, medrecord: MedRecord) -> float:
        """Calculates the odds ratio (OR).

        The OR quantifies the association between exposure to a treatment and the
        occurrence of an outcome.

        OR compares the odds of an event occurring in the treatment group to the odds
        in the control group, providing insight into the strength of the association
        between the treatment and the outcome.

        Interpretation of the odds ratio:
        - OR = 1 indicates no difference in odds between the two groups.
        - OR > 1 suggests the event is more likely in the treatment group.
        - OR < 1 suggests the event is less likely in the treatment group.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            float: The calculated odds ratio between the treatment and control groups.
        """
        (
            number_treated_outcome_true,
            number_treated_outcome_false,
            number_control_outcome_true,
            number_control_outcome_false,
        ) = self._compute_subject_counts(medrecord=medrecord)

        return (number_treated_outcome_true / number_control_outcome_true) / (
            number_treated_outcome_false / number_control_outcome_false
        )

    def confounding_bias(self, medrecord: MedRecord) -> float:
        """Calculates the confounding bias (CB).

        The CB is used to assess the impact of potential confounders on the observed
        association between treatment and outcome. A confounder is a variable that
        influences both the dependent (outcome) and independent (treatment) variables,
        potentially biasing the study results.

        Interpretation of CB:
        - CB = 1 indicates no confounding bias.
        - CB != 1 suggests the presence of confounding bias.

        The method relies on the relative risk (RR) as an intermediary measure and
        adjusts the observed association for potential confounding effects. This
        adjustment helps in identifying whether the observed association might be
        influenced by factors other than the treatment.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            float: The calculated confounding bias.
        """
        (
            number_treated_outcome_true,
            number_treated_outcome_false,
            number_control_outcome_true,
            number_control_outcome_false,
        ) = self._compute_subject_counts(medrecord=medrecord)
        relative_risk = self.relative_risk(medrecord)

        if relative_risk == 1:
            return 1.0

        multiplier = relative_risk - 1
        numerator = (
            number_treated_outcome_true
            / (number_treated_outcome_true + number_treated_outcome_false)
        ) * multiplier + 1
        denominator = (
            number_control_outcome_true
            / (number_control_outcome_true + number_control_outcome_false)
        ) * multiplier + 1

        return numerator / denominator

    def absolute_risk_reduction(self, medrecord: MedRecord) -> float:
        """Calculates the absolute risk reduction (ARR).

        AR (absolute risk) is a measure of the incidence of an event in each group.
        ARR, in turn, quantifies the difference in risk between the treatment and
        control groups. It is positive if the treatment reduces the risk, and negative
        if it increases the risk.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            float: The calculated absolute risk reduction between the treatment and
                control groups.
        """
        (
            number_treated_outcome_true,
            number_treated_outcome_false,
            number_control_outcome_true,
            number_control_outcome_false,
        ) = self._compute_subject_counts(medrecord=medrecord)

        ar_treated_group = number_treated_outcome_true / (
            number_treated_outcome_true + number_treated_outcome_false
        )
        ar_control_group = number_control_outcome_true / (
            number_control_outcome_true + number_control_outcome_false
        )

        return ar_control_group - ar_treated_group

    def number_needed_to_treat(self, medrecord: MedRecord) -> float:
        """Calculates the number needed to treat (NNT) to prevent one extra bad outcome.

        NNT is derived from the absolute risk reduction (ARR) and provides an estimate
        of the number of patients that need to be treated to prevent one additional bad
        outcome.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            float: The calculated number needed to treat between the treatment and
            control groups.

        Raises:
            ValueError: Raises Error if the required groups are not present in the
                MedRecord (patients, treatments, outcomes).
            ValueError: If there are no subjects in the group of treated with no
                outcome, in the one of controls with outcome or in the one of controls
                with no outcome, an error is raised. This would result in division by
                zero errors.
            ValueError: If the ARR is zero, cannot calculate NNT.
        """
        absolute_risk_reduction = self.absolute_risk_reduction(medrecord)
        if absolute_risk_reduction == 0:
            msg = "Absolute Risk Reduction is zero, cannot calculate NNT"
            raise ValueError(msg)

        return 1 / absolute_risk_reduction

    def hazard_ratio(self, medrecord: MedRecord) -> float:
        """Calculates the hazard ratio (HR).

        HR is used to compare the hazard rates of two groups in survival analysis.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.

        Returns:
            float: The calculated hazard ratio between the treatment and control groups.
        """
        (
            number_treated_outcome_true,
            number_treated_outcome_false,
            number_control_outcome_true,
            number_control_outcome_false,
        ) = self._compute_subject_counts(medrecord=medrecord)

        hazard_treat = number_treated_outcome_true / (
            number_treated_outcome_true + number_treated_outcome_false
        )
        hazard_control = number_control_outcome_true / (
            number_control_outcome_true + number_control_outcome_false
        )

        # hazard_control cannot be 0
        return hazard_treat / hazard_control

    def average_treatment_effect(
        self,
        medrecord: MedRecord,
        outcome_variable: MedRecordAttribute,
        reference: Literal["first", "last"] = "last",
    ) -> float:
        """Calculates the Average Treatment Effect (ATE).

        It is calculated as the difference between the outcome means of the treated and
        control sets. A positive ATE indicates that the treatment increased the outcome,
        while a negative ATE suggests a decrease.

        The ATE is computed as follows when the numbers of observations in treated and
        control sets are N and M, respectively:

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing medical
                data.
            outcome_variable (MedRecordAttribute): The attribute in the edge that
                contains the outcome variable. It must be numeric and continuous.
            reference (Literal["first", "last"], optional): The reference point for the
                exposure time. Options include "first" and "last". If "first", the
                function returns the earliest exposure edge. If "last", the function
                returns the latest exposure edge. Defaults to "last".

        Returns:
            float: The average treatment effect.
        """
        subjects = self.subject_indices(medrecord=medrecord)

        return average_treatment_effect(
            medrecord=medrecord,
            treatment_outcome_true_set=subjects.get("treated_outcome_true"),
            control_outcome_true_set=subjects.get("control_outcome_true"),
            outcome_group=self._treatment_effect._outcomes_group,
            outcome_variable=outcome_variable,
            reference=reference,
            time_attribute=self._treatment_effect._time_attribute,
        )

    def cohens_d(
        self,
        medrecord: MedRecord,
        outcome_variable: MedRecordAttribute,
        reference: Literal["first", "last"] = "last",
    ) -> float:
        """Calculates Cohen's D, the standardized mean difference between two sets.

        This measures the effect size of the difference between two outcome means.
        It's applicable for any two sets but is recommended for sets of the same size.
        Cohen's D indicates how many standard deviations the two groups differ by, with
        1 standard deviation equal to 1 z-score.

        A rule of thumb for interpreting Cohen's D:
        - Small effect = ±0.2
        - Medium effect = ±0.5
        - Large effect = ±0.8

        If the difference is negative, it indicates the mean in the treated group is
        lower than the control group.

        This metric provides a dimensionless measure of effect size, facilitating the
        comparison across different studies and contexts.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing medical
                data.
            outcome_variable (MedRecordAttribute): The attribute in the edge that
                contains the outcome variable. It must be numeric and continuous.
            reference (Literal["first", "last"], optional): The reference point for the
                exposure time. Options include "first" and "last". If "first", the
                function returns the earliest exposure edge. If "last", the function
                returns the latest exposure edge. Defaults to "last".
            add_correction (bool, optional): Whether to apply a correction factor for
                small sample sizes. Defaults to False.

        Returns:
            float: The Cohen's D coefficient, representing the effect size.
        """
        subjects = self.subject_indices(medrecord=medrecord)

        return cohens_d(
            medrecord=medrecord,
            treatment_outcome_true_set=subjects.get("treated_outcome_true"),
            control_outcome_true_set=subjects.get("control_outcome_true"),
            outcome_group=self._treatment_effect._outcomes_group,
            outcome_variable=outcome_variable,
            reference=reference,
            time_attribute=self._treatment_effect._time_attribute,
        )

    def hedges_g(
        self,
        medrecord: MedRecord,
        outcome_variable: MedRecordAttribute,
        reference: Literal["first", "last"] = "last",
    ) -> float:
        """Calculates Hedges' g, the unbiased effect size estimate.

        Hedges' g is a corrected version of Cohen's d that provides an unbiased estimate
        of the effect size, especially important when sample sizes are small (under 50).

        The correction factor is applied regardless of the sample size.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing medical
                data.
            outcome_variable (MedRecordAttribute): The attribute in the edge that
                contains the outcome variable. It must be numeric and continuous.
            reference (Literal["first", "last"], optional): The reference point for the
                exposure time. Options include "first" and "last". If "first", the
                function returns the earliest exposure edge. If "last", the function
                returns the latest exposure edge. Defaults to "last".
            add_correction (bool, optional): Whether to apply a correction factor for
                small sample sizes. Defaults to False.

        Returns:
            float: The Hedges' g coefficient, representing the effect size.
        """
        subjects = self.subject_indices(medrecord=medrecord)

        return hedges_g(
            medrecord=medrecord,
            treatment_outcome_true_set=subjects.get("treated_outcome_true"),
            control_outcome_true_set=subjects.get("control_outcome_true"),
            outcome_group=self._treatment_effect._outcomes_group,
            outcome_variable=outcome_variable,
            reference=reference,
            time_attribute=self._treatment_effect._time_attribute,
        )
