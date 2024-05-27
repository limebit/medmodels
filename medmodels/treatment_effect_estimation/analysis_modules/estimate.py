from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect


class Estimate:
    _treatment_effect: TreatmentEffect

    def __init__(self, treatment_effect: TreatmentEffect) -> None:
        self._treatment_effect = treatment_effect

    def _compute_subject_counts(self) -> tuple[int, int, int, int]:
        """Computes the subject counts for the treatment and control groups.

        Returns:
            tuple[int, int, int, int]: The number of true and false subjects in the
                treatment and control groups, respectively.
        """
        if self._treatment_effect.groups_sorted is False:
            self._treatment_effect._find_groups()
        (
            num_treat_true,
            num_treat_false,
            num_control_true,
            num_control_false,
        ) = self._treatment_effect._subject_counts

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
        ) = self._compute_subject_counts()

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
        ) = self._compute_subject_counts()

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

        Precondition:
            - Subject counts in each group must be non-zero to avoid division by zero
                errors.

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
        ) = self._compute_subject_counts()
        relative_risk = self.relative_risk()

        if relative_risk == 1:
            return 1

        multiplier = relative_risk - 1
        numerator = (
            num_treat_true / (num_treat_true + num_treat_false)
        ) * multiplier + 1
        denominator = (
            num_control_true / (num_control_true + num_control_false)
        ) * multiplier + 1

        return numerator / denominator

    def absolute_risk(self) -> float:
        """
        Calculates the absolute risk (AR) of an event occurring in the treatment group
        compared to the control group. AR is a measure of the incidence of an event in
        each group.

        Returns:
            float: The calculated absolute risk difference between the treatment and control
                groups.

        Preconditions:
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
        ) = self._compute_subject_counts()

        risk_treat = num_treat_true / (num_treat_true + num_treat_false)
        risk_control = num_control_true / (num_control_true + num_control_false)

        return risk_treat - risk_control

    def number_needed_to_treat(self) -> float:
        """
        Calculates the number needed to treat (NNT) to prevent one additional bad outcome.
        NNT is derived from the absolute risk reduction.

        Returns:
            float: The calculated number needed to treat between the treatment and control
                groups.

        Preconditions:
            - Subject counts for each group must be non-zero to avoid division by zero
                errors.

        Raises:
            AssertionError: If the preconditions are not met, indicating a potential
                issue with group formation or subject count retrieval.
        """
        ar = self.absolute_risk()
        if ar == 0:
            raise ValueError("Absolute risk is zero, cannot calculate NNT.")
        return 1 / ar

    def hazard_ratio(self) -> float:
        """
        Calculates the hazard ratio (HR) for the treatment group compared to the control
        group. HR is used to compare the hazard rates of two groups in survival analysis.

        Returns:
            float: The calculated hazard ratio between the treatment and control groups.

        Preconditions:
            - Hazard rates for each group must be calculable.

        Raises:
            AssertionError: If the preconditions are not met, indicating a potential
                issue with group formation or hazard rate calculation.
        """
        (
            num_treat_true,
            num_treat_false,
            num_control_true,
            num_control_false,
        ) = self._compute_subject_counts()

        hazard_treat = num_treat_true / (num_treat_true + num_treat_false)
        hazard_control = num_control_true / (num_control_true + num_control_false)

        if hazard_control == 0:
            raise ValueError(
                "Control hazard rate is zero, cannot calculate hazard ratio."
            )

        return hazard_treat / hazard_control
