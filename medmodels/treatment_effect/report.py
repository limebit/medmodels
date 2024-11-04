"""This module contains functions to generate reports of the treatment effect class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from medmodels.medrecord.medrecord import MedRecord
    from medmodels.medrecord.types import MedRecordAttribute
    from medmodels.treatment_effect.treatment_effect import TreatmentEffect


class FullReport(TypedDict):
    """A dictionary containing the results of all estimation methods."""

    relative_risk: float
    odds_ratio: float
    confounding_bias: float
    absolute_risk_reduction: float
    number_needed_to_treat: float
    hazard_ratio: float


class ContinuousReport(TypedDict):
    """A dictionary containing the results of continuous treatment effect estimators."""

    average_treatment_effect: float
    cohens_d: float
    hedges_g: float


class Report:
    """Class to generate reports of the treatment effect class."""

    _treatment_effect: TreatmentEffect

    def __init__(self, treatment_effect: TreatmentEffect) -> None:
        """Initializes the Report class.

        Args:
            treatment_effect (TreatmentEffect): An instance of the TreatmentEffect
                class.
        """
        self._treatment_effect = treatment_effect

    def full_report(self, medrecord: MedRecord) -> FullReport:
        """Generates a full report of the treatment effect estimation.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing medical
                data.

        Returns:
            FullReport: A dictionary containing the results of all estimation
                methods: relative risk, odds ratio, confounding bias, absolute risk
                reduction, number needed to treat, and hazard ratio.
        """
        return {
            "relative_risk": self._treatment_effect.estimate.relative_risk(medrecord),
            "odds_ratio": self._treatment_effect.estimate.odds_ratio(medrecord),
            "confounding_bias": self._treatment_effect.estimate.confounding_bias(
                medrecord
            ),
            "absolute_risk_reduction": self._treatment_effect.estimate.absolute_risk_reduction(
                medrecord
            ),
            "number_needed_to_treat": self._treatment_effect.estimate.number_needed_to_treat(
                medrecord
            ),
            "hazard_ratio": self._treatment_effect.estimate.hazard_ratio(medrecord),
        }

    def continuous_estimators_report(
        self,
        medrecord: MedRecord,
        outcome_variable: MedRecordAttribute,
        reference: Literal["first", "last"] = "last",
    ) -> ContinuousReport:
        """Generates a report of continuous treatment effect estimators.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing medical
                data.
            outcome_variable (MedRecordAttribute): The attribute in the edge that
                contains the outcome variable.
            reference (Literal["first", "last"], optional): The reference point for the
                exposure time. Options include "first" and "last". If "first", the
                function returns the earliest exposure edge. If "last", the function
                returns the latest exposure edge. Defaults to "last".

        Returns:
            ContinuousReport: A dictionary containing the results of continuous
                treatment effect estimators: average treatment effect, Cohen's d and
                Hedges' g.
        """
        average_treatment_effect = (
            self._treatment_effect.estimate.average_treatment_effect(
                medrecord,
                outcome_variable,
                reference=reference,
            )
        )
        cohens_d_value = self._treatment_effect.estimate.cohens_d(
            medrecord,
            outcome_variable,
            reference=reference,
        )
        hedges_g_value = self._treatment_effect.estimate.hedges_g(
            medrecord,
            outcome_variable,
            reference=reference,
        )

        return {
            "average_treatment_effect": average_treatment_effect,
            "cohens_d": cohens_d_value,
            "hedges_g": hedges_g_value,
        }
