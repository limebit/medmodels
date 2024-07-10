from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from medmodels.medrecord.medrecord import MedRecord

if TYPE_CHECKING:
    from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect


class FullReport(TypedDict):
    relative_risk: float
    odds_ratio: float
    confounding_bias: float
    absolute_risk: float
    number_needed_to_treat: float
    hazard_ratio: float


class Report:
    _treatment_effect: TreatmentEffect

    def __init__(self, treatment_effect: TreatmentEffect) -> None:
        self._treatment_effect = treatment_effect

    def full_report(self, medrecord: MedRecord) -> FullReport:
        """
        Generates a full report of the treatment effect estimation.

        Returns:
            FullReport: A dictionary containing the results of all estimation
                methods: relative risk, odds ratio, confounding bias, absolute risk,
                number needed to treat, and hazard ratio.
        """
        return {
            "relative_risk": self._treatment_effect.estimate.relative_risk(medrecord),
            "odds_ratio": self._treatment_effect.estimate.odds_ratio(medrecord),
            "confounding_bias": self._treatment_effect.estimate.confounding_bias(
                medrecord
            ),
            "absolute_risk": self._treatment_effect.estimate.absolute_risk(medrecord),
            "number_needed_to_treat": self._treatment_effect.estimate.number_needed_to_treat(
                medrecord
            ),
            "hazard_ratio": self._treatment_effect.estimate.hazard_ratio(medrecord),
        }
