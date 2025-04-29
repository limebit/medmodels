"""Test the TreatmentEffect report class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from medmodels.treatment_effect.treatment_effect import TreatmentEffect
from tests.treatment_effect.helper import create_medrecord

if TYPE_CHECKING:
    from medmodels import MedRecord


@pytest.fixture
def medrecord() -> MedRecord:
    return create_medrecord()


class TestTreatmentEffectReport:
    def test_full_report(self, medrecord: MedRecord) -> None:
        """Test the full reporting of the TreatmentEffect class."""
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        # Calculate metrics
        full_report = tee.report.full_report(medrecord)

        report_test = {
            "absolute_risk_reduction": tee.estimate.absolute_risk_reduction(medrecord),
            "relative_risk": tee.estimate.relative_risk(medrecord),
            "odds_ratio": tee.estimate.odds_ratio(medrecord),
            "confounding_bias": tee.estimate.confounding_bias(medrecord),
            "hazard_ratio": tee.estimate.hazard_ratio(medrecord),
            "number_needed_to_treat": tee.estimate.number_needed_to_treat(medrecord),
        }
        assert full_report == report_test

    def test_continuous_estimators_report(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        report_test = {
            "average_treatment_effect": tee.estimate.average_treatment_effect(
                medrecord,
                outcome_variable="intensity",
            ),
            "cohens_d": tee.estimate.cohens_d(medrecord, outcome_variable="intensity"),
            "hedges_g": tee.estimate.hedges_g(medrecord, outcome_variable="intensity"),
        }

        assert report_test == tee.report.continuous_estimators_report(
            medrecord, outcome_variable="intensity"
        )

    def test_continuous_estimators_report_with_time(self, medrecord: MedRecord) -> None:
        """Test the continuous report of the TreatmentEffect with time attribute."""
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .build()
        )

        report_test = {
            "average_treatment_effect": tee.estimate.average_treatment_effect(
                medrecord,
                outcome_variable="intensity",
            ),
            "cohens_d": tee.estimate.cohens_d(medrecord, outcome_variable="intensity"),
            "hedges_g": tee.estimate.hedges_g(medrecord, outcome_variable="intensity"),
        }

        assert report_test == tee.report.continuous_estimators_report(
            medrecord, outcome_variable="intensity"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
