"""Test the Estimate class of the TreatmentEffect."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from medmodels.treatment_effect.estimate import ContingencyTable, SubjectIndices
from medmodels.treatment_effect.treatment_effect import TreatmentEffect
from tests.treatment_effect.helper import create_medrecord

if TYPE_CHECKING:
    from medmodels import MedRecord


@pytest.fixture
def medrecord() -> MedRecord:
    return create_medrecord()


class TestEstimate:
    def test_check_medrecord(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("no_treatment")
            .build()
        )

        with pytest.raises(
            ValueError, match="Treatment group not found in the MedRecord"
        ):
            tee.estimate._check_medrecord(medrecord=medrecord)

        tee2 = (
            TreatmentEffect.builder()
            .with_outcome("no_outcome")
            .with_treatment("Rivaroxaban")
            .build()
        )

        with pytest.raises(
            ValueError, match="Outcome group not found in the MedRecord"
        ):
            tee2.estimate._check_medrecord(medrecord=medrecord)

        patient_group = "subjects"
        tee3 = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Rivaroxaban")
            .with_patients_group(patient_group)
            .build()
        )

        with pytest.raises(
            ValueError,
            match=f"Patient group {patient_group} not found in the MedRecord",
        ):
            tee3.estimate._check_medrecord(medrecord=medrecord)

    def test_compute_subject_counts(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        counts = tee.estimate._compute_subject_counts(medrecord)

        assert counts == (2, 1, 3, 3)

    def test_invalid_compute_subject_counts(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        (
            treatment_outcome_true,
            treatment_outcome_false,
            control_outcome_true,
            control_outcome_false,
        ) = tee._find_groups(medrecord)
        all_patients = set().union(
            *[
                treatment_outcome_true,
                treatment_outcome_false,
                control_outcome_true,
                control_outcome_false,
            ]
        )

        medrecord2 = create_medrecord(
            patient_list=list(all_patients - control_outcome_false)
        )
        with pytest.raises(
            ValueError,
            match="No subjects found in the group of controls with no outcome",
        ):
            tee.estimate._compute_subject_counts(medrecord=medrecord2)

        medrecord3 = create_medrecord(
            patient_list=list(all_patients - treatment_outcome_false)
        )
        with pytest.raises(
            ValueError,
            match="No subjects found in the group of treated with no outcome",
        ):
            tee.estimate._compute_subject_counts(medrecord=medrecord3)

        medrecord4 = create_medrecord(
            patient_list=list(all_patients - control_outcome_true)
        )
        with pytest.raises(
            ValueError, match="No subjects found in the group of controls with outcome"
        ):
            tee.estimate._compute_subject_counts(medrecord=medrecord4)

    def test_subject_counts(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        subjects_tee = tee.estimate.subject_counts(medrecord)

        assert isinstance(subjects_tee, ContingencyTable)
        assert subjects_tee["control_outcome_false"] == 3
        assert subjects_tee["control_outcome_true"] == 3
        assert subjects_tee["treated_outcome_false"] == 1
        assert subjects_tee["treated_outcome_true"] == 2

        assert "\n".join(
            [
                "-----------------------------------",
                "                   Outcome   ",
                "Group           True     False   ",
                "-----------------------------------",
                "Treated         2        1       ",
                "Control         3        3       ",
                "-----------------------------------",
            ]
        ) == str(subjects_tee)

        assert repr(subjects_tee) == str(subjects_tee)

    def test_subjects_indices(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        subjects_test = SubjectIndices(
            treated_outcome_true={"P2", "P3"},
            treated_outcome_false={"P6"},
            control_outcome_true={"P1", "P4", "P7"},
            control_outcome_false={"P5", "P8", "P9"},
        )
        subjects_tee = tee.estimate.subject_indices(medrecord)
        assert subjects_tee == subjects_test

    def test_metrics(self, medrecord: MedRecord) -> None:
        """Test the metrics of the TreatmentEffect class."""
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        # Calculate metrics
        assert tee.estimate.absolute_risk_reduction(medrecord) == pytest.approx(-1 / 6)
        assert tee.estimate.relative_risk(medrecord) == pytest.approx(4 / 3)
        assert tee.estimate.odds_ratio(medrecord) == pytest.approx(2)
        assert tee.estimate.confounding_bias(medrecord) == pytest.approx(22 / 21)
        assert tee.estimate.hazard_ratio(medrecord) == pytest.approx(4 / 3)
        assert tee.estimate.number_needed_to_treat(medrecord) == pytest.approx(-6)

        medrecord = create_medrecord(["P2", "P3", "P4", "P5", "P6", "P7"])
        assert tee.estimate.confounding_bias(medrecord) == 1

    def test_invalid_metrics(self) -> None:
        """Test the invalid metrics of the TreatmentEffect class."""
        medrecord = create_medrecord(["P2", "P3", "P4", "P5", "P6", "P7"])
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        with pytest.raises(
            ValueError, match="Absolute Risk Reduction is zero, cannot calculate NNT"
        ):
            tee.estimate.number_needed_to_treat(medrecord)

    def test_nearest_neighbors(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_nearest_neighbors_matching()
            .build()
        )

        subjects = tee.estimate.subject_indices(medrecord)

        # Multiple patients are equally similar to the treatment group
        # These are exact macthes and should always be included
        assert "P4" in subjects["control_outcome_true"]
        assert "P5" in subjects["control_outcome_false"]
        assert "P8" in subjects["control_outcome_false"]

    def test_propensity_matching(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_propensity_matching()
            .build()
        )

        subjects = tee.estimate.subject_indices(medrecord)

        assert "P4" in subjects["control_outcome_true"]
        assert "P5" in subjects["control_outcome_false"]
        assert "P1" in subjects["control_outcome_true"]

    def test_continuous_estimators(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        average_treatment_effect = tee.estimate.average_treatment_effect(
            medrecord,
            outcome_variable="intensity",
        )
        cohens_d = tee.estimate.cohens_d(
            medrecord,
            outcome_variable="intensity",
        )
        hedges_g = tee.estimate.hedges_g(
            medrecord,
            outcome_variable="intensity",
        )

        assert average_treatment_effect == pytest.approx(-0.1)
        assert cohens_d == pytest.approx(-0.5)
        assert hedges_g == pytest.approx(-0.4)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
