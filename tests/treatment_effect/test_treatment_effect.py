from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from medmodels.medrecord.querying import EdgeDirection, NodeIndicesOperand, NodeOperand
from medmodels.treatment_effect.treatment_effect import TreatmentEffect
from tests.treatment_effect.helper import create_medrecord

if TYPE_CHECKING:
    from medmodels import MedRecord


@pytest.fixture
def medrecord() -> MedRecord:
    return create_medrecord()


def assert_treatment_effects_equal(
    treatment_effect1: TreatmentEffect,
    treatment_effect2: TreatmentEffect,
) -> None:
    assert treatment_effect1._treatments_group == treatment_effect2._treatments_group
    assert treatment_effect1._outcomes_group == treatment_effect2._outcomes_group
    assert treatment_effect1._patients_group == treatment_effect2._patients_group
    assert treatment_effect1._time_attribute == treatment_effect2._time_attribute
    assert (
        treatment_effect1._washout_period_days == treatment_effect2._washout_period_days
    )
    assert (
        treatment_effect1._washout_period_reference
        == treatment_effect2._washout_period_reference
    )
    assert treatment_effect1._grace_period_days == treatment_effect2._grace_period_days
    assert (
        treatment_effect1._grace_period_reference
        == treatment_effect2._grace_period_reference
    )
    assert (
        treatment_effect1._follow_up_period_days
        == treatment_effect2._follow_up_period_days
    )
    assert (
        treatment_effect1._follow_up_period_reference
        == treatment_effect2._follow_up_period_reference
    )
    assert (
        treatment_effect1._outcome_before_treatment_days
        == treatment_effect2._outcome_before_treatment_days
    )
    assert (
        treatment_effect1._filter_controls_query
        == treatment_effect2._filter_controls_query
    )
    assert treatment_effect1._matching_method == treatment_effect2._matching_method
    assert (
        treatment_effect1._matching_essential_covariates
        == treatment_effect2._matching_essential_covariates
    )
    assert (
        treatment_effect1._matching_one_hot_covariates
        == treatment_effect2._matching_one_hot_covariates
    )
    assert treatment_effect1._matching_model == treatment_effect2._matching_model
    assert (
        treatment_effect1._matching_number_of_neighbors
        == treatment_effect2._matching_number_of_neighbors
    )
    assert (
        treatment_effect1._outcome_before_treatment_days
        == treatment_effect2._outcome_before_treatment_days
    )
    assert (
        treatment_effect1._filter_controls_query
        == treatment_effect2._filter_controls_query
    )
    assert treatment_effect1._matching_method == treatment_effect2._matching_method
    assert (
        treatment_effect1._matching_essential_covariates
        == treatment_effect2._matching_essential_covariates
    )
    assert (
        treatment_effect1._matching_one_hot_covariates
        == treatment_effect2._matching_one_hot_covariates
    )
    assert treatment_effect1._matching_model == treatment_effect2._matching_model
    assert (
        treatment_effect1._matching_number_of_neighbors
        == treatment_effect2._matching_number_of_neighbors
    )
    assert (
        treatment_effect1._matching_hyperparameters
        == treatment_effect2._matching_hyperparameters
    )


class TestTreatmentEffect:
    """Class to test the TreatmentEffect class in the treatment_effect module."""

    def test_init(self) -> None:
        # Initialize TreatmentEffect object
        tee = TreatmentEffect(
            treatment="Rivaroxaban",
            outcome="Stroke",
        )

        tee_builder = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        assert_treatment_effects_equal(tee, tee_builder)

    def test_default_properties(self) -> None:
        tee = TreatmentEffect(
            treatment="Rivaroxaban",
            outcome="Stroke",
        )

        tee_builder = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_patients_group("patients")
            .with_washout_period(reference="first")
            .with_grace_period(days=0, reference="last")
            .with_follow_up_period(365000, reference="last")
            .build()
        )

        assert_treatment_effects_equal(tee, tee_builder)

    def test_time_warnings_washout(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _ = (
                TreatmentEffect.builder()
                .with_treatment("Rivaroxaban")
                .with_outcome("Stroke")
                .with_washout_period({"Warfarin": 30})
                .build()
            )

        assert (
            "Washout period is not applied because the time attribute is not set."
            in caplog.records[0].message
        )

    def test_time_warnings_follow_up(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _ = (
                TreatmentEffect.builder()
                .with_treatment("Rivaroxaban")
                .with_outcome("Stroke")
                .with_follow_up_period(365)
                .build()
            )

        assert (
            "Time attribute is not set, thus the grace period, follow-up "
            + "period, and outcome before treatment cannot be applied. The "
            + "treatment effect analysis is performed in a static way."
        ) in caplog.records[0].message

    def test_query_node_within_time_window(self, medrecord: MedRecord) -> None:
        # check if patient has outcome a year after treatment
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Rivaroxaban")
            .with_time_attribute("time")
            .build()
        )
        treated_set = tee._find_treated_patients(medrecord)

        nodes = medrecord.query_nodes(
            lambda node: tee._query_node_within_time_window(
                node, treated_set, "Stroke", 0, 365, "last"
            )
        )
        assert "P3" in nodes
        assert "P2" in nodes

        # Only one not having an outcome in that time period (no outcome at all)
        assert "P6" not in nodes
        assert "P6" in treated_set

        # check which patients have outcome within 30 days after treatment
        nodes = medrecord.query_nodes(
            lambda node: tee._query_node_within_time_window(
                node, treated_set, "Stroke", 0, 30, "last"
            )
        )
        assert "P3" in nodes
        assert (
            "P2" not in nodes
        )  # P2 has no outcome in the 30 days window after treatment

        # If we reduce the window to 3 days, no patients with outcome in that window
        nodes = medrecord.query_nodes(
            lambda node: tee._query_node_within_time_window(
                node, treated_set, "Stroke", 0, 3, "last"
            )
        )
        assert "P3" not in nodes
        assert "P2" not in nodes

    def test_find_treated_patients(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Rivaroxaban")
            .build()
        )

        treated_set = tee._find_treated_patients(medrecord)
        assert treated_set == set({"P2", "P3", "P6"})

        # no treatment_group
        patients = set(medrecord.nodes_in_group("patients"))
        medrecord2 = create_medrecord(list(patients - treated_set))

        with pytest.raises(
            ValueError,
            match="No patients found for the treatment group in this MedRecord",
        ):
            tee._find_treated_patients(medrecord=medrecord2)

    def test_follow_up_period(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .with_follow_up_period(30)
            .build()
        )

        assert tee._follow_up_period_days == 30

        counts_tee = tee.estimate._compute_subject_counts(medrecord)

        assert counts_tee == (1, 2, 3, 3)

    def test_find_groups(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Rivaroxaban")
            .build()
        )

        (
            treatment_outcome_true,
            treatment_outcome_false,
            control_outcome_true,
            control_outcome_false,
        ) = tee._find_groups(medrecord)

        assert treatment_outcome_true == set({"P2", "P3"})
        assert treatment_outcome_false == set({"P6"})
        assert control_outcome_true == set({"P1", "P4", "P7"})
        assert control_outcome_false == set({"P5", "P8", "P9"})

        # for this scenario, it works the same in temporal and static analysis
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .build()
        )
        (
            treatment_outcome_true,
            treatment_outcome_false,
            control_outcome_true,
            control_outcome_false,
        ) = tee._find_groups(medrecord)

        assert treatment_outcome_true == set({"P2", "P3"})
        assert treatment_outcome_false == set({"P6"})
        assert control_outcome_true == set({"P1", "P4", "P7"})
        assert control_outcome_false == set({"P5", "P8", "P9"})

        # for this scenario, it works the same in temporal and static analysis
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .build()
        )
        (
            treatment_outcome_true,
            treatment_outcome_false,
            control_outcome_true,
            control_outcome_false,
        ) = tee._find_groups(medrecord)

        assert treatment_outcome_true == set({"P2", "P3"})
        assert treatment_outcome_false == set({"P6"})
        assert control_outcome_true == set({"P1", "P4", "P7"})
        assert control_outcome_false == set({"P5", "P8", "P9"})

    def test_grace_period(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_grace_period(10)
            .with_time_attribute("time")
            .build()
        )

        assert tee._grace_period_days == 10

        counts_tee = tee.estimate._compute_subject_counts(medrecord)

        assert counts_tee == (1, 2, 3, 3)

    def test_invalid_grace_period(self) -> None:
        with pytest.raises(
            ValueError,
            match="The follow-up period must be greater than or equal to the grace period",
        ):
            (
                TreatmentEffect.builder()
                .with_treatment("Rivaroxaban")
                .with_outcome("Stroke")
                .with_grace_period(1000)
                .with_follow_up_period(365)
                .with_time_attribute("time")
                .build()
            )

    def test_washout_period(
        self, caplog: pytest.LogCaptureFixture, medrecord: MedRecord
    ) -> None:
        washout_dict = {"Warfarin": 30}

        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_washout_period(washout_dict)
            .with_time_attribute("time")
            .build()
        )

        assert tee._washout_period_days == washout_dict

        treated_set = tee._find_treated_patients(medrecord)
        with caplog.at_level(logging.WARNING):
            treated_set, washout_nodes = tee._apply_washout_period(
                medrecord, treated_set
            )

        assert treated_set == set({"P3", "P6"})
        assert washout_nodes == set({"P2"})
        assert (
            "1 subject was dropped due to having a treatment in the washout period."
            in caplog.records[0].message
        )

        # smaller washout period
        washout_dict2 = {"Warfarin": 10}

        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_washout_period(washout_dict2)
            .with_time_attribute("time")
            .build()
        )

        assert tee2._washout_period_days == washout_dict2

        treated_set = tee2._find_treated_patients(medrecord)
        treated_set, washout_nodes = tee2._apply_washout_period(medrecord, treated_set)

        assert treated_set == set({"P2", "P3", "P6"})
        assert washout_nodes == set({})

    def test_outcome_before_treatment(
        self, caplog: pytest.LogCaptureFixture, medrecord: MedRecord
    ) -> None:
        # case 1 find outcomes for default tee
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .build()
        )
        treated_set = tee._find_treated_patients(medrecord)
        treated_set, treatment_outcome_true, outcome_before_treatment_nodes = (
            tee._find_outcomes(medrecord, treated_set)
        )

        assert treated_set == set({"P2", "P3", "P6"})
        assert treatment_outcome_true == set({"P2", "P3"})
        assert outcome_before_treatment_nodes == set()

        # case 2 set exclusion time for outcome before treatment
        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .with_outcome_before_treatment_exclusion(30)
            .build()
        )

        assert tee2._outcome_before_treatment_days == 30

        treated_set = tee2._find_treated_patients(medrecord)
        with caplog.at_level(logging.WARNING):
            treated_set, treatment_outcome_true, outcome_before_treatment_nodes = (
                tee2._find_outcomes(medrecord, treated_set)
            )

        assert treated_set == set({"P2", "P6"})
        assert treatment_outcome_true == set({"P2"})
        assert outcome_before_treatment_nodes == set({"P3"})

        assert (
            "1 subject was dropped due to having an outcome before the treatment."
            in caplog.records[0].message
        )

        # case 3 no outcome
        medrecord.unfreeze_schema()
        medrecord.add_group("no_outcome")

        tee3 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("no_outcome")
            .with_time_attribute("time")
            .with_outcome_before_treatment_exclusion(30)
            .build()
        )

        with pytest.raises(
            ValueError, match="No outcomes found in the MedRecord for group no_outcome"
        ):
            tee3._find_outcomes(medrecord=medrecord, treated_set=treated_set)

    def test_filter_controls(self, medrecord: MedRecord) -> None:
        def query_neighbors_to_m2(node: NodeOperand) -> NodeIndicesOperand:
            node.neighbors(EdgeDirection.BOTH).index().equal_to("M2")

            return node.index()

        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .filter_controls(query_neighbors_to_m2)
            .build()
        )
        counts_tee = tee.estimate._compute_subject_counts(medrecord)

        assert counts_tee == (2, 1, 1, 2)

        # filter females only
        def query_female_patients(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("gender").equal_to("female")

            return node.index()

        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .filter_controls(query_female_patients)
            .build()
        )

        counts_tee2 = tee2.estimate._compute_subject_counts(medrecord)

        assert counts_tee2 == (2, 1, 1, 1)

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

    def test_find_controls(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        patients = set(medrecord.nodes_in_group("patients"))
        treated_set = {"P2", "P3", "P6"}

        control_outcome_true, control_outcome_false = tee._find_controls(
            medrecord,
            control_set=patients - treated_set,
            treated_set=patients.intersection(treated_set),
        )

        assert control_outcome_true == {"P1", "P4", "P7"}
        assert control_outcome_false == {"P5", "P8", "P9"}

        with pytest.raises(
            ValueError, match="No patients found for control groups in this MedRecord"
        ):
            tee._find_controls(
                medrecord,
                control_set=patients - treated_set,
                treated_set=patients.intersection(treated_set),
                rejected_nodes=patients - treated_set,
            )

        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("no_outcome")
            .build()
        )

        medrecord.unfreeze_schema()
        medrecord.add_group("no_outcome")

        with pytest.raises(
            ValueError, match="No outcomes found in the MedRecord for group no_outcome"
        ):
            tee2._find_controls(
                medrecord,
                control_set=patients - treated_set,
                treated_set=patients.intersection(treated_set),
            )

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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
