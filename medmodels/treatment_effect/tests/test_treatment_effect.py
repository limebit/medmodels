from __future__ import annotations

import unittest
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
import pytest

from medmodels import MedRecord
from medmodels.medrecord.querying import EdgeDirection, NodeOperand
from medmodels.treatment_effect.estimate import ContingencyTable, SubjectIndices
from medmodels.treatment_effect.treatment_effect import TreatmentEffect

if TYPE_CHECKING:
    from medmodels.medrecord.types import NodeIndex


def create_patients(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates a patients dataframe.

    Returns:
        pd.DataFrame: A patients dataframe.
    """
    patients = pd.DataFrame(
        {
            "index": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
            "age": [20, 30, 40, 30, 40, 50, 60, 70, 80],
            "gender": [
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
            ],
        }
    )

    return patients.loc[patients["index"].isin(patient_list)]


def create_diagnoses() -> pd.DataFrame:
    """Creates a diagnoses dataframe.

    Returns:
        pd.DataFrame: A diagnoses dataframe.
    """
    return pd.DataFrame(
        {
            "index": ["D1"],
            "name": ["Stroke"],
        }
    )


def create_prescriptions() -> pd.DataFrame:
    """Creates a prescriptions dataframe.

    Returns:
        pd.DataFrame: A prescriptions dataframe.
    """
    return pd.DataFrame(
        {
            "index": ["M1", "M2"],
            "name": ["Rivaroxaban", "Warfarin"],
        }
    )


def create_edges1(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates an edges dataframe.

    Returns:
        pd.DataFrame: An edges dataframe.
    """
    edges = pd.DataFrame(
        {
            "source": [
                "M2",
                "M1",
                "M2",
                "M1",
                "M2",
                "M1",
                "M2",
            ],
            "target": [
                "P1",
                "P2",
                "P2",
                "P3",
                "P5",
                "P6",
                "P9",
            ],
            "time": [
                datetime(1999, 10, 15),
                datetime(2000, 1, 1),
                datetime(1999, 12, 15),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
        }
    )
    return edges.loc[edges["target"].isin(patient_list)]


def create_edges2(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates an edges dataframe with attribute "intensity".

    Returns:
        pd.DataFrame: An edges dataframe.
    """
    edges = pd.DataFrame(
        {
            "source": [
                "D1",
                "D1",
                "D1",
                "D1",
                "D1",
                "D1",
            ],
            "target": [
                "P1",
                "P2",
                "P3",
                "P3",
                "P4",
                "P7",
            ],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 7, 1),
                datetime(1999, 12, 15),
                datetime(2000, 1, 5),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
            "intensity": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
            ],
        }
    )
    return edges.loc[edges["target"].isin(patient_list)]


def create_medrecord(
    patient_list: Optional[List[NodeIndex]] = None,
) -> MedRecord:
    """Creates a MedRecord object.

    Returns:
        MedRecord: A MedRecord object.
    """
    if patient_list is None:
        patient_list = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    patients = create_patients(patient_list=patient_list)
    diagnoses = create_diagnoses()
    prescriptions = create_prescriptions()
    edges1 = create_edges1(patient_list=patient_list)
    edges2 = create_edges2(patient_list=patient_list)
    medrecord = MedRecord.from_pandas(
        nodes=[(patients, "index"), (diagnoses, "index"), (prescriptions, "index")],
        edges=[(edges1, "source", "target")],
    )
    medrecord.add_group(group="patients", nodes=patients["index"].to_list())
    medrecord.add_group(
        "Stroke",
        ["D1"],
    )
    medrecord.add_group(
        "Rivaroxaban",
        ["M1"],
    )
    medrecord.add_group(
        "Warfarin",
        ["M2"],
    )
    medrecord.add_edges((edges2, "source", "target"))
    return medrecord


def assert_treatment_effects_equal(
    test_case: unittest.TestCase,
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


class TestTreatmentEffect(unittest.TestCase):
    """Class to test the TreatmentEffect class in the treatment_effect module."""

    def setUp(self) -> None:
        self.medrecord = create_medrecord()

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

        assert_treatment_effects_equal(self, tee, tee_builder)

    def test_default_properties(self) -> None:
        tee = TreatmentEffect(
            treatment="Rivaroxaban",
            outcome="Stroke",
        )

        tee_builder = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .with_patients_group("patients")
            .with_washout_period(reference="first")
            .with_grace_period(days=0, reference="last")
            .with_follow_up_period(365, reference="last")
            .build()
        )

        assert_treatment_effects_equal(self, tee, tee_builder)

    def test_check_medrecord(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Aspirin")
            .build()
        )

        with pytest.raises(
            ValueError, match="Treatment group not found in the MedRecord"
        ):
            tee.estimate._check_medrecord(medrecord=self.medrecord)

        tee2 = (
            TreatmentEffect.builder()
            .with_outcome("Headache")
            .with_treatment("Rivaroxaban")
            .build()
        )

        with pytest.raises(
            ValueError, match="Outcome group not found in the MedRecord"
        ):
            tee2.estimate._check_medrecord(medrecord=self.medrecord)

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
            tee3.estimate._check_medrecord(medrecord=self.medrecord)

    def test_find_treated_patients(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Rivaroxaban")
            .build()
        )

        treated_set = tee._find_treated_patients(self.medrecord)
        assert treated_set == set({"P2", "P3", "P6"})

        # no treatment_group
        patients = set(self.medrecord.nodes_in_group("patients"))
        medrecord2 = create_medrecord(list(patients - treated_set))

        with pytest.raises(
            ValueError,
            match="No patients found for the treatment group in this MedRecord",
        ):
            tee.estimate._compute_subject_counts(medrecord=medrecord2)

    def test_query_node_within_time_window(self) -> None:
        # check if patient has outcome a year after treatment
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Rivaroxaban")
            .build()
        )
        treated_set = tee._find_treated_patients(self.medrecord)

        nodes = self.medrecord.select_nodes(
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
        nodes = self.medrecord.select_nodes(
            lambda node: tee._query_node_within_time_window(
                node, treated_set, "Stroke", 0, 30, "last"
            )
        )
        assert "P3" in nodes
        assert (
            "P2" not in nodes
        )  # P2 has no outcome in the 30 days window after treatment

        # If we reduce the window to 3 days, no patients with outcome in that window
        nodes = self.medrecord.select_nodes(
            lambda node: tee._query_node_within_time_window(
                node, treated_set, "Stroke", 0, 3, "last"
            )
        )
        assert "P3" not in nodes
        assert "P2" not in nodes

    def test_find_groups(self) -> None:
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
        ) = tee._find_groups(self.medrecord)
        assert treatment_outcome_true == set({"P2", "P3"})
        assert treatment_outcome_false == set({"P6"})
        assert control_outcome_true == set({"P1", "P4", "P7"})
        assert control_outcome_false == set({"P5", "P8", "P9"})

    def test_compute_subject_counts(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        counts = tee.estimate._compute_subject_counts(self.medrecord)

        assert counts == (2, 1, 3, 3)

    def test_invalid_compute_subject_counts(self) -> None:
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
        ) = tee._find_groups(self.medrecord)
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

    def test_subject_counts(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        subjects_tee = tee.estimate.subject_counts(self.medrecord)
        assert isinstance(subjects_tee, ContingencyTable)
        assert subjects_tee["control_outcome_false"] == 3
        assert subjects_tee["control_outcome_true"] == 3
        assert subjects_tee["treated_outcome_false"] == 1
        assert subjects_tee["treated_outcome_true"] == 2

    def test_subjects_indices(self) -> None:
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
        subjects_tee = tee.estimate.subject_indices(self.medrecord)
        assert subjects_tee == subjects_test

    def test_follow_up_period(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_follow_up_period(30)
            .build()
        )

        assert tee._follow_up_period_days == 30

        counts_tee = tee.estimate._compute_subject_counts(self.medrecord)

        assert counts_tee == (1, 2, 3, 3)

    def test_grace_period(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_grace_period(10)
            .build()
        )

        assert tee._grace_period_days == 10

        counts_tee = tee.estimate._compute_subject_counts(self.medrecord)

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
                .build()
            )

    def test_washout_period(self) -> None:
        washout_dict = {"Warfarin": 30}

        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_washout_period(washout_dict)
            .build()
        )

        assert tee._washout_period_days == washout_dict

        treated_set = tee._find_treated_patients(self.medrecord)
        treated_set, washout_nodes = tee._apply_washout_period(
            self.medrecord, treated_set
        )

        assert treated_set == set({"P3", "P6"})
        assert washout_nodes == set({"P2"})

        # smaller washout period
        washout_dict2 = {"Warfarin": 10}

        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_washout_period(washout_dict2)
            .build()
        )

        assert tee2._washout_period_days == washout_dict2

        treated_set = tee2._find_treated_patients(self.medrecord)
        treated_set, washout_nodes = tee2._apply_washout_period(
            self.medrecord, treated_set
        )

        assert treated_set == set({"P2", "P3", "P6"})
        assert washout_nodes == set({})

    def test_outcome_before_treatment(self) -> None:
        # case 1 find outcomes for default tee
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        treated_set = tee._find_treated_patients(self.medrecord)
        treated_set, treatment_outcome_true, outcome_before_treatment_nodes = (
            tee._find_outcomes(self.medrecord, treated_set)
        )

        assert treated_set == set({"P2", "P3", "P6"})
        assert treatment_outcome_true == set({"P2", "P3"})
        assert outcome_before_treatment_nodes == set()

        # case 2 set exclusion time for outcome before treatment
        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_outcome_before_treatment_exclusion(30)
            .build()
        )

        assert tee2._outcome_before_treatment_days == 30

        treated_set = tee2._find_treated_patients(self.medrecord)
        treated_set, treatment_outcome_true, outcome_before_treatment_nodes = (
            tee2._find_outcomes(self.medrecord, treated_set)
        )

        assert treated_set == set({"P2", "P6"})
        assert treatment_outcome_true == set({"P2"})
        assert outcome_before_treatment_nodes == set({"P3"})

        # case 3 no outcome
        self.medrecord.add_group("Headache")

        tee3 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Headache")
            .with_outcome_before_treatment_exclusion(30)
            .build()
        )

        with pytest.raises(
            ValueError, match="No outcomes found in the MedRecord for group Headache"
        ):
            tee3._find_outcomes(medrecord=self.medrecord, treated_set=treated_set)

    def test_filter_controls(self) -> None:
        def query1(node: NodeOperand) -> None:
            node.neighbors(EdgeDirection.BOTH).index().equal_to("M2")

        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .filter_controls(query1)
            .build()
        )
        counts_tee = tee.estimate._compute_subject_counts(self.medrecord)

        assert counts_tee == (2, 1, 1, 2)

        # filter females only
        def query2(node: NodeOperand) -> None:
            node.attribute("gender").equal_to("female")

        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .filter_controls(query2)
            .build()
        )

        counts_tee2 = tee2.estimate._compute_subject_counts(self.medrecord)

        assert counts_tee2 == (2, 1, 1, 1)

    def test_nearest_neighbors(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_nearest_neighbors_matching()
            .build()
        )

        subjects = tee.estimate.subject_indices(self.medrecord)

        # Multiple patients are equally similar to the treatment group
        # These are exact macthes and should always be included
        assert "P4" in subjects["control_outcome_true"]
        assert "P5" in subjects["control_outcome_false"]
        assert "P8" in subjects["control_outcome_false"]

    def test_propensity_matching(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_propensity_matching()
            .build()
        )

        subjects = tee.estimate.subject_indices(self.medrecord)

        assert "P4" in subjects["control_outcome_true"]
        assert "P5" in subjects["control_outcome_false"]
        assert "P1" in subjects["control_outcome_true"]

    def test_find_controls(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        patients = set(self.medrecord.nodes_in_group("patients"))
        treated_set = {"P2", "P3", "P6"}

        control_outcome_true, control_outcome_false = tee._find_controls(
            self.medrecord,
            control_set=patients - treated_set,
            treated_set=patients.intersection(treated_set),
        )

        assert control_outcome_true == {"P1", "P4", "P7"}
        assert control_outcome_false == {"P5", "P8", "P9"}

        with pytest.raises(
            ValueError, match="No patients found for control groups in this MedRecord"
        ):
            tee._find_controls(
                self.medrecord,
                control_set=patients - treated_set,
                treated_set=patients.intersection(treated_set),
                rejected_nodes=patients - treated_set,
            )

        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Headache")
            .build()
        )

        self.medrecord.add_group("Headache")

        with pytest.raises(
            ValueError, match="No outcomes found in the MedRecord for group Headache"
        ):
            tee2._find_controls(
                self.medrecord,
                control_set=patients - treated_set,
                treated_set=patients.intersection(treated_set),
            )

    def test_metrics(self) -> None:
        """Test the metrics of the TreatmentEffect class."""
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        # Calculate metrics
        assert tee.estimate.absolute_risk_reduction(self.medrecord) == pytest.approx(
            -1 / 6
        )
        assert tee.estimate.relative_risk(self.medrecord) == pytest.approx(4 / 3)
        assert tee.estimate.odds_ratio(self.medrecord) == pytest.approx(2)
        assert tee.estimate.confounding_bias(self.medrecord) == pytest.approx(22 / 21)
        assert tee.estimate.hazard_ratio(self.medrecord) == pytest.approx(4 / 3)
        assert tee.estimate.number_needed_to_treat(self.medrecord) == pytest.approx(-6)

    def test_full_report(self) -> None:
        """Test the full reporting of the TreatmentEffect class."""
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        # Calculate metrics
        full_report = tee.report.full_report(self.medrecord)

        report_test = {
            "absolute_risk_reduction": tee.estimate.absolute_risk_reduction(
                self.medrecord
            ),
            "relative_risk": tee.estimate.relative_risk(self.medrecord),
            "odds_ratio": tee.estimate.odds_ratio(self.medrecord),
            "confounding_bias": tee.estimate.confounding_bias(self.medrecord),
            "hazard_ratio": tee.estimate.hazard_ratio(self.medrecord),
            "number_needed_to_treat": tee.estimate.number_needed_to_treat(
                self.medrecord
            ),
        }
        assert full_report == report_test

    def test_continuous_estimators_report(self) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        report_test = {
            "average_treatment_effect": tee.estimate.average_treatment_effect(
                self.medrecord,
                outcome_variable="intensity",
            ),
            "cohens_d": tee.estimate.cohens_d(
                self.medrecord, outcome_variable="intensity"
            ),
            "hedges_g": tee.estimate.hedges_g(
                self.medrecord, outcome_variable="intensity"
            ),
        }

        assert report_test == tee.report.continuous_estimators_report(
            self.medrecord, outcome_variable="intensity"
        )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestTreatmentEffect)
    unittest.TextTestRunner(verbosity=2).run(run_test)
