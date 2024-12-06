"""Tests for the TreatmentEffect class in the treatment_effect module."""

import unittest
from datetime import datetime
from typing import List

import pandas as pd

from medmodels import MedRecord
from medmodels.medrecord.querying import EdgeDirection, NodeOperand
from medmodels.medrecord.types import NodeIndex
from medmodels.treatment_effect.estimate import ContingencyTable, SubjectIndices
from medmodels.treatment_effect.treatment_effect import TreatmentEffect


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

    patients = patients.loc[patients["index"].isin(patient_list)]
    return patients


def create_diagnoses() -> pd.DataFrame:
    """Creates a diagnoses dataframe.

    Returns:
        pd.DataFrame: A diagnoses dataframe.
    """
    diagnoses = pd.DataFrame(
        {
            "index": ["D1"],
            "name": ["Stroke"],
        }
    )
    return diagnoses


def create_prescriptions() -> pd.DataFrame:
    """Creates a prescriptions dataframe.

    Returns:
        pd.DataFrame: A prescriptions dataframe.
    """
    prescriptions = pd.DataFrame(
        {
            "index": ["M1", "M2"],
            "name": ["Rivaroxaban", "Warfarin"],
        }
    )
    return prescriptions


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
    edges = edges.loc[edges["target"].isin(patient_list)]
    return edges


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
    edges = edges.loc[edges["target"].isin(patient_list)]
    return edges


def create_medrecord(
    patient_list: List[NodeIndex] = [
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
    ],
) -> MedRecord:
    """Creates a MedRecord object.

    Returns:
        MedRecord: A MedRecord object.
    """
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
):
    test_case.assertEqual(
        treatment_effect1._treatments_group, treatment_effect2._treatments_group
    )
    test_case.assertEqual(
        treatment_effect1._outcomes_group, treatment_effect2._outcomes_group
    )
    test_case.assertEqual(
        treatment_effect1._patients_group, treatment_effect2._patients_group
    )
    test_case.assertEqual(
        treatment_effect1._time_attribute, treatment_effect2._time_attribute
    )
    test_case.assertEqual(
        treatment_effect1._washout_period_days, treatment_effect2._washout_period_days
    )
    test_case.assertEqual(
        treatment_effect1._washout_period_reference,
        treatment_effect2._washout_period_reference,
    )
    test_case.assertEqual(
        treatment_effect1._grace_period_days, treatment_effect2._grace_period_days
    )
    test_case.assertEqual(
        treatment_effect1._grace_period_reference,
        treatment_effect2._grace_period_reference,
    )
    test_case.assertEqual(
        treatment_effect1._follow_up_period_days,
        treatment_effect2._follow_up_period_days,
    )
    test_case.assertEqual(
        treatment_effect1._follow_up_period_reference,
        treatment_effect2._follow_up_period_reference,
    )
    test_case.assertEqual(
        treatment_effect1._outcome_before_treatment_days,
        treatment_effect2._outcome_before_treatment_days,
    )
    test_case.assertEqual(
        treatment_effect1._filter_controls_query,
        treatment_effect2._filter_controls_query,
    )
    test_case.assertEqual(
        treatment_effect1._matching_method, treatment_effect2._matching_method
    )
    test_case.assertEqual(
        treatment_effect1._matching_essential_covariates,
        treatment_effect2._matching_essential_covariates,
    )
    test_case.assertEqual(
        treatment_effect1._matching_one_hot_covariates,
        treatment_effect2._matching_one_hot_covariates,
    )
    test_case.assertEqual(
        treatment_effect1._matching_model, treatment_effect2._matching_model
    )
    test_case.assertEqual(
        treatment_effect1._matching_number_of_neighbors,
        treatment_effect2._matching_number_of_neighbors,
    )
    test_case.assertEqual(
        treatment_effect1._matching_hyperparam, treatment_effect2._matching_hyperparam
    )


class TestTreatmentEffect(unittest.TestCase):
    """Class to test the TreatmentEffect class in the treatment_effect module."""

    def setUp(self):
        self.medrecord = create_medrecord()

    def test_init(self):
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

    def test_default_properties(self):
        tee = TreatmentEffect(
            treatment="Rivaroxaban",
            outcome="Stroke",
        )

        tee_builder = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute(None)
            .with_patients_group("patients")
            .with_washout_period(reference="first")
            .with_grace_period(days=0, reference="last")
            .with_follow_up_period(365000, reference="last")
            .build()
        )

        assert_treatment_effects_equal(self, tee, tee_builder)

    def test_time_warnings(self):
        """Test the warnings raised by the TreatmentEffect class with no time attribute."""
        with self.assertLogs(level="WARNING") as log_capture:
            _ = (
                TreatmentEffect.builder()
                .with_treatment("Rivaroxaban")
                .with_outcome("Stroke")
                .with_washout_period({"Warfarin": 30})
                .build()
            )

        self.assertIn(
            "Washout period is not applied because the time attribute is not set.",
            log_capture.output[0],
        )

        with self.assertLogs(level="WARNING") as log_capture:
            _ = (
                TreatmentEffect.builder()
                .with_treatment("Rivaroxaban")
                .with_outcome("Stroke")
                .with_follow_up_period(365)
                .build()
            )

        self.assertIn(
            "Time attribute is not set, thus the grace period, follow-up "
            + "period, and outcome before treatment cannot be applied. The "
            + "treatment effect analysis is performed in a static way.",
            log_capture.output[0],
        )

    def test_check_medrecord(self):
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Aspirin")
            .build()
        )

        with self.assertRaisesRegex(
            ValueError, "Treatment group not found in the MedRecord"
        ):
            tee.estimate._check_medrecord(medrecord=self.medrecord)

        tee2 = (
            TreatmentEffect.builder()
            .with_outcome("Headache")
            .with_treatment("Rivaroxaban")
            .build()
        )

        with self.assertRaisesRegex(
            ValueError, "Outcome group not found in the MedRecord"
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

        with self.assertRaisesRegex(
            ValueError, f"Patient group {patient_group} not found in the MedRecord"
        ):
            tee3.estimate._check_medrecord(medrecord=self.medrecord)

    def test_find_treated_patients(self):
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Rivaroxaban")
            .build()
        )

        treated_set = tee._find_treated_patients(self.medrecord)
        self.assertEqual(treated_set, set({"P2", "P3", "P6"}))

        # no treatment_group
        patients = set(self.medrecord.nodes_in_group("patients"))
        medrecord2 = create_medrecord(list(patients - treated_set))

        with self.assertRaisesRegex(
            ValueError, "No patients found for the treatment groups in this MedRecord."
        ):
            tee.estimate._compute_subject_counts(medrecord=medrecord2)

    def test_query_node_within_time_window(self):
        # check if patient has outcome a year after treatment
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Rivaroxaban")
            .with_time_attribute("time")
            .build()
        )
        treated_set = tee._find_treated_patients(self.medrecord)

        nodes = self.medrecord.select_nodes(
            lambda node: tee._query_node_within_time_window(
                node, treated_set, "Stroke", 0, 365, "last"
            )
        )
        self.assertIn("P3", nodes)
        self.assertIn("P2", nodes)

        # Only one not having an outcome in that time period (no outcome at all)
        self.assertNotIn("P6", nodes)
        self.assertIn("P6", treated_set)

        # check which patients have outcome within 30 days after treatment
        nodes = self.medrecord.select_nodes(
            lambda node: tee._query_node_within_time_window(
                node, treated_set, "Stroke", 0, 30, "last"
            )
        )
        self.assertIn("P3", nodes)
        self.assertNotIn(
            "P2", nodes
        )  # P2 has no outcome in the 30 days window after treatment

        # If we reduce the window to 3 days, we have no patients with outcome in that window
        nodes = self.medrecord.select_nodes(
            lambda node: tee._query_node_within_time_window(
                node, treated_set, "Stroke", 0, 3, "last"
            )
        )
        self.assertNotIn("P3", nodes)
        self.assertNotIn("P2", nodes)

    def test_find_groups(self):
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
        self.assertEqual(treatment_outcome_true, set({"P2", "P3"}))
        self.assertEqual(treatment_outcome_false, set({"P6"}))
        self.assertEqual(control_outcome_true, set({"P1", "P4", "P7"}))
        self.assertEqual(control_outcome_false, set({"P5", "P8", "P9"}))

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
        ) = tee._find_groups(self.medrecord)
        self.assertEqual(treatment_outcome_true, set({"P2", "P3"}))
        self.assertEqual(treatment_outcome_false, set({"P6"}))
        self.assertEqual(control_outcome_true, set({"P1", "P4", "P7"}))
        self.assertEqual(control_outcome_false, set({"P5", "P8", "P9"}))

    def test_compute_subject_counts(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        counts = tee.estimate._compute_subject_counts(self.medrecord)

        self.assertEqual(counts, (2, 1, 3, 3))

    def test_invalid_compute_subject_counts(self):
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
        with self.assertRaisesRegex(
            ValueError, "No subjects found in the group of controls with no outcome"
        ):
            tee.estimate._compute_subject_counts(medrecord=medrecord2)

        medrecord3 = create_medrecord(
            patient_list=list(all_patients - treatment_outcome_false)
        )
        with self.assertRaisesRegex(
            ValueError, "No subjects found in the group of treated with no outcome"
        ):
            tee.estimate._compute_subject_counts(medrecord=medrecord3)

        medrecord4 = create_medrecord(
            patient_list=list(all_patients - control_outcome_true)
        )
        with self.assertRaisesRegex(
            ValueError, "No subjects found in the group of controls with outcome"
        ):
            tee.estimate._compute_subject_counts(medrecord=medrecord4)

    def test_subject_counts(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        subjects_tee = tee.estimate.subject_counts(self.medrecord)
        self.assertEqual(3, subjects_tee["control_outcome_false"])
        self.assertEqual(3, subjects_tee["control_outcome_true"])
        self.assertEqual(1, subjects_tee["treated_outcome_false"])
        self.assertEqual(2, subjects_tee["treated_outcome_true"])
        self.assertIsInstance(subjects_tee, ContingencyTable)

    def test_subjects_indices(self):
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
        self.assertEqual(
            subjects_test["control_outcome_false"],
            subjects_tee["control_outcome_false"],
        )
        self.assertEqual(
            subjects_test["control_outcome_true"],
            subjects_tee["control_outcome_true"],
        )
        self.assertEqual(
            subjects_test["treated_outcome_false"],
            subjects_tee["treated_outcome_false"],
        )
        self.assertEqual(
            subjects_test["treated_outcome_true"],
            subjects_tee["treated_outcome_true"],
        )

    def test_follow_up_period(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .with_follow_up_period(30)
            .build()
        )

        self.assertEqual(tee._follow_up_period_days, 30)

        counts_tee = tee.estimate._compute_subject_counts(self.medrecord)

        self.assertEqual((1, 2, 3, 3), counts_tee)

    def test_grace_period(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_grace_period(10)
            .with_time_attribute("time")
            .build()
        )

        self.assertEqual(tee._grace_period_days, 10)

        counts_tee = tee.estimate._compute_subject_counts(self.medrecord)

        self.assertEqual((1, 2, 3, 3), counts_tee)

    def test_invalid_grace_period(self):
        with self.assertRaisesRegex(
            ValueError,
            "The follow-up period must be greater than or equal to the grace period.",
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

    def test_washout_period(self):
        washout_dict = {"Warfarin": 30}

        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_washout_period(washout_dict)
            .with_time_attribute("time")
            .build()
        )

        self.assertDictEqual(tee._washout_period_days, washout_dict)

        treated_set = tee._find_treated_patients(self.medrecord)
        with self.assertLogs(level="WARNING") as log_capture:
            treated_set, washout_nodes = tee._apply_washout_period(
                self.medrecord, treated_set
            )

        self.assertEqual(treated_set, set({"P3", "P6"}))
        self.assertEqual(washout_nodes, set({"P2"}))
        self.assertIn(
            "1 subject was dropped due to having a treatment in the washout period.",
            log_capture.output[0],
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

        self.assertDictEqual(tee2._washout_period_days, washout_dict2)

        treated_set = tee2._find_treated_patients(self.medrecord)
        treated_set, washout_nodes = tee2._apply_washout_period(
            self.medrecord, treated_set
        )

        self.assertEqual(treated_set, set({"P2", "P3", "P6"}))
        self.assertEqual(washout_nodes, set({}))

    def test_outcome_before_treatment(self):
        # case 1 find outcomes for default tee
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .build()
        )
        treated_set = tee._find_treated_patients(self.medrecord)
        treated_set, treatment_outcome_true, outcome_before_treatment_nodes = (
            tee._find_outcomes(self.medrecord, treated_set)
        )
        self.assertEqual(treated_set, set({"P2", "P3", "P6"}))
        self.assertEqual(treatment_outcome_true, set({"P2", "P3"}))
        self.assertEqual(outcome_before_treatment_nodes, set())

        # case 2 set exclusion time for outcome before treatment
        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .with_outcome_before_treatment_exclusion(30)
            .build()
        )

        self.assertEqual(tee2._outcome_before_treatment_days, 30)

        treated_set = tee2._find_treated_patients(self.medrecord)
        with self.assertLogs(level="WARNING") as log_capture:
            treated_set, treatment_outcome_true, outcome_before_treatment_nodes = (
                tee2._find_outcomes(self.medrecord, treated_set)
            )

        self.assertEqual(treated_set, set({"P2", "P6"}))
        self.assertEqual(treatment_outcome_true, set({"P2"}))
        self.assertEqual(outcome_before_treatment_nodes, set({"P3"}))

        self.assertIn(
            "1 subject was dropped due to having an outcome before the treatment.",
            log_capture.output[0],
        )

        # case 3 no outcome
        self.medrecord.add_group("no_outcome")

        tee3 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("no_outcome")
            .with_time_attribute("time")
            .with_outcome_before_treatment_exclusion(30)
            .build()
        )

        with self.assertRaisesRegex(
            ValueError, "No outcomes found in the MedRecord for group "
        ):
            tee3._find_outcomes(medrecord=self.medrecord, treated_set=treated_set)

    def test_filter_controls(self):
        def query_neighbors_to_m2(node: NodeOperand):
            node.neighbors(EdgeDirection.BOTH).index().equal_to("M2")

        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .filter_controls(query_neighbors_to_m2)
            .build()
        )
        counts_tee = tee.estimate._compute_subject_counts(self.medrecord)

        self.assertEqual(counts_tee, (2, 1, 1, 2))

        # filter females only
        def query_female_patients(node: NodeOperand):
            node.attribute("gender").equal_to("female")

        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .filter_controls(query_female_patients)
            .build()
        )

        counts_tee2 = tee2.estimate._compute_subject_counts(self.medrecord)

        self.assertEqual(counts_tee2, (2, 1, 1, 1))

    def test_nearest_neighbors(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_nearest_neighbors_matching()
            .build()
        )

        subjects = tee.estimate.subject_indices(self.medrecord)

        # multiple patients are equally similar to the treatment group
        # these are exact macthes and should always be included
        self.assertIn("P4", subjects["control_outcome_true"])
        self.assertIn("P5", subjects["control_outcome_false"])
        self.assertIn("P8", subjects["control_outcome_false"])

    def test_propensity_matching(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_propensity_matching()
            .build()
        )

        subjects = tee.estimate.subject_indices(self.medrecord)

        self.assertIn("P4", subjects["control_outcome_true"])
        self.assertIn("P5", subjects["control_outcome_false"])
        self.assertIn("P1", subjects["control_outcome_true"])

    def test_find_controls(self):
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
        self.assertEqual(control_outcome_true, {"P1", "P4", "P7"})
        self.assertEqual(control_outcome_false, {"P5", "P8", "P9"})

        with self.assertRaisesRegex(
            ValueError, "No patients found for control groups in this MedRecord."
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

        with self.assertRaisesRegex(
            ValueError, "No outcomes found in the MedRecord for group."
        ):
            tee2._find_controls(
                self.medrecord,
                control_set=patients - treated_set,
                treated_set=patients.intersection(treated_set),
            )

    def test_metrics(self):
        """Test the metrics of the TreatmentEffect class."""
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        # Calculate metrics
        self.assertAlmostEqual(
            tee.estimate.absolute_risk_reduction(self.medrecord), -1 / 6
        )
        self.assertAlmostEqual(tee.estimate.relative_risk(self.medrecord), 4 / 3)
        self.assertAlmostEqual(tee.estimate.odds_ratio(self.medrecord), 2)
        self.assertAlmostEqual(tee.estimate.confounding_bias(self.medrecord), 22 / 21)
        self.assertAlmostEqual(tee.estimate.hazard_ratio(self.medrecord), 4 / 3)
        self.assertAlmostEqual(tee.estimate.number_needed_to_treat(self.medrecord), -6)

    def test_full_report(self):
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
        self.assertDictEqual(report_test, full_report)

    def test_continuous_estimators_report(self):
        """Test the continuous report of the TreatmentEffect."""
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
        }

        self.assertDictEqual(
            report_test,
            tee.report.continuous_estimators_report(
                self.medrecord, outcome_variable="intensity"
            ),
        )

    def test_continuous_estimators_report_with_time(self):
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
                self.medrecord,
                outcome_variable="intensity",
            ),
            "cohens_d": tee.estimate.cohens_d(
                self.medrecord, outcome_variable="intensity"
            ),
        }

        self.assertDictEqual(
            report_test,
            tee.report.continuous_estimators_report(
                self.medrecord, outcome_variable="intensity"
            ),
        )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestTreatmentEffect)
    unittest.TextTestRunner(verbosity=2).run(run_test)
