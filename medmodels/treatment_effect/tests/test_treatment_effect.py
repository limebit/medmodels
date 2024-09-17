"""Tests for the TreatmentEffect class in the treatment_effect module."""

import unittest
from typing import List

import pandas as pd

from medmodels import MedRecord
from medmodels.medrecord import edge, node
from medmodels.medrecord.types import NodeIndex
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
                "1999-10-15",
                "2000-01-01",
                "1999-12-15",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
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
                "2000-01-01",
                "2000-07-01",
                "1999-12-15",
                "2000-01-05",
                "2000-01-01",
                "2000-01-01",
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
        treatment_effect1._filter_controls_operation,
        treatment_effect2._filter_controls_operation,
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
            .with_time_attribute("time")
            .with_patients_group("patients")
            .with_washout_period(reference="first")
            .with_grace_period(days=0, reference="last")
            .with_follow_up_period(365, reference="last")
            .build()
        )

        assert_treatment_effects_equal(self, tee, tee_builder)

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

        treated_group = tee._find_treated_patients(self.medrecord)
        self.assertEqual(treated_group, set({"P2", "P3", "P6"}))

        # no treatment_group
        patients = set(self.medrecord.nodes_in_group("patients"))
        medrecord2 = create_medrecord(list(patients - treated_group))

        with self.assertRaisesRegex(
            ValueError, "No patients found for the treatment groups in this MedRecord."
        ):
            tee.estimate.subject_counts(medrecord=medrecord2)

    def test_find_groups(self):
        tee = (
            TreatmentEffect.builder()
            .with_outcome("Stroke")
            .with_treatment("Rivaroxaban")
            .build()
        )

        treatment_true, treatment_false, control_true, control_false = tee._find_groups(
            self.medrecord
        )
        self.assertEqual(treatment_true, set({"P2", "P3"}))
        self.assertEqual(treatment_false, set({"P6"}))
        self.assertEqual(control_true, set({"P1", "P4", "P7"}))
        self.assertEqual(control_false, set({"P5", "P8", "P9"}))

    def test_compute_subject_counts(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        counts = tee.estimate._compute_subject_counts(self.medrecord)

        self.assertEqual(counts, (2, 1, 3, 3))

        # test value errors if no subjects are found
        treatment_true, treatment_false, control_true, control_false = tee._find_groups(
            self.medrecord
        )
        all_patients = set().union(
            *[treatment_true, treatment_false, control_true, control_false]
        )

        medrecord2 = create_medrecord(patient_list=list(all_patients - control_false))
        with self.assertRaisesRegex(
            ValueError, "No subjects found in the control false group"
        ):
            tee.estimate.subject_counts(medrecord=medrecord2)

        medrecord3 = create_medrecord(patient_list=list(all_patients - treatment_false))
        with self.assertRaisesRegex(
            ValueError, "No subjects found in the treatment false group"
        ):
            tee.estimate.subject_counts(medrecord=medrecord3)

        medrecord4 = create_medrecord(patient_list=list(all_patients - control_true))
        with self.assertRaisesRegex(
            ValueError, "No subjects found in the control true group"
        ):
            tee.estimate.subject_counts(medrecord=medrecord4)

    def test_subject_counts(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        counts_tee = tee.estimate.subject_counts(medrecord=self.medrecord)
        counts_test = {
            "treatment_true": 2,
            "treatment_false": 1,
            "control_true": 3,
            "control_false": 3,
        }

        self.assertDictEqual(counts_tee, counts_test)

    def test_subjects_contigency_table(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        subjects_test = {
            "treatment_true": {"P2", "P3"},
            "treatment_false": {"P6"},
            "control_true": {"P1", "P4", "P7"},
            "control_false": {"P5", "P8", "P9"},
        }
        subjects_tee = tee.estimate.subjects_contingency_table(self.medrecord)
        self.assertDictEqual(subjects_test, subjects_tee)

    def test_follow_up_period(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_follow_up_period(30)
            .build()
        )

        self.assertEqual(tee._follow_up_period_days, 30)

        counts_test = {
            "treatment_true": 1,
            "treatment_false": 2,
            "control_true": 3,
            "control_false": 3,
        }
        counts_tee = tee.estimate.subject_counts(self.medrecord)

        self.assertDictEqual(counts_tee, counts_test)

    def test_grace_period(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_grace_period(10)
            .build()
        )

        self.assertEqual(tee._grace_period_days, 10)

        counts_test = {
            "treatment_true": 1,
            "treatment_false": 2,
            "control_true": 3,
            "control_false": 3,
        }
        counts_tee = tee.estimate.subject_counts(self.medrecord)

        self.assertDictEqual(counts_tee, counts_test)

    def test_washout_period(self):
        washout_dict = {"Warfarin": 30}

        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_washout_period(washout_dict)
            .build()
        )

        self.assertDictEqual(tee._washout_period_days, washout_dict)

        treated_group = tee._find_treated_patients(self.medrecord)
        treated_group, washout_nodes = tee._apply_washout_period(
            self.medrecord, treated_group
        )

        self.assertEqual(treated_group, set({"P3", "P6"}))
        self.assertEqual(washout_nodes, set({"P2"}))

        # smaller washout period
        washout_dict2 = {"Warfarin": 10}

        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_washout_period(washout_dict2)
            .build()
        )

        self.assertDictEqual(tee2._washout_period_days, washout_dict2)

        treated_group = tee2._find_treated_patients(self.medrecord)
        treated_group, washout_nodes = tee2._apply_washout_period(
            self.medrecord, treated_group
        )

        self.assertEqual(treated_group, set({"P2", "P3", "P6"}))
        self.assertEqual(washout_nodes, set({}))

    def test_outcome_before_treatment(self):
        # case 1 find outcomes for default tee
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )
        treated_group = tee._find_treated_patients(self.medrecord)
        treated_group, treatment_true, outcome_before_treatment_nodes = (
            tee._find_outcomes(self.medrecord, treated_group)
        )
        self.assertEqual(treated_group, set({"P2", "P3", "P6"}))
        self.assertEqual(treatment_true, set({"P2", "P3"}))
        self.assertEqual(outcome_before_treatment_nodes, set())

        # case 2 set exclusion time for outcome before treatment
        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_outcome_before_treatment_exclusion(30)
            .build()
        )

        self.assertEqual(tee2._outcome_before_treatment_days, 30)

        treated_group = tee2._find_treated_patients(self.medrecord)
        treated_group, treatment_true, outcome_before_treatment_nodes = (
            tee2._find_outcomes(self.medrecord, treated_group)
        )
        self.assertEqual(treated_group, set({"P2", "P6"}))
        self.assertEqual(treatment_true, set({"P2"}))
        self.assertEqual(outcome_before_treatment_nodes, set({"P3"}))

        # case 3 no outcome

        self.medrecord.add_group("Headache")

        tee3 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Headache")
            .with_outcome_before_treatment_exclusion(30)
            .build()
        )

        with self.assertRaisesRegex(
            ValueError, "No outcomes found in the MedRecord for group "
        ):
            tee3._find_outcomes(medrecord=self.medrecord, treated_group=treated_group)

    def test_filter_controls(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .filter_controls(
                node().has_outgoing_edge_with(edge().connected_target("M2"))
                | node().has_incoming_edge_with(edge().connected_source("M2"))
            )
            .build()
        )
        counts_test = {
            "treatment_true": 2,
            "treatment_false": 1,
            "control_true": 1,
            "control_false": 2,
        }
        counts_tee = tee.estimate.subject_counts(self.medrecord)

        self.assertDictEqual(counts_tee, counts_test)

        # filter females only

        tee2 = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .filter_controls(node().attribute("gender").equal("female"))
            .build()
        )

        counts_test2 = {
            "treatment_true": 2,
            "treatment_false": 1,
            "control_true": 1,
            "control_false": 1,
        }
        counts_tee2 = tee2.estimate.subject_counts(self.medrecord)

        self.assertDictEqual(counts_tee2, counts_test2)

    def test_nearest_neighbors(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_nearest_neighbors_matching()
            .build()
        )

        subjects = tee.estimate.subjects_contingency_table(self.medrecord)

        # multiple patients are equally similar to the treatment group
        # these are exact macthes and should always be included
        self.assertIn("P4", subjects["control_true"])
        self.assertIn("P5", subjects["control_false"])
        self.assertIn("P8", subjects["control_false"])

    def test_propensity_matching(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_propensity_matching()
            .build()
        )

        subjects = tee.estimate.subjects_contingency_table(self.medrecord)

        self.assertIn("P4", subjects["control_true"])
        self.assertIn("P5", subjects["control_false"])
        self.assertIn("P1", subjects["control_true"])

    def test_find_controls(self):
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        patients = set(self.medrecord.nodes_in_group("patients"))
        treated_group = {"P2", "P3", "P6"}

        control_true, control_false = tee._find_controls(
            self.medrecord,
            control_group=patients - treated_group,
            treated_group=patients.intersection(treated_group),
        )
        self.assertEqual(control_true, {"P1", "P4", "P7"})
        self.assertEqual(control_false, {"P5", "P8", "P9"})

        with self.assertRaisesRegex(
            ValueError, "No patients found for control groups in this MedRecord."
        ):
            tee._find_controls(
                self.medrecord,
                control_group=patients - treated_group,
                treated_group=patients.intersection(treated_group),
                rejected_nodes=patients - treated_group,
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
                control_group=patients - treated_group,
                treated_group=patients.intersection(treated_group),
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
        self.assertAlmostEqual(tee.estimate.risk_difference(self.medrecord), 1 / 6)
        self.assertAlmostEqual(tee.estimate.relative_risk(self.medrecord), 4 / 3)
        self.assertAlmostEqual(tee.estimate.odds_ratio(self.medrecord), 2)
        self.assertAlmostEqual(tee.estimate.confounding_bias(self.medrecord), 22 / 21)
        self.assertAlmostEqual(tee.estimate.hazard_ratio(self.medrecord), 4 / 3)
        self.assertAlmostEqual(tee.estimate.number_needed_to_treat(self.medrecord), 6)

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
            "risk_difference": tee.estimate.risk_difference(self.medrecord),
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
        """Test the continuous report of the TreatmentEffect class."""
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


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestTreatmentEffect)
    unittest.TextTestRunner(verbosity=2).run(run_test)
