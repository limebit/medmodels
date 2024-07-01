"""Tests for the TreatmentEffect class in the treatment_effect module."""

import unittest

import pandas as pd

from medmodels import MedRecord
from medmodels.medrecord import edge, node
from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect


def create_patients() -> pd.DataFrame:
    """
    Create a patients dataframe.

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
    return patients


def create_diagnoses() -> pd.DataFrame:
    """
    Create a diagnoses dataframe.

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
    """
    Create a prescriptions dataframe.

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


def create_edges() -> pd.DataFrame:
    """
    Create an edges dataframe.

    Returns:
        pd.DataFrame: An edges dataframe.
    """
    edges = pd.DataFrame(
        {
            "source": [
                "D1",
                "M2",
                "M1",
                "M2",
                "D1",
                "D1",
                "D1",
                "M1",
                "D1",
                "M2",
                "M1",
                "D1",
                "M2",
            ],
            "target": [
                "P1",
                "P1",
                "P2",
                "P2",
                "P2",
                "P3",
                "P3",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P9",
            ],
            "time": [
                "2000-01-01",
                "1999-10-15",
                "2000-01-01",
                "1999-12-15",
                "2000-07-01",
                "1999-12-15",
                "2000-01-05",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
            ],
        }
    )
    return edges


def create_medrecord() -> MedRecord:
    """
    Create a MedRecord object.

    Returns:
        MedRecord: A MedRecord object.
    """
    patients = create_patients()
    diagnoses = create_diagnoses()
    prescriptions = create_prescriptions()
    edges = create_edges()
    medrecord = MedRecord.from_pandas(
        nodes=[(patients, "index"), (diagnoses, "index"), (prescriptions, "index")],
        edges=[(edges, "source", "target")],
    )
    medrecord.add_group(group="patients", node=patients["index"].to_list())
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
        treatment_effect1._matching_distance_metric,
        treatment_effect2._matching_distance_metric,
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
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .finish()
        )

        assert_treatment_effects_equal(self, tee, tee_builder)

    def test_default_properties(self):
        tee = TreatmentEffect(
            treatment="Rivaroxaban",
            outcome="Stroke",
        )

        tee_builder = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .set_time_attribute("time")
            .set_patients_group("patients")
            .set_washout_period(reference="first")
            .set_grace_period(days=0, reference="last")
            .set_follow_up_period(365, reference="last")
            .finish()
        )

        assert_treatment_effects_equal(self, tee, tee_builder)

    def test_check_medrecord(self):
        tee = (
            TreatmentEffect.builder()
            .set_outcome("Stroke")
            .set_treatment("Aspirin")
            .finish()
        )

        with self.assertRaises(AssertionError) as context:
            tee.estimate.subject_counts(medrecord=self.medrecord)
            self.assertTrue(
                "Treatment group not found in the data." in str(context.exception)
            )

        tee2 = (
            TreatmentEffect.builder()
            .set_outcome("Headache")
            .set_treatment("Rivaroxaban")
            .finish()
        )

        with self.assertRaises(AssertionError) as context:
            tee2.estimate.subject_counts(medrecord=self.medrecord)
            self.assertTrue(
                "Outcome group not found in the data." in str(context.exception)
            )

    def test_find_treated_patients(self):
        tee = (
            TreatmentEffect.builder()
            .set_outcome("Stroke")
            .set_treatment("Rivaroxaban")
            .finish()
        )

        treated_group = tee._find_treated_patients(self.medrecord)
        self.assertEqual(treated_group, set({"P2", "P3", "P6"}))

    def test_find_groups(self):
        tee = (
            TreatmentEffect.builder()
            .set_outcome("Stroke")
            .set_treatment("Rivaroxaban")
            .finish()
        )

        treatment_true, treatment_false, control_true, control_false = tee._find_groups(
            self.medrecord
        )
        self.assertEqual(treatment_true, set({"P2", "P3"}))
        self.assertEqual(treatment_false, set({"P6"}))
        self.assertEqual(control_true, set({"P1", "P4", "P7"}))
        self.assertEqual(control_false, set({"P5", "P8", "P9"}))

    def test_find_reference_time(self):
        tee = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .finish()
        )
        time = tee._find_reference_time(
            self.medrecord, node_index="P6", reference="last"
        )
        self.assertEqual(pd.Timestamp("2000-01-01"), time)

        # adding medication time
        self.medrecord.add_edge(
            source_node="M1", target_node="P6", attributes={"time": "2000-01-15"}
        )

        time = tee._find_reference_time(
            self.medrecord, node_index="P6", reference="last"
        )
        self.assertEqual(pd.Timestamp("2000-01-15"), time)

    def test_node_in_time_window(self):
        tee = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .finish()
        )
        # check if patient has outcome a year after treatment
        node_found = tee._find_node_in_time_window(
            self.medrecord,
            node_index="P2",
            event_node="D1",
            start_days=0,
            end_days=365,
            reference="last",
        )
        self.assertTrue(node_found)

        # check if patient has outcome 30 days after treatment
        node_found = tee._find_node_in_time_window(
            self.medrecord,
            node_index="P2",
            event_node="D1",
            start_days=0,
            end_days=30,
            reference="last",
        )
        self.assertFalse(node_found)

    def test_subject_counts(self):
        tee = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .finish()
        )
        counts_tee = tee.estimate.subject_counts(medrecord=self.medrecord)
        counts_test = {
            "treatment_true": 2,
            "treatment_false": 1,
            "control_true": 3,
            "control_false": 3,
        }

        self.assertDictEqual(counts_tee, counts_test)

    def test_subjects_treatment_control(self):
        tee = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .finish()
        )

        subjects_test = {
            "treatment_true": {"P2", "P3"},
            "treatment_false": {"P6"},
            "control_true": {"P1", "P4", "P7"},
            "control_false": {"P5", "P8", "P9"},
        }
        subjects_tee = tee.estimate.subjects_treatment_control(self.medrecord)
        self.assertDictEqual(subjects_test, subjects_tee)

    def test_follow_up_period(self):
        tee = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .set_follow_up_period(30)
            .finish()
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
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .set_grace_period(10)
            .finish()
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
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .set_washout_period(washout_dict)
            .finish()
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
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .set_washout_period(washout_dict2)
            .finish()
        )

        self.assertDictEqual(tee2._washout_period_days, washout_dict2)

        treated_group = tee2._find_treated_patients(self.medrecord)
        treated_group, washout_nodes = tee2._apply_washout_period(
            self.medrecord, treated_group
        )

        self.assertEqual(treated_group, set({"P2", "P3", "P6"}))
        self.assertEqual(washout_nodes, set({}))

    def test_outcome_before_treatment(self):
        # find outcomes for default tee
        tee = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .finish()
        )
        treated_group = tee._find_treated_patients(self.medrecord)
        treated_group, treatment_true, outcome_before_treatment_nodes = (
            tee._find_outcomes(self.medrecord, treated_group)
        )
        self.assertEqual(treated_group, set({"P2", "P3", "P6"}))
        self.assertEqual(treatment_true, set({"P2", "P3"}))
        self.assertEqual(outcome_before_treatment_nodes, set())

        # set exclusion time for outcome before treatment
        tee2 = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .set_outcome_before_treatment_exclusion(30)
            .finish()
        )

        self.assertEqual(tee2._outcome_before_treatment_days, 30)

        treated_group = tee2._find_treated_patients(self.medrecord)
        treated_group, treatment_true, outcome_before_treatment_nodes = (
            tee2._find_outcomes(self.medrecord, treated_group)
        )
        self.assertEqual(treated_group, set({"P2", "P6"}))
        self.assertEqual(treatment_true, set({"P2"}))
        self.assertEqual(outcome_before_treatment_nodes, set({"P3"}))

    def test_filter_controls(self):
        tee = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .filter_controls(
                node().has_outgoing_edge_with(edge().connected_target("M2"))
                | node().has_incoming_edge_with(edge().connected_source("M2"))
            )
            .finish()
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
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .filter_controls(node().attribute("gender").equal("female"))
            .finish()
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
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .adjust_with_nearest_neighbors_matching(distance_metric="absolute")
            .finish()
        )

        subjects = tee.estimate.subjects_treatment_control(self.medrecord)

        # multiple patients are equally similar to the treatment group
        # these are exact macthes and should always be included
        self.assertIn("P4", subjects["control_true"])
        self.assertIn("P5", subjects["control_false"])

    def test_repeat_test(self):
        n_tests = 30
        for n in range(n_tests):
            print("--------------------------------")
            print(f"Test {n} out of {n_tests}")

    def test_metrics(self):
        """Test the metrics of the TreatmentEffect class."""

        tee = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .finish()
        )

        # Calculate metrics
        self.assertAlmostEqual(tee.estimate.absolute_risk(self.medrecord), 1 / 6)
        self.assertAlmostEqual(tee.estimate.relative_risk(self.medrecord), 4 / 3)
        self.assertAlmostEqual(tee.estimate.odds_ratio(self.medrecord), 2)
        self.assertAlmostEqual(tee.estimate.confounding_bias(self.medrecord), 22 / 21)
        self.assertAlmostEqual(tee.estimate.hazard_ratio(self.medrecord), 4 / 3)
        self.assertAlmostEqual(tee.estimate.number_needed_to_treat(self.medrecord), 6)

    def test_full_report(self):
        """Test the reporting of the TreatmentEffect class."""

        tee = (
            TreatmentEffect.builder()
            .set_treatment("Rivaroxaban")
            .set_outcome("Stroke")
            .finish()
        )

        # Calculate metrics
        full_report = tee.report.full_report(self.medrecord)

        report_test = {
            "absolute_risk": tee.estimate.absolute_risk(self.medrecord),
            "relative_risk": tee.estimate.relative_risk(self.medrecord),
            "odds_ratio": tee.estimate.odds_ratio(self.medrecord),
            "confounding_bias": tee.estimate.confounding_bias(self.medrecord),
            "hazard_ratio": tee.estimate.hazard_ratio(self.medrecord),
            "number_needed_to_treat": tee.estimate.number_needed_to_treat(
                self.medrecord
            ),
        }
        self.assertDictEqual(report_test, full_report)


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestTreatmentEffect)
    unittest.TextTestRunner(verbosity=2).run(run_test)
