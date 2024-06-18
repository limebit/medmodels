"""Tests for the TreatmentEffect class in the treatment_effect module."""

import unittest

import pandas as pd

from medmodels import MedRecord
from medmodels.medrecord import node
from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect


def create_patients() -> pd.DataFrame:
    """
    Create a patients dataframe.

    Returns:
        pd.DataFrame: A patients dataframe.
    """
    patients = pd.DataFrame(
        {
            "index": ["P1", "P2", "P3", "P4"],
            "age": [30, 40, 50, 60],
            "gender": ["male", "female", "male", "female"],
        }
    )
    return patients.set_index("index")


def create_diagnoses() -> pd.DataFrame:
    """
    Create a diagnoses dataframe.

    Returns:
        pd.DataFrame: A diagnoses dataframe.
    """
    diagnoses = pd.DataFrame(
        {
            "index": ["D1", "D2", "D3"],
            "code": [1, 2, 3],
        }
    )
    return diagnoses.set_index("index")


def create_prescriptions() -> pd.DataFrame:
    """
    Create a prescriptions dataframe.

    Returns:
        pd.DataFrame: A prescriptions dataframe.
    """
    prescriptions = pd.DataFrame(
        {
            "index": ["M1", "M2"],
        }
    )
    return prescriptions.set_index("index")


def create_edges() -> pd.DataFrame:
    """
    Create an edges dataframe.

    Returns:
        pd.DataFrame: An edges dataframe.
    """
    edges = pd.DataFrame(
        {
            "source": ["D1", "D2", "D2", "D1", "M1", "M2"],
            "target": ["P1", "P2", "P4", "P1", "P1", "P2"],
            "time": [
                "2000-01-01",
                "2000-07-01",
                "2000-01-01",
                "2000-01-01",
                "2100-01-01",
                "2000-01-01",
            ],
        }
    )
    return edges.set_index(["source", "target"])


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
        nodes=[patients, diagnoses, prescriptions], edges=[edges]
    )
    medrecord.add_group(group="patients", node=patients.index.to_list())
    return medrecord


class TestTreatmentEffect(unittest.TestCase):
    """Class to test the TreatmentEffect class in the treatment_effect module."""

    def test_init(self):
        """Test the initialization of the TreatmentEffect class."""
        medrecord = create_medrecord()

        # The treatment and outcome must be lists and not empty
        with self.assertRaises(AssertionError) as context:
            te = TreatmentEffect(medrecord, treatments=[], outcomes=["D1"])
            self.assertTrue("Treatment list is empty" in str(context.exception))
        with self.assertRaises(AssertionError) as context:
            te = TreatmentEffect(medrecord, treatments=["M1"], outcomes=[])
            self.assertTrue("Outcome list is empty" in str(context.exception))

        # Test if the patients dimension is found correctly
        patients_group = "not_existing_group"
        with self.assertRaises(AssertionError) as context:
            te = TreatmentEffect(
                medrecord,
                treatments=["M1"],
                outcomes=["D1"],
                patients_group=patients_group,
            )
            self.assertTrue(
                f"Patient group {patients_group} not found in the data. "
                f"Available groups: {medrecord.groups}" in str(context.exception)
            )

        # Test if the class can be initialized correctly
        treatments = ["M1", "M3"]
        outcomes = ["D1", "D3", "D4"]
        te = TreatmentEffect(medrecord, treatments=treatments, outcomes=outcomes)
        self.assertEqual(te.patients_group, "patients")
        self.assertEqual(te.time_attribute, "time")

        # In case we parse a treatment/oucome that has a prefix in the medrecord,
        # the prefix should be added to the treatment/outcome
        self.assertIn("M1", te.treatments)
        self.assertEqual(["D1", "D3"], te.outcomes)

        # Test if the treatments and outcomes are found in the medrecord
        self.assertIn("M3", te.not_found_treatments)
        self.assertIn("D4", te.not_found_outcomes)

        # Test if the groups are initialized correctly
        self.assertFalse(te.groups_sorted)
        self.assertFalse(te.control_true)
        self.assertFalse(te.control_false)
        self.assertFalse(te.treatment_true)
        self.assertFalse(te.treatment_false)

    def test_find_groups(self):
        """Test the find_groups method of the TreatmentEffect class."""
        medrecord = create_medrecord()

        # Initialize TreatmentEffect object
        te = TreatmentEffect(
            medrecord,
            treatments=["M1", "M2", "not_appearing_treatment"],
            outcomes=["D1", "D2"],
        )
        self.assertFalse(te.groups_sorted)

        # Test if the groups are found correctly
        te.find_groups()
        self.assertEqual(te.control_true, set({"P4"}))
        self.assertEqual(te.control_false, set({"P3"}))
        self.assertEqual(te.treatment_true, set({"P2"}))
        self.assertEqual(te.treatment_false, set({"P1"}))
        self.assertTrue(te.groups_sorted)

        # Test if the groups are found correctly when criteria is used
        te.find_groups(criteria_filter=node().attribute("gender").equal("female"))
        self.assertEqual(
            te.control_true, set({"P4"})
        )  # Empty set (we leave out female patients)
        self.assertFalse(te.control_false)
        self.assertEqual(te.treatment_true, set({"P2"}))
        self.assertFalse(te.treatment_false)
        self.assertTrue(te.groups_sorted)

        # Test what happens with a treatment that is not found
        te = TreatmentEffect(
            medrecord,
            treatments=["non_existing_treatment"],
            outcomes=["D1", "D2"],
        )
        with self.assertRaises(AssertionError) as context:
            te.find_groups()
            self.assertTrue(
                "No patients found for the treatment groups in this MedRecord"
                in str(context.exception)
            )

        # Test what happens when the attribute of time is not given correctly
        te = TreatmentEffect(
            medrecord,
            treatments=["M1", "M2"],
            outcomes=["D1", "D2"],
            time_attribute="non_existing_time_attribute",
        )
        with self.assertRaises(AssertionError) as context:
            te.find_groups()
            self.assertTrue(
                "Time attribute not found in the edge attributes"
                in str(context.exception)
            )

    def test_find_first_time(self):
        """Test the find_first_time method of the TreatmentEffect class."""
        medrecord = create_medrecord()
        te = TreatmentEffect(
            medrecord,
            treatments=["M1"],
            outcomes=["D1"],
        )

        # The first occurring time is obtained: we have 2 edges with that node.
        self.assertEqual(te.find_first_time("P1"), pd.Timestamp("2100-01-01"))
        with self.assertRaises(AssertionError) as context:
            te.find_first_time("not_existing_node")
            self.assertTrue(
                "No treatment found for node not_existing_node in this MedRecord"
                in str(context.exception)
            )

        # Test what happens when the attribute of time is not given correctly
        te = TreatmentEffect(
            medrecord,
            treatments=["M1"],
            outcomes=["D1"],
            time_attribute="non_existing_time_attribute",
        )
        with self.assertRaises(AssertionError) as context:
            te.find_first_time("P1")
            self.assertTrue(
                "Time attribute not found in the edge attributes"
                in str(context.exception)
            )

    def test_is_outcome_after_treatment(self):
        """Test the _is_outcome_after_treatment method of the TreatmentEffect class."""
        medrecord = create_medrecord()
        te = TreatmentEffect(medrecord, treatments=["M1", "M2"], outcomes=["D1", "D2"])

        # The outcome occurs here before the treatment
        self.assertFalse(te._is_outcome_after_treatment("P1", "D2", 1.0))

        # Test if there is an outcome after the treatment within the max time
        self.assertTrue(te._is_outcome_after_treatment("P2", "D2", 1.0))

        # The max_time here is smaller than the time btw  treatment & outcome
        self.assertFalse(te._is_outcome_after_treatment("P2", "D2", 1e-3))

    def test_find_controls(self):
        """Test the find_controls method of the TreatmentEffect class."""
        medrecord = create_medrecord()

        # Test if the controls are found correctly
        te = TreatmentEffect(medrecord, treatments=["M1", "M2"], outcomes=["D1", "D2"])
        te.treated_group = set({"P1", "P2"})
        self.assertEqual(te.find_controls(), ({"P4"}, {"P3"}))

        # Find controls requires treated_group to be set first
        te.treated_group = set()
        self.assertNotEqual(te.find_controls(), ({"P4"}, {"P3"}))

        # Test if the controls are found correctly when criteria is used
        te.treated_group = set({"P1", "P2"})
        self.assertEqual(
            te.find_controls(criteria_filter=node().attribute("gender").equal("male")),
            (set(), {"P3"}),
        )

        # Test what happens when the criteria is not met
        with self.assertRaises(AssertionError) as context:
            (
                te.find_controls(
                    criteria_filter=node().attribute("gender").equal("non_existing")
                )
            )
            self.assertTrue(
                "No patients found for the control groups in this MedRecord"
                in str(context.exception)
            )

    def test_count_subjects(self):
        """Test the count_subjects method of the TreatmentEffect class."""
        medrecord = create_medrecord()

        # count_subjects should only be called after find_groups()
        te = TreatmentEffect(medrecord, treatments=["M1", "M2"], outcomes=["D1", "D2"])
        with self.assertRaises(AssertionError) as context:
            te.subject_counts
            self.assertTrue(
                "Groups must be sorted, use find_groups() method first"
                in str(context.exception)
            )

        # Test if the subjects are counted correctly
        te.find_groups()
        self.assertEqual(te.subject_counts, (1, 1, 1, 1))

        # Test what happens if no treatment subjects are found
        te.treatment_false = set()
        with self.assertRaises(AssertionError) as context:
            te.subject_counts
            self.assertTrue(
                "No subjects found in the treatment false group"
                in str(context.exception)
            )

        te.find_groups()
        te.control_true = set()
        with self.assertRaises(AssertionError) as context:
            te.subject_counts
            self.assertTrue(
                "No subjects found in the control true group" in str(context.exception)
            )

        te.find_groups()
        te.control_false = set()
        with self.assertRaises(AssertionError) as context:
            te.subject_counts
            self.assertTrue(
                "No subjects found in the control false group" in str(context.exception)
            )

    def test_metrics(self):
        """Test the metrics of the TreatmentEffect class."""
        medrecord = create_medrecord()

        te = TreatmentEffect(medrecord, treatments=["M1", "M2"], outcomes=["D1", "D2"])
        te.find_groups()

        # Calculate metrics
        rr = te.relative_risk()
        self.assertAlmostEqual(rr, 1.0)

        odds_ratio = te.odds_ratio()
        self.assertAlmostEqual(odds_ratio, 1.0)

        confounding_bias = te.confounding_bias()
        self.assertAlmostEqual(confounding_bias, 1)

        # Test with more patients
        te.treatment_true = set({"P1"})
        te.treatment_false = set({"P2", "P3"})
        te.control_true = set({"P4"})
        te.control_false = set({"P5"})

        # Calculate relative risk
        rr = te.relative_risk()
        self.assertAlmostEqual(rr, 2 / 3)

        odds_ratio = te.odds_ratio()
        self.assertAlmostEqual(odds_ratio, 0.5)

        confounding_bias = te.confounding_bias()
        self.assertAlmostEqual(confounding_bias, 16 / 15)


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestTreatmentEffect)
    unittest.TextTestRunner(verbosity=2).run(run_test)
