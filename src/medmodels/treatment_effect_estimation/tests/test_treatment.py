import unittest
import numpy as np
import pandas as pd

from medmodels.dataclass.dataclass import MedRecord
from medmodels.dataclass.utils import df_to_nodes, df_to_edges
from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect


class TestTreatmentEffect(unittest.TestCase):
    def setUp(self):
        medrecord = MedRecord()
        patients = pd.DataFrame(
            {
                "patients_id": ["P1", "P2", "P3", "P4"],
                "age": [30, 40, 50, 60],
                "gender": ["male", "male", "female", "female"],
            }
        ).astype(str)
        patient_nodes = df_to_nodes(patients, "patients_id", ["age", "gender"])

        diagnoses = pd.DataFrame(
            {
                "diagnoses_id": ["D1", "D2", "D1", "D2", "diagnoses_D3", "D1"],
                "code": [1, 2, 1, 2, 3, 1],
                "patients_id": ["P1", "P2", "P3", "P4", "P1", "P1"],
                "time": [
                    "2000-01-01 10:05:00",
                    "2000-01-02 10:05:00",
                    "2000-01-03 10:05:00",
                    "2000-01-04 10:05:00",
                    "2000-01-05 10:05:00",
                    "2100-01-06 10:05:00",
                ],
            }
        )
        diagnoses["relation_type"] = "patients_diagnoses"
        diagnoses_nodes = df_to_nodes(diagnoses, "diagnoses_id", [])
        diagnoses_patient_edges = df_to_edges(
            diagnoses, "diagnoses_id", "patients_id", ["relation_type", "time"]
        )

        medrecord.add_nodes(patient_nodes, "patients")
        medrecord.add_nodes(diagnoses_nodes, "diagnoses")
        medrecord.add_edges(diagnoses_patient_edges)
        prescriptions = pd.DataFrame(
            {
                "prescriptions_id": np.array(["M1", "M2", "M1", "M1"]),
                "patients_id": np.array(["P1", "P2", "P3", "P4"]),
                "time": [
                    "2000-01-05 10:05:00",
                    "2000-01-01 09:05:00",
                    "2100-01-02 10:05:00",
                    "2100-01-07 10:05:00",
                ],
            }
        ).astype(str)

        prescriptions["relation_type"] = "patients_prescriptions"
        patients_prescriptions_edges = df_to_edges(
            data=prescriptions,
            identifier1="patients_id",
            identifier2="prescriptions_id",
            attributes=["relation_type", "time"],
        )
        medrecord.add_edges(patients_prescriptions_edges)
        self.medrecord = medrecord

    def test_TreatmentEffect_init(self):
        # The treatment and outcome must be lists and not empty
        with self.assertRaises(AssertionError) as context:
            te = TreatmentEffect(self.medrecord, treatments="D1", outcomes="M1")
            self.assertTrue("Treatment must be a list" in str(context.exception))
        with self.assertRaises(AssertionError) as context:
            te = TreatmentEffect(self.medrecord, treatments=[], outcomes="M1")
            self.assertTrue("Treatment list is empty" in str(context.exception))

        with self.assertRaises(AssertionError) as context:
            te = TreatmentEffect(self.medrecord, treatments=["D1"], outcomes="M1")
            self.assertTrue("Outcome must be a list" in str(context.exception))
        with self.assertRaises(AssertionError) as context:
            te = TreatmentEffect(self.medrecord, treatments=["D1"], outcomes=[])
            self.assertTrue("Outcome list is empty" in str(context.exception))

        # Test if the patients dimension is found correctly
        patients_dimension = "not_functioning_dim"
        with self.assertRaises(AssertionError) as context:
            te = TreatmentEffect(
                self.medrecord,
                treatments=["D1"],
                outcomes=["M1"],
                patients_dimension=patients_dimension,
            )
            self.assertTrue(
                f"Dimension {patients_dimension} not found in the data. "
                f"Available dimensions: {self.medrecord.dimensions}"
                in str(context.exception)
            )

        # Test if the class can be initialized correctly
        treatments = ["D1", "D3", "D4"]
        outcomes = ["M1", "M3"]
        te = TreatmentEffect(self.medrecord, treatments=treatments, outcomes=outcomes)
        self.assertEqual(te.patients_dimension, "patients")
        self.assertEqual(te.time_attribute, "time")

        # In case we parse a treatment/oucome that has a prefix in the medrecord,
        # the prefix should be added to the treatment/outcome
        self.assertEqual(["D1", "diagnoses_D3"], te.treatments)
        self.assertIn("M1", te.outcomes)

        # Test if the treatments and outcomes are found in the medrecord
        self.assertIn("D4", te.not_found_treatments)
        self.assertIn("M3", te.not_found_outcomes)

        # Test if the groups are initialized correctly
        self.assertFalse(te.groups_sorted)
        self.assertFalse(te.control_true)
        self.assertFalse(te.control_false)
        self.assertFalse(te.treatment_true)
        self.assertFalse(te.treatment_false)

    def test_format_concepts(self):
        # Initialize TreatmentEffect object
        te = TreatmentEffect(
            self.medrecord,
            treatments=["initializing node"],
            outcomes=["initializing_node"],
        )

        # Test if the concepts can be formatted correctly
        self.assertEqual(
            te.format_concepts(["D1", "D2", "M1"]), (["D1", "D2", "M1"], [])
        )
        self.assertEqual(te.format_concepts(["D1", "D3"]), (["D1", "diagnoses_D3"], []))

        self.assertEqual(
            te.format_concepts(["D1", "M1", "non_existent_node"]),
            (["D1", "M1"], ["non_existent_node"]),
        )

    def test_find_groups(self):
        # Initialize TreatmentEffect object
        te = TreatmentEffect(
            self.medrecord,
            treatments=["D1", "not_appearing_treatment"],
            outcomes=["M1"],
        )
        self.assertFalse(te.groups_sorted)

        # Test if the groups are found correctly
        te.find_groups()
        self.assertEqual(
            self.medrecord.groups, []
        )  # We remove the groups from the medrecord after finding them
        self.assertEqual(te.control_true, set({"P4"}))
        self.assertEqual(te.control_false, set({"P2"}))
        self.assertEqual(te.treatment_true, set({"P1"}))
        self.assertEqual(te.treatment_false, set({"P3"}))
        self.assertTrue(te.groups_sorted)

        # Test if the groups are found correctly when criteria is used
        te.find_groups(criteria_filter=["patients gender == male"])
        self.assertFalse(te.control_true)  # Empty set (we leave out female patients)
        self.assertEqual(te.control_false, set({"P2"}))
        self.assertEqual(te.treatment_true, set({"P1"}))
        self.assertFalse(te.treatment_false)  # Empty set
        self.assertTrue(te.groups_sorted)

        # Test what happens with a treatment that is not found
        te = TreatmentEffect(
            self.medrecord,
            treatments=["non_existing_treatment"],
            outcomes=["M1"],
        )
        with self.assertRaises(AssertionError) as context:
            te.find_groups()
            self.assertTrue(
                "No patients found for the treatment groups in this MedRecord"
                in str(context.exception)
            )

        # Test what happens when the attribute of time is not given correctly
        te = TreatmentEffect(
            self.medrecord,
            treatments=["D1"],
            outcomes=["M1"],
            time_attribute="non_existing_time_attribute",
        )
        with self.assertRaises(AssertionError) as context:
            te.find_groups()
            self.assertTrue(
                "Time attribute not found in the edge attributes"
                in str(context.exception)
            )

    def test_find_first_time(self):
        te = TreatmentEffect(
            self.medrecord,
            treatments=["D1"],
            outcomes=["M1"],
        )
        # The first occurring time is obtained: we have 2 edges with that node.
        self.assertEqual(te.find_first_time("P1"), pd.Timestamp("2000-01-01 10:05:00"))
        with self.assertRaises(AssertionError) as context:
            te.find_first_time("not_existing_node")
            self.assertTrue(
                "No treatment found for node not_existing_node in this MedRecord"
                in str(context.exception)
            )

        # Test what happens when the attribute of time is not given correctly
        te = TreatmentEffect(
            self.medrecord,
            treatments=["D1"],
            outcomes=["M1"],
            time_attribute="non_existing_time_attribute",
        )
        with self.assertRaises(AssertionError) as context:
            te.find_first_time("P1")
            self.assertTrue(
                "Time attribute not found in the edge attributes"
                in str(context.exception)
            )

    def test_is_outcome_after_treatment(self):
        te = TreatmentEffect(self.medrecord, treatments=["D1"], outcomes=["M1"])

        # Test if there is an outcome after the treatment within the max time
        self.assertTrue(te._is_outcome_after_treatment("P1", "M1", 1.0))

        # The max_time here is smaller than the time btw  treatment & outcome
        self.assertFalse(te._is_outcome_after_treatment("P1", "M1", 1e-3))

    def test_find_controls(self):
        # Test if the controls are found correctly
        te = TreatmentEffect(self.medrecord, treatments=["D1"], outcomes=["M1"])
        te.find_groups()
        self.assertEqual(te.find_controls(), ({"P4"}, {"P2"}))

        # Test if the controls are found correctly when criteria is used
        self.assertEqual(
            te.find_controls(criteria_filter=["patients gender == male"]),
            (set(), {"P2"}),
        )

        # Test what happens when the criteria is not met
        with self.assertRaises(AssertionError) as context:
            te.find_controls(
                criteria_filter=["patients gender == non_existing_gender"]
            ),
            self.assertTrue(
                "No patients found for the control groups in this MedRecord"
                in str(context.exception)
            )

    def test_subject_counts(self):
        # subject_counts should only be called after find_groups()
        te = TreatmentEffect(self.medrecord, treatments=["D1"], outcomes=["M1"])
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
        te = TreatmentEffect(self.medrecord, treatments=["D1"], outcomes=["M1"])
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
