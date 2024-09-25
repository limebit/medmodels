"""Tests for the MTGAN Preprocessor class in the data synthesis module."""

import datetime
import unittest

import pandas as pd

from medmodels import MedRecord
from medmodels.data_synthesis.mtgan.modules.preprocessor import (
    MTGANPreprocessor,
    PreprocessingAttributes,
    PreprocessingHyperparameters,
)
from medmodels.medrecord import edge


def create_patients() -> pd.DataFrame:
    """Creates a patients dataframe.

    Returns:
        pd.DataFrame: A patients dataframe.
    """
    return pd.DataFrame(
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


def create_concepts():
    """Creates a concepts dataframe.

    Returns:
        pd.DataFrame: A concepts dataframe.
    """
    return pd.DataFrame(
        {
            "index": ["M1", "M2", "D1"],
            "name": ["Rivaroxaban", "Warfarin", "Stroke"],
        }
    )


def create_edges() -> pd.DataFrame:
    """Creates an edges dataframe.

    Returns:
        pd.DataFrame: An edges dataframe.
    """
    return pd.DataFrame(
        {
            "source": [
                "M2",
                "M1",
                "M1",
                "M2",
                "M1",
                "M2",
                "M1",
                "M2",
                "M2",
                "D1",
                "D1",
                "D1",
                "D1",
                "D1",
                "D1",
            ],
            "target": [
                "P1",
                "P1",
                "P2",
                "P2",
                "P3",
                "P5",
                "P6",
                "P9",
                "P1",
                "P1",
                "P2",
                "P3",
                "P3",
                "P4",
                "P7",
            ],
            "time": [
                datetime.datetime.strptime("1999-10-15", "%Y-%m-%d"),
                datetime.datetime.strptime("1999-10-15", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
                datetime.datetime.strptime("1999-12-15", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-07-01", "%Y-%m-%d"),
                datetime.datetime.strptime("1999-12-15", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-05", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
                datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
            ],
        }
    )


def create_wrong_edges() -> pd.DataFrame:
    """Create edges with strings instead of datetimes.

    Returns:
        pd.DataFrame: An edges dataframe.
    """
    return pd.DataFrame(
        {
            "source": [
                "M2",
                "M1",
                "M1",
                "M2",
                "M1",
                "M2",
                "M1",
                "M2",
                "M2",
                "D1",
                "D1",
                "D1",
                "D1",
                "D1",
                "D1",
            ],
            "target": [
                "P1",
                "P1",
                "P2",
                "P2",
                "P3",
                "P5",
                "P6",
                "P9",
                "P1",
                "P1",
                "P2",
                "P3",
                "P3",
                "P4",
                "P7",
            ],
            "time": [
                "1999-10-15",
                "1999-10-15",
                "2000-01-01",
                "1999-12-15",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-07-01",
                "1999-12-15",
                "2000-01-05",
                "2000-01-01",
                "2000-01-01",
            ],
        }
    )


def create_medrecord() -> MedRecord:
    """Creates a MedRecord object.

    Returns:
        MedRecord: A MedRecord object.
    """
    patients = create_patients()
    concepts = create_concepts()
    edges = create_edges()
    medrecord = MedRecord.from_pandas(
        nodes=[(patients, "index"), (concepts, "index")],
        edges=[(edges, "source", "target")],
    )
    medrecord.add_group("patients", patients["index"].to_list())
    medrecord.add_group("concepts", concepts["index"].to_list())
    return medrecord


def create_medrecord_wrong_time_edges() -> MedRecord:
    """Creates a MedRecord object with edges with wrong time format.

    Returns:
        MedRecord: A MedRecord object.
    """
    patients = create_patients()
    concepts = create_concepts()
    edges = create_wrong_edges()
    medrecord = MedRecord.from_pandas(
        nodes=[(patients, "index"), (concepts, "index")],
        edges=[(edges, "source", "target")],
    )
    medrecord.add_group("patients", patients["index"].to_list())
    medrecord.add_group("concepts", concepts["index"].to_list())
    return medrecord


class TestMTGANPreprocessor(unittest.TestCase):
    """Class to test the MTGAN Preprocessor class in the data_synthesis module."""

    def setUp(self):
        self.hyperparameters = PreprocessingHyperparameters(
            minimum_occurrences_concept=1,
            time_interval_days=1,
            minimum_concepts_per_window=1,
            number_of_sampled_patients=0,
        )
        self.preprocessor = MTGANPreprocessor(self.hyperparameters)

    def test_init(self):
        """Tests the __init__ method of the MTGANPreprocessor class."""
        self.assertIsInstance(self.preprocessor, MTGANPreprocessor)

        self.assertEqual(self.preprocessor.hyperparameters, self.hyperparameters)
        self.assertEqual(self.preprocessor.patients_group, "patients")
        self.assertEqual(self.preprocessor.concepts_group, "concepts")
        self.assertEqual(self.preprocessor.time_attribute, "time")

    def test_get_attribute_name(self):
        """Tests the get_attribute_name method of the MTGANPreprocessor class."""
        medrecord = create_medrecord()

        # Test with a non-existing attribute: it does not change the attribute name
        attribute_name = "non_existing_attribute"
        self.assertEqual(
            self.preprocessor._get_attribute_name(medrecord, attribute_name),
            attribute_name,
        )

        # Test with an existing attribute: it adds a suffix to the attribute name
        attribute_name = "age"
        self.assertEqual(
            self.preprocessor._get_attribute_name(medrecord, attribute_name), "age_1"
        )

        # If already exists, and the one with a suffix also does, adds 1 to the number
        attribute_name = "age"
        medrecord.add_node("new_patient", {"age_1": 30})
        self.assertEqual(
            self.preprocessor._get_attribute_name(medrecord, attribute_name), "age_2"
        )

    def test_remove_unconnected_patients(self):
        """Tests the remove_unconnected_patients method of the MTGANPreprocessor class."""
        medrecord = create_medrecord()
        medrecord_to_compare = create_medrecord()

        self.preprocessor._remove_unconnected_patients(medrecord)
        self.assertLess(
            len(medrecord.nodes_in_group("patients")),
            len(medrecord_to_compare.nodes_in_group("patients")),
        )
        self.assertNotIn("P8", medrecord.nodes_in_group("patients"))

    def test_remove_uncommon_concepts(self):
        """Tests the remove_uncommon_concepts method of the MTGANPreprocessor class."""
        medrecord = create_medrecord()
        medrecord_to_compare = create_medrecord()

        # Test with minimum_number_ocurrences=2, none removed
        self.preprocessor._remove_uncommon_concepts(
            medrecord,
            minimum_number_ocurrences=2,
        )
        self.assertEqual(
            len(medrecord.nodes_in_group("concepts")),
            len(medrecord_to_compare.nodes_in_group("concepts")),
        )

        # Test with minimum_number_ocurrences=5, 2 removed, 1 kept
        self.preprocessor._remove_uncommon_concepts(
            medrecord,
            minimum_number_ocurrences=5,
        )
        self.assertEqual(len(medrecord.nodes_in_group("concepts")), 2)
        self.assertIn("D1", medrecord.nodes_in_group("concepts"))
        self.assertIn("M2", medrecord.nodes_in_group("concepts"))

        # Test with minimum_number_ocurrences=10, all removed
        self.preprocessor._remove_uncommon_concepts(
            medrecord,
            minimum_number_ocurrences=10,
        )
        self.assertEqual(len(medrecord.nodes_in_group("concepts")), 0)

    def test_assign_concept_indices(self):
        """Test the assign_concept_indices method of the MTGANPreprocessor class."""
        medrecord = create_medrecord()
        concept_index_attribute = self.preprocessor._get_attribute_name(
            medrecord, "concept_index"
        )
        concepts_list = self.preprocessor._assign_concept_indices(
            medrecord, concept_index_attribute
        )
        self.assertEqual(concepts_list[0], "D1")
        self.assertIn(concepts_list[1], "M1")
        self.assertIn(concepts_list[2], "M2")
        self.assertEqual(medrecord.node["D1", concept_index_attribute], 0)
        self.assertEqual(medrecord.node["M1", concept_index_attribute], 1)
        self.assertEqual(medrecord.node["M2", concept_index_attribute], 2)

        # To show what happens when removing concepts:
        self.preprocessor._remove_uncommon_concepts(
            medrecord,
            minimum_number_ocurrences=5,
        )
        concepts_list = self.preprocessor._assign_concept_indices(
            medrecord, concept_index_attribute
        )
        self.assertEqual(len(medrecord.nodes_in_group("concepts")), 2)
        self.assertEqual(concepts_list[0], "D1")
        self.assertEqual(concepts_list[1], "M2")

        # To show what is returned with no concepts left:
        self.preprocessor._remove_uncommon_concepts(
            medrecord,
            minimum_number_ocurrences=10,
        )
        concepts_list = self.preprocessor._assign_concept_indices(
            medrecord, concept_index_attribute
        )
        self.assertEqual(len(medrecord.nodes_in_group("concepts")), 0)
        self.assertEqual(concepts_list, [])

    def test_sample_patients(self):
        """Tests the sample_patients method of the MTGANPreprocessor class."""
        medrecord = create_medrecord()

        # Test with number_of_sampled_patients=0, all patients kept
        self.preprocessor._sample_patients(medrecord, number_of_sampled_patients=0)
        # There are initially 9 patients, none are removed
        self.assertEqual(len(medrecord.nodes_in_group("patients")), 9)

        # Test with number_of_sampled_patients=5, 5 patients sampled
        self.preprocessor._sample_patients(medrecord, number_of_sampled_patients=5)
        self.assertEqual(len(medrecord.nodes_in_group("patients")), 5)

    def test_invalid_sample_patients(self):
        """Tests the sample_patients method of the MTGANPreprocessor class with invalid inputs."""
        medrecord = create_medrecord()
        with self.assertRaises(ValueError) as context:
            self.preprocessor._sample_patients(medrecord, number_of_sampled_patients=10)

        expected_message = "Number of sampled patients (10) is greater than the number of patients in the MedRecord (9)"
        self.assertEqual(str(context.exception).strip(), expected_message.strip())

    def test_find_first_admission(self):
        """Tests the find_first_admission method of the MTGANPreprocessor class."""
        # We need to prune the medrecord to remove the patient without edges (with sample patients).
        medrecord = create_medrecord()
        self.preprocessor._remove_unconnected_patients(medrecord)
        first_admission_attribute = self.preprocessor._find_first_admission(medrecord)
        self.assertEqual(first_admission_attribute, "first_admission")
        self.assertEqual(
            medrecord.node["P1", first_admission_attribute],
            datetime.datetime(1999, 10, 15),
        )
        self.assertEqual(
            medrecord.node["P2", first_admission_attribute],
            datetime.datetime(1999, 12, 15),
        )

        # Removing the first edge of P1 should change the first admission date
        medrecord = create_medrecord()
        medrecord.remove_edge(
            medrecord.select_edges(
                edge().attribute("time").equal(datetime.datetime(1999, 10, 15))
            )
        )
        self.preprocessor._remove_unconnected_patients(medrecord)
        first_admission_attribute = self.preprocessor._find_first_admission(medrecord)
        self.assertEqual(
            medrecord.node["P1", first_admission_attribute],
            datetime.datetime(2000, 1, 1),
        )

    def test_invalid_find_first_admission(self):
        """Tests the find_first_admission method of the MTGANPreprocessor class with invalid inputs."""
        medrecord = create_medrecord()
        with self.assertRaises(ValueError) as context:
            self.preprocessor._find_first_admission(medrecord)

        self.assertEqual(
            str(context.exception).strip(),
            "No edge found for node P8 in this MedRecord",
        )

    def test_preprocess(self):
        """Tests the preprocess method of the MTGANPreprocessor class."""
        medrecord = create_medrecord()
        medrecord, concepts_list, preprocessing_attributes = (
            self.preprocessor.preprocess(medrecord)
        )

        # Only P1, P2, P3 kept, since they have at least 2 different time windows
        self.assertIn("P1", medrecord.nodes_in_group("patients"))
        self.assertIn("P2", medrecord.nodes_in_group("patients"))
        self.assertIn("P3", medrecord.nodes_in_group("patients"))
        self.assertEqual(concepts_list[0], "D1")
        self.assertIn(concepts_list[1], "M1")
        self.assertIn(concepts_list[2], "M2")
        self.assertEqual(
            preprocessing_attributes,
            PreprocessingAttributes(
                first_admission_attribute="first_admission",
                time_window_attribute="time_window",
                concept_index_attribute="concept_index",
                concept_edge_attribute="concept_edge",
                number_of_windows_attribute="number_of_windows",
                absolute_time_window_attribute="absolute_time_window",
            ),
        )

        # Check the differently added attributes
        self.assertEqual(
            medrecord.node["P1", "first_admission"], datetime.datetime(1999, 10, 15)
        )
        self.assertEqual(medrecord.node["P1", "number_of_windows"], 2)
        edges = medrecord.edges_connecting("P1", "M2", directed=False)
        self.assertEqual(set(edges), set([0, 8]))
        self.assertEqual(medrecord.edge[edges, "absolute_time_window"], {0: 0, 8: 78})
        self.assertEqual(medrecord.edge[edges, "time_window"], {0: 0, 8: 1})
        self.assertEqual(medrecord.edge[edges, "concept_edge"], {0: "M2", 8: "M2"})

        ## Changing the minimum number of occurrences of a concept, results differ
        medrecord = create_medrecord()
        hyperparameters = self.hyperparameters.copy()
        hyperparameters["minimum_occurrences_concept"] = 6
        preprocessor = MTGANPreprocessor(hyperparameters)
        medrecord, concepts_list, _ = preprocessor.preprocess(medrecord)

        # Patient 3 is the only one that has a concept with at least 6 occurrences in two different time windows
        self.assertEqual(len(medrecord.nodes_in_group("patients")), 1)
        self.assertIn("P3", medrecord.nodes_in_group("patients"))
        self.assertEqual(len(medrecord.nodes_in_group("concepts")), 1)
        self.assertEqual(concepts_list[0], "D1")

        ## Extending the time intervals used to encapsulate the time windows, results differ
        medrecord = create_medrecord()
        hyperparameters = self.hyperparameters.copy()
        hyperparameters["time_interval_days"] = 30
        preprocessor = MTGANPreprocessor(hyperparameters)
        medrecord, _, _ = preprocessor.preprocess(medrecord)

        # Patient 3 is removed, since it only has one time window now
        self.assertEqual(len(medrecord.nodes_in_group("patients")), 2)
        self.assertIn("P1", medrecord.nodes_in_group("patients"))
        self.assertIn("P2", medrecord.nodes_in_group("patients"))

        ## Changing the number of minimum concepts per window to 2, results differ
        medrecord = create_medrecord()
        hyperparameters = self.hyperparameters.copy()
        hyperparameters["minimum_concepts_per_window"] = 2
        preprocessor = MTGANPreprocessor(hyperparameters)
        medrecord, _, _ = preprocessor.preprocess(medrecord)

        # Only Patient 1 is kept, since it has at least 2 concepts in each time window
        self.assertEqual(len(medrecord.nodes_in_group("patients")), 1)
        self.assertIn("P1", medrecord.nodes_in_group("patients"))

        ## By changing the number of sampled patients and the sampling seed, we get different results
        medrecord = create_medrecord()
        hyperparameters = self.hyperparameters.copy()
        hyperparameters["number_of_sampled_patients"] = 2
        preprocessor = MTGANPreprocessor(hyperparameters)
        medrecord, _, _ = preprocessor.preprocess(medrecord)

        medrecord_to_compare = create_medrecord()
        preprocessor_to_compare = MTGANPreprocessor(self.hyperparameters, seed=0)
        medrecord_to_compare, _, _ = preprocessor_to_compare.preprocess(
            medrecord_to_compare
        )
        self.assertNotEqual(
            medrecord.nodes_in_group("patients"),
            medrecord_to_compare.nodes_in_group("patients"),
        )

    def test_invalid_preprocess(self):
        """Tests the preprocess method of the MTGANPreprocessor class with invalid inputs."""
        # Non existing group names or time attribute
        medrecord = create_medrecord()
        preprocessor = MTGANPreprocessor(
            self.hyperparameters, patients_group="non_existing_patients_group"
        )
        with self.assertRaises(IndexError) as context:
            preprocessor.preprocess(medrecord)
        self.assertEqual(
            str(context.exception), "Cannot find group non_existing_patients_group"
        )
        preprocessor = MTGANPreprocessor(
            self.hyperparameters, concepts_group="non_existing_concepts_group"
        )
        with self.assertRaises(IndexError) as context:
            preprocessor.preprocess(medrecord)
        self.assertEqual(
            str(context.exception), "Cannot find group non_existing_concepts_group"
        )
        preprocessor = MTGANPreprocessor(
            self.hyperparameters, time_attribute="non_existing_attribute"
        )
        with self.assertRaises(ValueError) as context:
            preprocessor.preprocess(medrecord)
        self.assertEqual(
            str(context.exception).strip(),
            "No edges in the MedRecord with that time attribute",
        )

        # If the time attribute is not in the right format, it raises an error
        medrecord = create_medrecord_wrong_time_edges()
        with self.assertRaises(ValueError) as context:
            self.preprocessor.preprocess(medrecord)
        self.assertEqual(
            str(context.exception),
            "First admission attribute needs to be a datetime object, but got <class 'str'>",
        )

        # If the minimum number of occurrences of a concept is too large, no patients are left after preprocessing
        medrecord = create_medrecord()
        hyperparameters = self.hyperparameters.copy()
        hyperparameters["minimum_occurrences_concept"] = 10
        preprocessor = MTGANPreprocessor(hyperparameters)
        with self.assertRaises(ValueError) as context:
            preprocessor.preprocess(medrecord)
        self.assertEqual(str(context.exception), "No patients left after preprocessing")

        # If the time interval is too large, no patients are left after preprocessing (minimum of 2 time windows)
        medrecord = create_medrecord()
        hyperparameters = self.hyperparameters.copy()
        hyperparameters["time_interval_days"] = 365
        preprocessor = MTGANPreprocessor(hyperparameters)
        with self.assertRaises(ValueError) as context:
            preprocessor.preprocess(medrecord)
        self.assertEqual(str(context.exception), "No patients left after preprocessing")

        # With minimum 3 concepts per window, no patients are left after preprocessing
        medrecord = create_medrecord()
        hyperparameters = self.hyperparameters.copy()
        hyperparameters["minimum_concepts_per_window"] = 3
        preprocessor = MTGANPreprocessor(hyperparameters)
        with self.assertRaises(ValueError) as context:
            preprocessor.preprocess(medrecord)
        self.assertEqual(str(context.exception), "No patients left after preprocessing")


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestMTGANPreprocessor)
    unittest.TextTestRunner(verbosity=2).run(run_test)
