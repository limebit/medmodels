"""Tests for the NeighborsMatching class in the matching module."""

import unittest
from typing import List, Set

import pandas as pd

from medmodels import MedRecord
from medmodels.medrecord.types import NodeIndex
from medmodels.treatment_effect.matching.neighbors import NeighborsMatching


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
    medrecord = MedRecord.from_pandas(nodes=[(patients, "index")])
    medrecord.add_group(group="patients", nodes=patients["index"].to_list())
    return medrecord


class TestNeighborsMatching(unittest.TestCase):
    """Class to test the NeighborsMatching class in the matching module."""

    def setUp(self):
        self.medrecord = create_medrecord()

    def test_preprocess_data(self):
        neighbors_matching = NeighborsMatching(number_of_neighbors=1)

        control_set: Set[NodeIndex] = {"P1", "P3", "P5", "P7", "P9"}
        treated_set: Set[NodeIndex] = {"P2", "P4", "P6"}

        data_treated, data_control = neighbors_matching._preprocess_data(
            medrecord=self.medrecord,
            control_set=control_set,
            treated_set=treated_set,
            patients_group="patients",
            essential_covariates=["age", "gender"],
            one_hot_covariates=["gender"],
        )

        # Assert that the treated and control dataframes have the correct columns
        self.assertIn("age", data_treated.columns)
        self.assertIn("age", data_control.columns)
        self.assertTrue(
            "gender_female" in data_treated.columns
            or "gender_male" in data_treated.columns
        )
        self.assertTrue(
            "gender_female" in data_control.columns
            or "gender_male" in data_control.columns
        )

        # Assert that the treated and control dataframes have the correct number of rows
        self.assertEqual(len(data_treated), len(treated_set))
        self.assertEqual(len(data_control), len(control_set))

        # Try automatic detection of attributes
        data_treated, data_control = neighbors_matching._preprocess_data(
            medrecord=self.medrecord,
            control_set=control_set,
            treated_set=treated_set,
            patients_group="patients",
        )

        # Assert that the treated and control dataframes have the correct columns
        self.assertIn("age", data_treated.columns)
        self.assertIn("age", data_control.columns)
        self.assertTrue(
            "gender_female" in data_treated.columns
            or "gender_male" in data_treated.columns
        )
        self.assertTrue(
            "gender_female" in data_control.columns
            or "gender_male" in data_control.columns
        )

        # Assert that the treated and control dataframes have the correct number of rows
        self.assertEqual(len(data_treated), len(treated_set))
        self.assertEqual(len(data_control), len(control_set))

    def test_match_controls(self):
        neighbors_matching = NeighborsMatching(number_of_neighbors=1)

        control_set: Set[NodeIndex] = {"P1", "P3", "P5", "P7", "P9"}
        treated_set: Set[NodeIndex] = {"P2", "P4", "P6"}

        matched_controls = neighbors_matching.match_controls(
            medrecord=self.medrecord,
            control_set=control_set,
            treated_set=treated_set,
            patients_group="patients",
            essential_covariates=["age", "gender"],
            one_hot_covariates=["gender"],
        )

        # Assert that the matched controls are a subset of the control set
        self.assertTrue(matched_controls.issubset(control_set))

        # Assert that the correct number of controls were matched
        self.assertEqual(len(matched_controls), len(treated_set))

        # Assert it works equally if no covariates are given (automatically assigned)
        matched_controls_no_covariates_specified = neighbors_matching.match_controls(
            medrecord=self.medrecord,
            control_set=control_set,
            treated_set=treated_set,
            patients_group="patients",
        )

        self.assertTrue(matched_controls_no_covariates_specified.issubset(control_set))
        self.assertEqual(
            len(matched_controls_no_covariates_specified), len(treated_set)
        )

    def test_check_nodes(self):
        neighbors_matching = NeighborsMatching(number_of_neighbors=1)

        control_set: Set[NodeIndex] = {"P1", "P3", "P5", "P7", "P9"}
        treated_set: Set[NodeIndex] = {"P2", "P4", "P6", "P8"}

        # Test valid case
        valid_control_set = neighbors_matching._check_nodes(
            medrecord=self.medrecord,
            treated_set=treated_set,
            control_set=control_set,
            essential_covariates=["age", "gender"],
        )
        self.assertEqual(valid_control_set, control_set)

    def test_invalid_check_nodes(self):
        neighbors_matching = NeighborsMatching(number_of_neighbors=1)

        control_set: Set[NodeIndex] = {"P1", "P3", "P5", "P7", "P9"}
        treated_set: Set[NodeIndex] = {"P2", "P4", "P6"}

        # Test insufficient control subjects
        with self.assertRaises(ValueError) as context:
            neighbors_matching._check_nodes(
                medrecord=self.medrecord,
                treated_set=treated_set,
                control_set={"P1"},
                essential_covariates=["age", "gender"],
            )
        self.assertEqual(
            str(context.exception),
            "Not enough control subjects to match the treated subjects",
        )

        with self.assertRaises(ValueError) as context:
            neighbors_matching = NeighborsMatching(number_of_neighbors=2)
            neighbors_matching._check_nodes(
                medrecord=self.medrecord,
                treated_set=treated_set,
                control_set=control_set,
                essential_covariates=["age", "gender"],
            )
        self.assertEqual(
            str(context.exception),
            "Not enough control subjects to match the treated subjects",
        )

        # Test missing essential covariates in treated set
        with self.assertRaises(ValueError) as context:
            neighbors_matching._check_nodes(
                medrecord=self.medrecord,
                treated_set={"P2", "P10"},
                control_set=control_set,
                essential_covariates=["age", "gender"],
            )
        self.assertEqual(
            str(context.exception),
            "Some treated nodes do not have all the essential covariates",
        )

    def test_invalid_match_controls(self):
        neighbors_matching = NeighborsMatching(number_of_neighbors=1)

        control_set: Set[NodeIndex] = {"P1", "P3", "P5", "P7", "P9"}
        treated_set: Set[NodeIndex] = {"P2", "P4", "P6"}

        with self.assertRaisesRegex(
            AssertionError, "One-hot covariates must be in the essential covariates"
        ):
            neighbors_matching.match_controls(
                medrecord=self.medrecord,
                control_set=control_set,
                treated_set=treated_set,
                patients_group="patients",
                essential_covariates=["age"],
                one_hot_covariates=["gender"],
            )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestNeighborsMatching)
    unittest.TextTestRunner(verbosity=2).run(run_test)
