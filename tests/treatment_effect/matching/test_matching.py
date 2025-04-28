"""Tests for the NeighborsMatching class in the matching module."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, List, Optional, Set

import pandas as pd
import pytest

from medmodels import MedRecord
from medmodels.treatment_effect.matching.neighbors import NeighborsMatching

if TYPE_CHECKING:
    from medmodels.medrecord.types import NodeIndex


def create_patients(patients_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates a patients dataframe.

    Args:
        patients_list (List[NodeIndex]): List of patients to include in the dataframe.

    Returns:
        pd.DataFrame: A patients dataframe.
    """
    patients = pd.DataFrame(
        {
            "index": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
            "age": [20, 30, 40, 30, 40, 50, 60, 70, 80],
            "gender": [
                "male",
                "male",
                "female",
                "female",
                "male",
                "male",
                "female",
                "female",
                "male",
            ],
        }
    )

    return patients.loc[patients["index"].isin(patients_list)]


def create_medrecord(patients_list: Optional[List[NodeIndex]] = None) -> MedRecord:
    """Creates a MedRecord object.

    Args:
        patients_list (Optional[List[NodeIndex]], optional): List of patients to include
            in the MedRecord. Defaults to None.

    Returns:
        MedRecord: A MedRecord object.
    """
    if patients_list is None:
        patients_list = [
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "P9",
        ]
    patients = create_patients(patients_list=patients_list)
    medrecord = MedRecord.from_pandas(nodes=[(patients, "index")])
    medrecord.add_group(group="patients", nodes=patients["index"].to_list())
    medrecord.add_nodes(("P10", {}), "patients")
    return medrecord


class TestNeighborsMatching(unittest.TestCase):
    """Class to test the NeighborsMatching class in the matching module."""

    def setUp(self) -> None:
        self.medrecord = create_medrecord()

    def test_preprocess_data(self) -> None:
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
        assert "age" in data_treated.columns
        assert "age" in data_control.columns
        assert (
            "gender_female" in data_treated.columns
            or "gender_male" in data_treated.columns
        )
        assert (
            "gender_female" in data_control.columns
            or "gender_male" in data_control.columns
        )

        # Assert that the treated and control dataframes have the correct number of rows
        assert len(data_treated) == len(treated_set)
        assert len(data_control) == len(control_set)

        # Try automatic detection of attributes
        data_treated, data_control = neighbors_matching._preprocess_data(
            medrecord=self.medrecord,
            control_set=control_set,
            treated_set=treated_set,
            patients_group="patients",
        )

        # Assert that the treated and control dataframes have the correct columns
        assert "age" in data_treated.columns
        assert "age" in data_control.columns
        assert (
            "gender_female" in data_treated.columns
            or "gender_male" in data_treated.columns
        )
        assert (
            "gender_female" in data_control.columns
            or "gender_male" in data_control.columns
        )

        # Assert that the treated and control dataframes have the correct number of rows
        assert len(data_treated) == len(treated_set)
        assert len(data_control) == len(control_set)

    def test_match_controls(self) -> None:
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
        assert matched_controls.issubset(control_set)

        # Assert that the correct number of controls were matched
        assert len(matched_controls) == len(treated_set)

        # It should do the same if no covariates are given (all attributes assigned)
        matched_controls_no_covariates_specified = neighbors_matching.match_controls(
            medrecord=self.medrecord,
            control_set=control_set,
            treated_set=treated_set,
            patients_group="patients",
        )

        assert matched_controls_no_covariates_specified == matched_controls

    def test_check_nodes(self) -> None:
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
        assert valid_control_set == control_set

    def test_invalid_check_nodes(self) -> None:
        neighbors_matching = NeighborsMatching(number_of_neighbors=1)

        control_set: Set[NodeIndex] = {"P1", "P3", "P5", "P7", "P9"}
        treated_set: Set[NodeIndex] = {"P2", "P4", "P6"}

        # Test insufficient control subjects
        with pytest.raises(
            ValueError,
            match="Not enough control subjects to match the treated subjects. "
            + "Number of controls: 1, Number of treated subjects: 3, "
            + "Number of neighbors required per treated subject: 1, "
            + "Total controls needed: 3.",
        ):
            neighbors_matching._check_nodes(
                medrecord=self.medrecord,
                treated_set=treated_set,
                control_set={"P1"},
                essential_covariates=["age", "gender"],
            )

        neighbors_matching_two_neighbors = NeighborsMatching(number_of_neighbors=2)
        with pytest.raises(
            ValueError,
            match="Not enough control subjects to match the treated subjects. "
            + "Number of controls: 5, Number of treated subjects: 3, "
            + "Number of neighbors required per treated subject: 2, "
            + "Total controls needed: 6.",
        ):
            neighbors_matching_two_neighbors._check_nodes(
                medrecord=self.medrecord,
                treated_set=treated_set,
                control_set=control_set,
                essential_covariates=["age", "gender"],
            )

        # Test missing essential covariates in treated set
        with pytest.raises(
            ValueError,
            match="Some treated nodes do not have all the essential covariates",
        ):
            neighbors_matching._check_nodes(
                medrecord=self.medrecord,
                treated_set={"P2", "P10"},
                control_set=control_set,
                essential_covariates=["age", "gender"],
            )

    def test_invalid_match_controls(self) -> None:
        neighbors_matching = NeighborsMatching(number_of_neighbors=1)

        control_set: Set[NodeIndex] = {"P1", "P3", "P5", "P7", "P9"}
        treated_set: Set[NodeIndex] = {"P2", "P4", "P6"}

        with pytest.raises(
            AssertionError,
            match="One-hot covariates must be in the essential covariates",
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
    pytest.main([__file__])
