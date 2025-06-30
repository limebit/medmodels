"""Tests for the Matching class in the matching module.

Since the Matching class is an abstract class, this test file will test the methods
that are common to all matching algorithms.
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Set

import pytest

from medmodels.treatment_effect.matching.neighbors import NeighborsMatching
from tests.treatment_effect.matching.helper import (
    create_medrecord,
    create_medrecord_with_inferred_schema,
)

if TYPE_CHECKING:
    from medmodels.medrecord.types import NodeIndex


class TestMatching(unittest.TestCase):
    """Class to test the NeighborsMatching class in the matching module."""

    def setUp(self) -> None:
        self.medrecord = create_medrecord()

    def test_preprocess_data(self) -> None:
        neighbors_matching = NeighborsMatching(number_of_neighbors=1)

        control_set: Set[NodeIndex] = {"P1", "P3", "P5", "P7", "P9"}
        treated_set: Set[NodeIndex] = {"P2", "P4", "P6"}

        # Preprocess the data with default values
        data_treated, data_control = neighbors_matching._preprocess_data(
            medrecord=self.medrecord,
            control_set=control_set,
            treated_set=treated_set,
            patients_group="patients",
        )

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
        # Assert that the treated and control dataframes have the correct number of rows
        assert len(data_treated) == len(treated_set)
        assert len(data_control) == len(control_set)

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
        # Assert that the treated and control dataframes have the correct number of rows
        assert len(data_treated) == len(treated_set)
        assert len(data_control) == len(control_set)

    def test_invalid_preprocess_data(self) -> None:
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

        # The inferred schema does not recognize "gender" as a categorical variable.
        # Thus, it will not be able to one-hot encode it.
        # This will raise an error (values in the column are not numeric).
        medrecord_2 = create_medrecord_with_inferred_schema()

        with pytest.raises(
            ValueError,
            match="All covariates must be numeric",
        ):
            neighbors_matching.match_controls(
                medrecord=medrecord_2,
                control_set=control_set,
                treated_set=treated_set,
                patients_group="patients",
            )

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


if __name__ == "__main__":
    pytest.main([__file__])
