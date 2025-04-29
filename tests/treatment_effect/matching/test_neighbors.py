"""Tests for the NeighborsMatching class in the matching module."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Set

import pytest

from medmodels.treatment_effect.matching.neighbors import NeighborsMatching
from tests.treatment_effect.matching.helper import create_medrecord

if TYPE_CHECKING:
    from medmodels.medrecord.types import NodeIndex


class TestNeighborsMatching(unittest.TestCase):
    """Class to test the NeighborsMatching class in the matching module."""

    def setUp(self) -> None:
        self.medrecord = create_medrecord()

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


if __name__ == "__main__":
    pytest.main([__file__])
