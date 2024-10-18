from __future__ import annotations

from typing import Set

from medmodels import MedRecord
from medmodels.medrecord.types import MedRecordAttributeInputList, NodeIndex
from medmodels.treatment_effect.matching.algorithms.classic_distance_models import (
    nearest_neighbor,
)
from medmodels.treatment_effect.matching.matching import Matching


class NeighborsMatching(Matching):
    """Class for the nearest neighbor matching.

    The algorithm finds the nearest neighbors in the control group for each treated
    subject. The essential covariates are used for matching, and the one-hot covariates
    are one-hot encoded. The matched control subjects are saved in the matched_controls
    attribute.
    """

    number_of_neighbors: int

    def __init__(
        self,
        *,
        number_of_neighbors: int = 1,
    ):
        """Initializes the nearest neighbors class.

        Args:
            number_of_neighbors (int, optional): Number of nearest neighbors to find for
                each treated unit. Defaults to 1.
        """
        self.number_of_neighbors = number_of_neighbors

    def match_controls(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        essential_covariates: MedRecordAttributeInputList = ["gender", "age"],
        one_hot_covariates: MedRecordAttributeInputList = ["gender"],
    ) -> Set[NodeIndex]:
        """Matches the controls based on the nearest neighbor algorithm.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            treated_set (Set[NodeIndex]): Set of treated subjects.
            control_set (Set[NodeIndex]): Set of control subjects.
            essential_covariates (MedRecordAttributeInputList, optional): Covariates
                that are essential for matching
            one_hot_covariates (MedRecordAttributeInputList, optional): Covariates that
                are one-hot encoded for matching

        Returns:
            Set[NodeIndex]: Node Ids of the matched controls.
        """
        data_treated, data_control = self._preprocess_data(
            medrecord=medrecord,
            control_set=control_set,
            treated_set=treated_set,
            essential_covariates=essential_covariates,
            one_hot_covariates=one_hot_covariates,
        )

        # Run the algorithm to find the matched controls
        matched_controls = nearest_neighbor(
            data_treated,
            data_control,
            number_of_neighbors=self.number_of_neighbors,
            covariates=[col for col in data_treated.columns if col != "id"],
        )

        return set(matched_controls["id"])
