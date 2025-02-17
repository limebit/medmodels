"""Module for the nearest neighbor matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Set

from medmodels.treatment_effect.matching.algorithms.classic_distance_models import (
    nearest_neighbor,
)
from medmodels.treatment_effect.matching.matching import Matching

if TYPE_CHECKING:
    from medmodels import MedRecord
    from medmodels.medrecord.types import Group, MedRecordAttribute, NodeIndex


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
    ) -> None:
        """Initializes the nearest neighbors class.

        Args:
            number_of_neighbors (int, optional): Number of nearest neighbors to find for
                each treated unit. Defaults to 1.
        """
        super().__init__(number_of_neighbors)

    def match_controls(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        patients_group: Group,
        essential_covariates: Optional[Sequence[MedRecordAttribute]] = None,
        one_hot_covariates: Optional[Sequence[MedRecordAttribute]] = None,
    ) -> Set[NodeIndex]:
        """Matches the controls based on the nearest neighbor algorithm.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            control_set (Set[NodeIndex]): Set of control subjects.
            treated_set (Set[NodeIndex]): Set of treated subjects.
            patients_group (Group): Group of patients in the MedRecord.
            essential_covariates (Optional[Sequence[MedRecordAttribute]], optional):
                Covariates that are essential for matching. Defaults to None, meaning
                all the attributes of the patients are used.
            one_hot_covariates (Optional[Sequence[MedRecordAttribute]], optional):
                Covariates that are one-hot encoded for matching. Defaults to None,
                meaning all the categorical attributes of the patients are used.

        Returns:
            Set[NodeIndex]: Node Ids of the matched controls.
        """
        data_treated, data_control = self._preprocess_data(
            medrecord=medrecord,
            control_set=control_set,
            treated_set=treated_set,
            patients_group=patients_group,
            essential_covariates=list(essential_covariates)
            if essential_covariates
            else None,
            one_hot_covariates=list(one_hot_covariates) if one_hot_covariates else None,
        )

        # Run the algorithm to find the matched controls
        matched_controls = nearest_neighbor(
            data_treated,
            data_control,
            number_of_neighbors=self.number_of_neighbors,
            covariates=[col for col in data_treated.columns if col != "id"],
        )

        return set(matched_controls["id"])
