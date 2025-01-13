"""Module containing the matching abstract class.

Matching is the process of selecting control subjects that are similar to treated
subjects. The class provides the base for the matching algorithms, such as propensity
score matching and nearest neighbor matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional, Set, Tuple

import polars as pl

if TYPE_CHECKING:
    import sys

    from medmodels.medrecord.medrecord import MedRecord
    from medmodels.medrecord.types import MedRecordAttributeInputList, NodeIndex

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

MatchingMethod: TypeAlias = Literal["propensity", "nearest_neighbors"]


class Matching(ABC):
    """The Abstract Class for matching."""

    def _preprocess_data(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        essential_covariates: MedRecordAttributeInputList,
        one_hot_covariates: MedRecordAttributeInputList,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Prepared the data for the matching algorithms.

        Args:
            medrecord (MedRecord):  MedRecord object containing the data.
            control_set (Set[NodeIndex]): Set of treated subjects.
            treated_set (Set[NodeIndex]): Set of control subjects.
            essential_covariates (MedRecordAttributeInputList):  Covariates
                that are essential for matching
            one_hot_covariates (MedRecordAttributeInputList): Covariates that
                are one-hot encoded for matching

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Treated and control groups with their
                preprocessed covariates
        """
        essential_covariates = [str(covariate) for covariate in essential_covariates]

        if "id" not in essential_covariates:
            essential_covariates.append("id")

        # Dataframe
        data = pl.DataFrame(
            data=[
                {"id": k, **v}
                for k, v in medrecord.node[list(control_set | treated_set)].items()
            ]
        )
        original_columns = data.columns

        # One-hot encode the categorical variables
        data = data.to_dummies(
            columns=[str(covariate) for covariate in one_hot_covariates],
            drop_first=True,
        )
        new_columns = [col for col in data.columns if col not in original_columns]

        # Add to essential covariates the new columns created by one-hot encoding and
        # delete the ones that were one-hot encoded
        essential_covariates.extend(new_columns)
        [essential_covariates.remove(col) for col in one_hot_covariates]
        data = data.select(essential_covariates)

        # Select the sets of treated and control subjects
        data_treated = data.filter(pl.col("id").is_in(treated_set))
        data_control = data.filter(pl.col("id").is_in(control_set))

        return data_treated, data_control

    @abstractmethod
    def match_controls(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        essential_covariates: Optional[MedRecordAttributeInputList] = None,
        one_hot_covariates: Optional[MedRecordAttributeInputList] = None,
    ) -> Set[NodeIndex]:
        """Matches the controls based on the matching algorithm.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            control_set (Set[NodeIndex]): Set of control subjects.
            treated_set (Set[NodeIndex]): Set of treated subjects.
            essential_covariates (Optional[MedRecordAttributeInputList], optional):
                Covariates that are essential for matching. Defaults to None.
            one_hot_covariates (Optional[MedRecordAttributeInputList], optional):
                Covariates that are one-hot encoded for matching. Defaults to None.

        Returns:
            Set[NodeIndex]: Node Ids of the matched controls.
        """
        ...
