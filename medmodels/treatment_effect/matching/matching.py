from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Literal, Set, Tuple

import polars as pl

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import MedRecordAttributeInputList, NodeIndex

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

MatchingMethod: TypeAlias = Literal["propensity", "nearest_neighbors"]


class Matching(metaclass=ABCMeta):
    """The Base Class for matching."""

    def _preprocess_data(
        self,
        *,
        medrecord: MedRecord,
        control_group: Set[NodeIndex],
        treated_group: Set[NodeIndex],
        essential_covariates: MedRecordAttributeInputList,
        one_hot_covariates: MedRecordAttributeInputList,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Prepared the data for the matching algorithms.

        Args:
            medrecord (MedRecord):  MedRecord object containing the data.
            control_group (Set[NodeIndex]): Set of treated subjects.
            treated_group (Set[NodeIndex]): Set of control subjects.
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
                for k, v in medrecord.node[list(control_group | treated_group)].items()
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
        data_treated = data.filter(pl.col("id").is_in(treated_group))
        data_control = data.filter(pl.col("id").is_in(control_group))

        return data_treated, data_control

    @abstractmethod
    def match_controls(
        self,
        *,
        control_group: Set[NodeIndex],
        treated_group: Set[NodeIndex],
        medrecord: MedRecord,
        essential_covariates: MedRecordAttributeInputList = ["gender", "age"],
        one_hot_covariates: MedRecordAttributeInputList = ["gender"],
    ) -> Set[NodeIndex]: ...
