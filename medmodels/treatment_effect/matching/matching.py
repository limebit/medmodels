from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional, Set, Tuple

import polars as pl

from medmodels.medrecord._overview import extract_attribute_summary
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import NodeOperand
from medmodels.medrecord.types import Group, MedRecordAttributeInputList, NodeIndex

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

MatchingMethod: TypeAlias = Literal["propensity", "nearest_neighbors"]


class Matching(metaclass=ABCMeta):
    """The Base Class for matching."""

    number_of_neighbors: int

    def __init__(self, number_of_neighbors: int) -> None:
        """Initializes the matching class.

        Args:
            number_of_neighbors (int): Number of nearest neighbors to find for each treated unit.
        """
        self.number_of_neighbors = number_of_neighbors

    def _preprocess_data(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        patients_group: Group,
        essential_covariates: Optional[MedRecordAttributeInputList] = None,
        one_hot_covariates: Optional[MedRecordAttributeInputList] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Prepared the data for the matching algorithms.

        Args:
            medrecord (MedRecord):  MedRecord object containing the data.
            control_set (Set[NodeIndex]): Set of treated subjects.
            treated_set (Set[NodeIndex]): Set of control subjects.
            patients_group (Group): The group of patients.
            essential_covariates (Optional[MedRecordAttributeInputList]):
                Covariates that are essential for matching. Defaults to None.
            one_hot_covariates (Optional[MedRecordAttributeInputList]):
                Covariates that are one-hot encoded for matching. Defaults to None.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Treated and control groups with their
                preprocessed covariates

        Raises:
            ValueError: If not enough control subjects to match the treated subjects.
            ValueError: If some treated nodes do not have all the essential covariates.
            AssertionError: If the one-hot covariates are not in the essential covariates.
        """
        if essential_covariates is None:
            # If no essential covariates are provided, use all the attributes of the patients
            essential_covariates = list(
                extract_attribute_summary(
                    medrecord.node[medrecord.nodes_in_group(patients_group)]
                )
            )
        else:
            essential_covariates = [covariate for covariate in essential_covariates]

        control_set = self._check_nodes(
            medrecord=medrecord,
            treated_set=treated_set,
            control_set=control_set,
            essential_covariates=essential_covariates,
        )

        if "id" not in essential_covariates:
            essential_covariates.append("id")

        # Dataframe wth the essential covariates
        data = pl.DataFrame(
            data=[
                {"id": k, **v}
                for k, v in medrecord.node[list(control_set | treated_set)].items()
            ]
        )
        original_columns = data.columns

        if one_hot_covariates is None:
            # If no one-hot covariates are provided, use all the categorical attributes of the patients
            attributes = extract_attribute_summary(
                medrecord.node[medrecord.nodes_in_group(patients_group)]
            )
            one_hot_covariates = [
                covariate
                for covariate, values in attributes.items()
                if "values" in values
            ]

        if not all(
            covariate in essential_covariates for covariate in one_hot_covariates
        ):
            raise AssertionError(
                "One-hot covariates must be in the essential covariates"
            )

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

    def _check_nodes(
        self,
        medrecord: MedRecord,
        treated_set: Set[NodeIndex],
        control_set: Set[NodeIndex],
        essential_covariates: MedRecordAttributeInputList,
    ) -> Set[NodeIndex]:
        """Check if the treated and control sets are disjoint.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            treated_set (Set[NodeIndex]): Set of treated subjects.
            control_set (Set[NodeIndex]): Set of control subjects.
            essential_covariates (MedRecordAttributeInputList): Covariates that are
                essential for matching.

        Returns:
            Set[NodeIndex]: The control set.

        Raises:
            ValueError: If not enough control subjects to match the treated subjects.
        """

        def query_essential_covariates(
            node: NodeOperand, patients_set: Set[NodeIndex]
        ) -> None:
            """Query the nodes that have all the essential covariates."""
            for attribute in essential_covariates:
                node.has_attribute(attribute)

            node.index().is_in(list(patients_set))

        control_set = set(
            medrecord.select_nodes(
                lambda node: query_essential_covariates(node, control_set)
            )
        )
        if len(control_set) < self.number_of_neighbors * len(treated_set):
            raise ValueError(
                "Not enough control subjects to match the treated subjects"
            )

        if len(treated_set) != len(
            medrecord.select_nodes(
                lambda node: query_essential_covariates(node, treated_set)
            )
        ):
            raise ValueError(
                "Some treated nodes do not have all the essential covariates"
            )

        return control_set

    @abstractmethod
    def match_controls(
        self,
        *,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        medrecord: MedRecord,
        essential_covariates: Optional[MedRecordAttributeInputList] = None,
        one_hot_covariates: Optional[MedRecordAttributeInputList] = None,
    ) -> Set[NodeIndex]: ...
