"""Module containing the matching abstract class.

Matching is the process of selecting control subjects that are similar to treated
subjects. The class provides the base for the matching algorithms, such as propensity
score matching and nearest neighbor matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
)

import polars as pl

from medmodels.medrecord._overview import extract_attribute_summary
from medmodels.medrecord.medrecord import MedRecord

if TYPE_CHECKING:
    from medmodels.medrecord.medrecord import MedRecord
    from medmodels.medrecord.querying import NodeIndicesOperand, NodeOperand
    from medmodels.medrecord.types import Group, MedRecordAttribute, NodeIndex

MatchingMethod: TypeAlias = Literal["propensity", "nearest_neighbors"]


class Matching(ABC):
    """The Abstract Class for matching."""

    number_of_neighbors: int

    def __init__(self, number_of_neighbors: int) -> None:
        """Initializes the matching class.

        Args:
            number_of_neighbors (int): Number of nearest neighbors to find for each
                treated patient.
        """
        self.number_of_neighbors = number_of_neighbors

    def _preprocess_data(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        patients_group: Group,
        essential_covariates: Optional[List[MedRecordAttribute]] = None,
        one_hot_covariates: Optional[List[MedRecordAttribute]] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Prepared the data for the matching algorithms.

        Args:
            medrecord (MedRecord):  MedRecord object containing the data.
            control_set (Set[NodeIndex]): Set of treated subjects.
            treated_set (Set[NodeIndex]): Set of control subjects.
            patients_group (Group): The group of patients.
            essential_covariates (Optional[List[MedRecordAttribute]], optional):
                Covariates that are essential for matching. Defaults to None, meaning
                all the attributes of the patients are used.
            one_hot_covariates (Optional[List[MedRecordAttribute]], optional):
                Covariates that are one-hot encoded for matching. Defaults to None,
                meaning all the categorical attributes of the patients are used.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Treated and control groups with their
                preprocessed covariates

        Raises:
            AssertionError: If the one-hot covariates are not in the essential
                covariates.
        """
        if essential_covariates is None:
            # If no essential covariates provided, use all attributes of patients group
            nodes_attributes = medrecord.node[medrecord.nodes_in_group(patients_group)]
            essential_covariates = list(
                {key for attributes in nodes_attributes.values() for key in attributes}
            )

        control_set = self._check_nodes(
            medrecord=medrecord,
            treated_set=treated_set,
            control_set=control_set,
            essential_covariates=essential_covariates,
        )

        if "id" not in essential_covariates:
            essential_covariates.append("id")

        # Dataframe with the essential covariates
        data = pl.DataFrame(
            data=[
                {"id": k, **v}
                for k, v in medrecord.node[list(control_set | treated_set)].items()
            ]
        )
        original_columns = data.columns

        # If no one-hot covariates provided, use all categorical attributes of patients
        if one_hot_covariates is None:
            attributes = extract_attribute_summary(
                medrecord.node[medrecord.nodes_in_group(patients_group)]
            )
            one_hot_covariates = [
                covariate
                for covariate, values in attributes.items()
                if "Categorical" in values["type"]
            ]

            one_hot_covariates = [
                covariate
                for covariate in one_hot_covariates
                if covariate in essential_covariates
            ]

        # If there are one-hot covariates, check if all are in the essential covariates
        if (
            not all(
                covariate in essential_covariates for covariate in one_hot_covariates
            )
            and one_hot_covariates
        ):
            msg = "One-hot covariates must be in the essential covariates"
            raise AssertionError(msg)

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
        essential_covariates: List[MedRecordAttribute],
    ) -> Set[NodeIndex]:
        """Check if the treated and control sets are disjoint.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            treated_set (Set[NodeIndex]): Set of treated subjects.
            control_set (Set[NodeIndex]): Set of control subjects.
            essential_covariates (List[MedRecordAttribute]): Covariates that are
                essential for matching.

        Returns:
            Set[NodeIndex]: The control set.

        Raises:
            ValueError: If not enough control subjects to match the treated subjects.
            ValueError: If some treated nodes do not have all the essential covariates.
        """

        def query_essential_covariates(
            node: NodeOperand, patients_set: Set[NodeIndex]
        ) -> NodeIndicesOperand:
            """Query the nodes that have all the essential covariates.

            Returns:
                NodeIndicesOperand: The node indices of the queried node.
            """
            node.has_attribute(essential_covariates)

            node.index().is_in(list(patients_set))

            return node.index()

        control_set = set(
            medrecord.query_nodes(
                lambda node: query_essential_covariates(node, control_set)
            )
        )

        if len(control_set) < self.number_of_neighbors * len(treated_set):
            msg = (
                f"Not enough control subjects to match the treated subjects. "
                f"Number of controls: {len(control_set)}, "
                f"Number of treated subjects: {len(treated_set)}, "
                f"Number of neighbors required per treated subject: {self.number_of_neighbors}, "
                f"Total controls needed: {self.number_of_neighbors * len(treated_set)}."
            )
            raise ValueError(msg)

        if len(treated_set) != len(
            medrecord.query_nodes(
                lambda node: query_essential_covariates(node, treated_set)
            )
        ):
            msg = "Some treated nodes do not have all the essential covariates"
            raise ValueError(msg)

        return control_set

    @abstractmethod
    def match_controls(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        essential_covariates: Optional[Sequence[MedRecordAttribute]] = None,
        one_hot_covariates: Optional[Sequence[MedRecordAttribute]] = None,
    ) -> Set[NodeIndex]:
        """Matches the controls based on the matching algorithm.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            control_set (Set[NodeIndex]): Set of control subjects.
            treated_set (Set[NodeIndex]): Set of treated subjects.
            essential_covariates (Optional[Sequence[MedRecordAttribute]], optional):
                Covariates that are essential for matching. Defaults to None.
            one_hot_covariates (Optional[Sequence[MedRecordAttribute]], optional):
                Covariates that are one-hot encoded for matching. Defaults to None.

        Returns:
            Set[NodeIndex]: Node Ids of the matched controls.
        """
        ...
